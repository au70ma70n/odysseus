"""Helpers for ComfyUI MCP image generation: poll jobs to completion and deliver images."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.constants import GENERATED_IMAGES_DIR

logger = logging.getLogger(__name__)

# comfyui-mcp generate_* tools enqueue work and return immediately; Odysseus must
# poll until done and fetch the output image for the chat UI.
_COMFYUI_GEN_TOOLS = frozenset({
    "generate_image",
    "generate_with_controlnet",
    "generate_with_ip_adapter",
    "generate_with_api_node",
    "regenerate",
    "generate_audio",
})

_POLL_INTERVAL_SEC = float(os.environ.get("ODYSSEUS_COMFYUI_POLL_INTERVAL_SEC", "2"))
_POLL_TIMEOUT_SEC = float(os.environ.get("ODYSSEUS_COMFYUI_POLL_TIMEOUT_SEC", "1800"))

# comfyui-mcp txt2img defaults target SD/SDXL (cfg 8, scheduler normal). Flux Dev needs
# lower CFG and the simple scheduler or outputs look soft/overprocessed.
_FLUX_DEFAULTS: Dict[str, Any] = {
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "guidance": 3.5,
    "sampler": "euler",
    "scheduler": "simple",
}

_FLUX_UNET = os.environ.get("ODYSSEUS_FLUX_UNET", "flux1-dev-fp8.safetensors")
_FLUX_CLIP_L = os.environ.get("ODYSSEUS_FLUX_CLIP_L", "clip_l.safetensors")
_FLUX_T5 = os.environ.get("ODYSSEUS_FLUX_T5", "t5xxl_fp8_e4m3fn.safetensors")
_FLUX_VAE = os.environ.get("ODYSSEUS_FLUX_VAE", "ae.safetensors")
_COMFYUI_URL = os.environ.get("ODYSSEUS_COMFYUI_URL", "http://server.lan:8188").rstrip("/")


def _uses_flux_checkpoint(checkpoint: str) -> bool:
    ckpt = (checkpoint or "").strip().lower()
    if ckpt:
        return "flux" in ckpt
    return os.environ.get("ODYSSEUS_COMFYUI_ASSUME_FLUX", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def apply_flux_generation_defaults(args: Dict[str, Any]) -> Dict[str, Any]:
    """Backfill Flux-friendly sampling params without overriding explicit caller values."""
    checkpoint = str(args.get("checkpoint") or "").strip()
    if not _uses_flux_checkpoint(checkpoint):
        return args
    out = dict(args)
    for key, value in _FLUX_DEFAULTS.items():
        out.setdefault(key, value)
    if not checkpoint:
        out.setdefault("checkpoint", "flux_dev.safetensors")
    return out


def build_flux_txt2img_workflow(
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 3.5,
    seed: Optional[int] = None,
    sampler: str = "euler",
    scheduler: str = "simple",
    unet_name: str = _FLUX_UNET,
    clip_l_name: str = _FLUX_CLIP_L,
    t5_name: str = _FLUX_T5,
    vae_name: str = _FLUX_VAE,
) -> Dict[str, Any]:
    """ComfyUI API workflow for Flux Dev (UNET + dual CLIP + VAE, KSampler cfg=1.0)."""
    if seed is None:
        seed = random.randint(0, 2**48 - 1)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": clip_l_name,
                "clip_name2": t5_name,
                "type": "flux",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": prompt,
                "t5xxl": prompt,
                "guidance": guidance,
            },
            "_meta": {"title": "Positive Prompt"},
        },
        "5": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": negative_prompt,
                "t5xxl": negative_prompt,
                "guidance": guidance,
            },
            "_meta": {"title": "Negative Prompt"},
        },
        "6": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["4", 0], "guidance": guidance},
        },
        "7": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]},
        },
        "8": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["3", 0]},
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {"images": ["10", 0], "filename_prefix": "ComfyUI"},
        },
    }


async def _list_comfy_models(folder: str) -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{_COMFYUI_URL}/models/{folder}")
            if response.status_code != 200:
                return []
            data = response.json()
            return list(data) if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Failed to list ComfyUI models/%s: %s", folder, e)
        return []


async def flux_stack_status() -> Tuple[bool, List[str]]:
    """Return whether the Flux component files are present and any missing names."""
    diffusion, text_enc, vae = await asyncio.gather(
        _list_comfy_models("diffusion_models"),
        _list_comfy_models("text_encoders"),
        _list_comfy_models("vae"),
    )
    # ComfyUI may not expose /models/clip; text_encoders covers both clip_l and t5
    required = {
        f"diffusion_models/{_FLUX_UNET}": _FLUX_UNET in diffusion,
        f"text_encoders/{_FLUX_CLIP_L}": _FLUX_CLIP_L in text_enc,
        f"text_encoders/{_FLUX_T5}": _FLUX_T5 in text_enc,
        f"vae/{_FLUX_VAE}": _FLUX_VAE in vae,
    }
    missing = [name for name, ok in required.items() if not ok]
    return not missing, missing


def is_comfyui_generation_tool(qualified_name: str) -> bool:
    parts = qualified_name.split("__", 2)
    return len(parts) == 3 and parts[0] == "mcp" and parts[2] in _COMFYUI_GEN_TOOLS


def _server_tool(server_id: str, tool_name: str) -> str:
    return f"mcp__{server_id}__{tool_name}"


def _parse_prompt_id(stdout: str) -> Optional[str]:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("prompt_id"):
            return str(data["prompt_id"])
    except json.JSONDecodeError:
        pass
    match = re.search(r'"prompt_id"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else None


def _parse_output_image_path(history_text: str) -> Optional[Tuple[str, str]]:
    """Return (subfolder, filename) from get_history markdown output."""
    match = re.search(r"images\s*→\s*\*\*([^*]+)\*\*", history_text or "")
    if not match:
        return None
    raw = match.group(1).strip()
    if "/" in raw:
        subfolder, filename = raw.rsplit("/", 1)
        return subfolder, filename
    return "", raw


def _save_generated_image(images: list, *, prompt: str, model: str, size: str = "") -> Optional[str]:
    if not images:
        return None
    img = images[0] if isinstance(images[0], dict) else {}
    data = img.get("data")
    if not data:
        return None
    mime = img.get("mimeType") or img.get("mime_type") or "image/png"
    ext = ".jpg" if "jpeg" in mime or "jpg" in mime else ".png"
    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    out_dir = Path(GENERATED_IMAGES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(data) if isinstance(data, str) else data
    (out_dir / filename).write_bytes(raw)
    try:
        from src.database import GalleryImage, SessionLocal

        db = SessionLocal()
        try:
            db.add(GalleryImage(
                id=str(uuid.uuid4()),
                filename=filename,
                prompt=prompt[:500],
                model=model[:120],
                size=size or "",
                quality="",
            ))
            db.commit()
        finally:
            db.close()
    except Exception:
        pass
    return f"/api/generated-image/{filename}"


def _format_success_stdout(*, prompt: str, image_url: str, model: str, size: str = "") -> str:
    lines = [
        f"Generated image for: {prompt[:100]}",
        f"Direct link: {image_url}",
        f"model: {model}",
    ]
    if size:
        lines.append(f"size: {size}")
    return "\n".join(lines)


async def comfyui_generate_and_deliver(mcp, qualified_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a ComfyUI generate tool, poll to completion, fetch image, return image_url."""
    parts = qualified_name.split("__", 2)
    server_id = parts[1]
    tool_name = parts[2]

    gen_args = apply_flux_generation_defaults(dict(args))
    prompt = str(gen_args.get("prompt") or gen_args.get("positive_prompt") or "image").strip()
    negative = str(gen_args.get("negative_prompt") or "").strip()
    checkpoint = str(gen_args.get("checkpoint") or "").strip()
    width = int(gen_args.get("width") or _FLUX_DEFAULTS["width"])
    height = int(gen_args.get("height") or _FLUX_DEFAULTS["height"])
    size = f"{width}x{height}"

    use_flux_workflow = (
        tool_name == "generate_image"
        and _uses_flux_checkpoint(checkpoint)
        and os.environ.get("ODYSSEUS_COMFYUI_FLUX_WORKFLOW", "1").strip().lower()
        not in ("0", "false", "no")
    )
    if use_flux_workflow:
        try:
            ready, missing = await asyncio.wait_for(flux_stack_status(), timeout=30)
        except Exception as e:
            logger.warning("flux_stack_status() failed, falling back to generate_image: %s", e)
            ready, missing = False, ["status check failed"]
        if not ready:
            logger.info("Flux stack not ready (%s), falling back to generate_image MCP tool", missing)
            enqueue = await mcp.call_tool(qualified_name, gen_args)
        else:
            workflow = build_flux_txt2img_workflow(
                prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                steps=int(gen_args.get("steps") or _FLUX_DEFAULTS["steps"]),
                guidance=float(gen_args.get("guidance") or gen_args.get("cfg") or _FLUX_DEFAULTS["guidance"]),
                seed=gen_args.get("seed"),
                sampler=str(gen_args.get("sampler") or gen_args.get("sampler_name") or _FLUX_DEFAULTS["sampler"]),
                scheduler=str(gen_args.get("scheduler") or _FLUX_DEFAULTS["scheduler"]),
            )
            enqueue = await mcp.call_tool(
                _server_tool(server_id, "enqueue_workflow"),
                {"workflow": workflow},
            )
            if enqueue.get("exit_code") not in (0, None):
                logger.warning("enqueue_workflow failed (%s), retrying with generate_image", enqueue.get("error"))
                enqueue = await mcp.call_tool(qualified_name, gen_args)
            else:
                checkpoint = _FLUX_UNET
    else:
        enqueue = await mcp.call_tool(qualified_name, gen_args)
    if enqueue.get("exit_code") not in (0, None):
        return enqueue

    prompt_id = _parse_prompt_id(enqueue.get("stdout") or "")
    if not prompt_id:
        return {
            "error": "ComfyUI did not return a prompt_id; cannot poll for completion.",
            "stderr": enqueue.get("stdout") or "",
            "exit_code": 1,
        }

    deadline = time.monotonic() + _POLL_TIMEOUT_SEC
    status_data: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        status_result = await mcp.call_tool(
            _server_tool(server_id, "get_job_status"),
            {"prompt_id": prompt_id},
        )
        if status_result.get("exit_code") not in (0, None):
            return status_result
        try:
            status_data = json.loads(status_result.get("stdout") or "{}")
        except json.JSONDecodeError:
            status_data = {}
        if status_data.get("done"):
            if status_data.get("status_str") not in (None, "success"):
                err = status_data.get("error") or status_data.get("status_str") or "generation failed"
                return {"error": f"ComfyUI generation failed: {err}", "exit_code": 1}
            break
        await asyncio.sleep(_POLL_INTERVAL_SEC)
    else:
        return {
            "error": f"ComfyUI generation timed out after {_POLL_TIMEOUT_SEC:.0f}s (prompt_id={prompt_id})",
            "exit_code": 1,
        }

    history_result = await mcp.call_tool(
        _server_tool(server_id, "get_history"),
        {"prompt_id": prompt_id},
    )
    if history_result.get("exit_code") not in (0, None):
        return history_result

    image_path = _parse_output_image_path(history_result.get("stdout") or "")
    if not image_path:
        return {
            "error": "ComfyUI job finished but no output image was found in history.",
            "stderr": history_result.get("stdout") or "",
            "exit_code": 1,
        }
    subfolder, filename = image_path

    image_result = await mcp.call_tool(
        _server_tool(server_id, "get_image"),
        {"filename": filename, "subfolder": subfolder, "type": "output"},
    )
    if image_result.get("exit_code") not in (0, None):
        return image_result

    model_label = checkpoint or tool_name
    if not checkpoint:
        try:
            enqueue_data = json.loads(enqueue.get("stdout") or "{}")
            model_label = enqueue_data.get("checkpoint") or model_label
        except json.JSONDecodeError:
            pass

    image_url = _save_generated_image(
        image_result.get("images") or [],
        prompt=prompt,
        model=str(model_label),
        size=size,
    )
    if not image_url:
        return {
            "error": "Fetched ComfyUI image but failed to save it for display.",
            "stderr": image_result.get("stdout") or "",
            "exit_code": 1,
        }

    stdout = _format_success_stdout(
        prompt=prompt,
        image_url=image_url,
        model=str(model_label),
        size=size,
    )
    return {
        "stdout": stdout,
        "stderr": "",
        "exit_code": 0,
        "image_url": image_url,
        "image_prompt": prompt[:100],
        "image_model": str(model_label),
        "image_size": size,
    }
