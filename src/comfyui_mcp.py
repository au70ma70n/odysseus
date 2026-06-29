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
    "edit_image",
    "generate_with_controlnet",
    "generate_with_ip_adapter",
    "generate_with_api_node",
    "regenerate",
    "generate_audio",
    "enqueue_workflow",
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
_FLUX_KONTEXT_UNET = os.environ.get(
    "ODYSSEUS_FLUX_KONTEXT_UNET", "flux1-dev-kontext_fp8_scaled.safetensors"
)
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


def parse_lora_from_args(args: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Extract (lora_filename, strength) from generate_image-style args, if any."""
    loras = args.get("loras")
    if isinstance(loras, list) and loras:
        first = loras[0]
        if isinstance(first, str) and first.strip():
            strength = 0.8
            if len(loras) > 1 and isinstance(loras[1], (int, float)):
                strength = float(loras[1])
            return first.strip(), strength
        if isinstance(first, dict):
            name = str(first.get("name") or first.get("lora_name") or first.get("lora") or "").strip()
            if name:
                strength = float(first.get("strength") or first.get("strength_model") or 0.8)
                return name, strength

    lora_name = str(args.get("lora") or args.get("lora_name") or "").strip()
    if not lora_name:
        return None
    strength = float(args.get("lora_strength") or args.get("strength") or 0.8)
    return lora_name, strength


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
    lora_name: str = "",
    lora_strength: float = 0.8,
) -> Dict[str, Any]:
    """ComfyUI API workflow for Flux Dev (UNET + dual CLIP + VAE, KSampler cfg=1.0)."""
    if seed is None:
        seed = random.randint(0, 2**48 - 1)
    model_source: List[Any] = ["1", 0]
    workflow: Dict[str, Any] = {
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
    }
    if lora_name.strip():
        workflow["12"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora_name.strip(),
                "strength_model": lora_strength,
            },
        }
        model_source = ["12", 0]
    workflow.update({
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": model_source,
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
    })
    return workflow


def _build_source_load_node(image_name: str, image_folder: str = "output") -> Dict[str, Any]:
    """Load a prior ComfyUI output via LoadImageOutput (requires `` [output]`` suffix)."""
    raw = re.sub(r"\s*\[(output|input|temp)\]\s*$", "", (image_name or "").strip(), flags=re.I)
    if (image_folder or "output").lower() == "output":
        return {
            "class_type": "LoadImageOutput",
            "inputs": {"image": f"{raw} [output]"},
            "_meta": {"title": "Load Source (output)"},
        }
    return {
        "class_type": "LoadImage",
        "inputs": {"image": raw},
        "_meta": {"title": "Load Source (input)"},
    }


def build_flux_img2img_workflow(
    prompt: str,
    *,
    init_image: str,
    image_folder: str = "output",
    negative_prompt: str = "",
    denoise: float = 0.6,
    steps: int = 20,
    guidance: float = 3.5,
    seed: Optional[int] = None,
    sampler: str = "euler",
    scheduler: str = "simple",
    unet_name: str = _FLUX_UNET,
    clip_l_name: str = _FLUX_CLIP_L,
    t5_name: str = _FLUX_T5,
    vae_name: str = _FLUX_VAE,
    lora_name: str = "",
    lora_strength: float = 0.8,
) -> Dict[str, Any]:
    """FLUX.1-dev img2img: denoise an existing engine image with a new prompt."""
    if seed is None:
        seed = random.randint(0, 2**48 - 1)
    model_source: List[Any] = ["1", 0]
    workflow: Dict[str, Any] = {
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
        "4": _build_source_load_node(init_image, image_folder),
        "5": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["4", 0], "vae": ["3", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": prompt,
                "t5xxl": prompt,
                "guidance": guidance,
            },
            "_meta": {"title": "Positive Prompt"},
        },
        "7": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": negative_prompt,
                "t5xxl": negative_prompt,
                "guidance": guidance,
            },
            "_meta": {"title": "Negative Prompt"},
        },
        "8": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["6", 0], "guidance": guidance},
        },
        "9": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["7", 0]},
        },
    }
    if lora_name.strip():
        workflow["12"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora_name.strip(),
                "strength_model": lora_strength,
            },
        }
        model_source = ["12", 0]
    workflow.update({
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": model_source,
                "positive": ["8", 0],
                "negative": ["9", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["3", 0]},
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["11", 0], "filename_prefix": "ComfyUI_edit"},
        },
    })
    return workflow


def build_flux_kontext_workflow(
    prompt: str,
    *,
    init_image: str,
    image_folder: str = "output",
    steps: int = 20,
    guidance: float = 2.5,
    seed: Optional[int] = None,
    sampler: str = "euler",
    scheduler: str = "simple",
    unet_name: str = _FLUX_KONTEXT_UNET,
    clip_l_name: str = _FLUX_CLIP_L,
    t5_name: str = _FLUX_T5,
    vae_name: str = _FLUX_VAE,
) -> Dict[str, Any]:
    """FLUX.1 Kontext instruction edit — change only what the prompt asks."""
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
        "4": _build_source_load_node(init_image, image_folder),
        "5": {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": ["4", 0]},
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
            "_meta": {"title": "Edit Instruction"},
        },
        "8": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["7", 0], "guidance": guidance},
        },
        "9": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["8", 0], "latent": ["6", 0]},
        },
        "10": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["7", 0]},
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["9", 0],
                "negative": ["10", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": "ComfyUI_edit"},
        },
    }


COMFYUI_EDIT_IMAGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "Edit instruction (kontext) or full target prompt (img2img), e.g. 'change her hair to black'",
        },
        "init_image": {
            "type": "string",
            "description": "Source filename on the ComfyUI engine (e.g. ComfyUI_00025_.png from get_history)",
        },
        "mode": {
            "type": "string",
            "enum": ["kontext", "img2img"],
            "description": "kontext = targeted instruction edit (default); img2img = denoise redraw",
        },
        "denoise": {
            "type": "number",
            "description": "img2img only: 0.4-0.7 (lower keeps more of the original)",
        },
        "image_folder": {
            "type": "string",
            "enum": ["output", "input"],
            "description": "Where init_image lives (default output)",
        },
        "guidance": {"type": "number"},
        "steps": {"type": "integer"},
        "seed": {"type": "integer"},
        "sampler": {"type": "string"},
        "scheduler": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "checkpoint": {"type": "string"},
        "lora_name": {"type": "string"},
        "lora_strength": {"type": "number"},
    },
    "required": ["prompt", "init_image"],
}

COMFYUI_EDIT_IMAGE_DESCRIPTION = (
    "Make a TARGETED change to an EXISTING ComfyUI image (not a fresh generation). "
    "Use when the user says 'change/edit/make her hair black', 'add sunglasses', etc. "
    "`init_image` is the engine output filename from get_history (e.g. ComfyUI_00025_.png) — "
    "NOT a gallery asset_id. mode 'kontext' (default) keeps everything else identical; "
    "mode 'img2img' redraws at a denoise level (no Kontext model required)."
)


def comfyui_virtual_tools(server_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Odysseus-native tools backed by comfyui_mcp.py (not in vanilla comfyui-mcp npm)."""
    names = {t.get("name") for t in server_tools}
    if "edit_image" in names or not (
        "generate_image" in names or "enqueue_workflow" in names
    ):
        return []
    return [{
        "name": "edit_image",
        "description": COMFYUI_EDIT_IMAGE_DESCRIPTION,
        "input_schema": COMFYUI_EDIT_IMAGE_SCHEMA,
    }]


async def kontext_model_available() -> bool:
    diffusion = await _list_comfy_models("diffusion_models")
    return _FLUX_KONTEXT_UNET in diffusion


async def _resolve_init_image(mcp: Any, server_id: str, init_image: str) -> str:
    """Normalize init_image — resolve comfyui-mcp asset_ids to filenames."""
    raw = (init_image or "").strip()
    if not raw:
        return raw
    if re.search(r"\.(png|jpe?g|webp|gif)$", raw, re.I):
        return re.sub(r"\s*\[(output|input|temp)\]\s*$", "", raw, flags=re.I)
    if raw.startswith("a_"):
        try:
            assets_result = await mcp.call_tool(
                _server_tool(server_id, "list_assets"),
                {"limit": 100},
            )
            if assets_result.get("exit_code") in (0, None):
                stdout = assets_result.get("stdout") or ""
                try:
                    data = json.loads(stdout)
                except json.JSONDecodeError:
                    data = {}
                for asset in data.get("assets") or []:
                    if isinstance(asset, dict) and asset.get("asset_id") == raw:
                        fn = str(asset.get("filename") or "").strip()
                        if fn:
                            return fn
        except Exception as e:
            logger.warning("Failed to resolve asset_id %s: %s", raw, e)
    return raw


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


def _prompt_from_workflow(workflow: Any) -> str:
    if not isinstance(workflow, dict):
        return ""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") not in ("CLIPTextEncodeFlux", "CLIPTextEncode"):
            continue
        inputs = node.get("inputs") or {}
        if inputs.get("clip_l"):
            return str(inputs["clip_l"]).strip()
        if inputs.get("text"):
            return str(inputs["text"]).strip()
    return ""


async def comfyui_generate_and_deliver(mcp, qualified_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a ComfyUI generate/edit tool, poll to completion, fetch image, return image_url."""
    parts = qualified_name.split("__", 2)
    server_id = parts[1]
    tool_name = parts[2]

    gen_args = dict(args)
    if tool_name not in ("enqueue_workflow", "edit_image"):
        gen_args = apply_flux_generation_defaults(gen_args)
    prompt = str(gen_args.get("prompt") or gen_args.get("positive_prompt") or "image").strip()
    negative = str(gen_args.get("negative_prompt") or "").strip()
    checkpoint = str(gen_args.get("checkpoint") or "").strip()
    width = int(gen_args.get("width") or _FLUX_DEFAULTS["width"])
    height = int(gen_args.get("height") or _FLUX_DEFAULTS["height"])
    size = f"{width}x{height}"
    lora_spec = parse_lora_from_args(gen_args)
    enqueue: Optional[Dict[str, Any]] = None

    if tool_name == "edit_image":
        init_image = await _resolve_init_image(
            mcp, server_id, str(gen_args.get("init_image") or "").strip()
        )
        if not init_image:
            return {
                "error": (
                    "edit_image requires init_image — the ComfyUI output filename from "
                    "get_history (e.g. ComfyUI_00025_.png), not a gallery asset_id."
                ),
                "exit_code": 1,
            }
        prompt = str(gen_args.get("prompt") or prompt or "").strip()
        if not prompt or prompt == "image":
            return {"error": "edit_image requires prompt (the edit instruction)", "exit_code": 1}

        mode = str(gen_args.get("mode") or "kontext").lower()
        image_folder = str(gen_args.get("image_folder") or "output")
        steps = int(gen_args.get("steps") or 20)
        sampler = str(
            gen_args.get("sampler") or gen_args.get("sampler_name") or "euler"
        )
        scheduler = str(gen_args.get("scheduler") or "simple")
        seed = gen_args.get("seed")
        workflow: Optional[Dict[str, Any]] = None

        if mode == "kontext":
            try:
                has_kontext = await asyncio.wait_for(kontext_model_available(), timeout=15)
            except Exception:
                has_kontext = False
            if has_kontext:
                guidance = float(gen_args.get("guidance") or 2.5)
                unet = checkpoint or _FLUX_KONTEXT_UNET
                workflow = build_flux_kontext_workflow(
                    prompt,
                    init_image=init_image,
                    image_folder=image_folder,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    sampler=sampler,
                    scheduler=scheduler,
                    unet_name=unet,
                )
                checkpoint = unet
            else:
                logger.info(
                    "Kontext model %s not installed; falling back to img2img edit",
                    _FLUX_KONTEXT_UNET,
                )
                mode = "img2img"

        if mode == "img2img":
            try:
                ready, missing = await asyncio.wait_for(flux_stack_status(), timeout=30)
            except Exception as e:
                logger.warning("flux_stack_status() failed for edit_image: %s", e)
                ready, missing = False, ["status check failed"]
            if not ready:
                return {
                    "error": (
                        "Cannot edit image: Flux stack not ready"
                        + (f" (missing {', '.join(missing)})" if missing else "")
                        + (
                            f". Install {_FLUX_KONTEXT_UNET} for kontext edits, or ensure "
                            f"{_FLUX_UNET} is present for img2img fallback."
                        )
                    ),
                    "exit_code": 1,
                }
            denoise = float(gen_args.get("denoise") or 0.6)
            guidance = float(gen_args.get("guidance") or 3.5)
            unet = checkpoint or _FLUX_UNET
            lora_name, lora_strength = lora_spec if lora_spec else ("", 0.8)
            workflow = build_flux_img2img_workflow(
                prompt,
                init_image=init_image,
                image_folder=image_folder,
                negative_prompt=negative,
                denoise=denoise,
                steps=steps,
                guidance=guidance,
                seed=seed,
                sampler=sampler,
                scheduler=scheduler,
                unet_name=unet,
                lora_name=lora_name,
                lora_strength=lora_strength,
            )
            checkpoint = unet
            size = ""

        if workflow is None:
            return {"error": f"Unsupported edit_image mode: {mode}", "exit_code": 1}
        enqueue = await mcp.call_tool(
            _server_tool(server_id, "enqueue_workflow"),
            {"workflow": workflow},
        )
    elif tool_name == "enqueue_workflow":
        workflow = gen_args.get("workflow")
        if not isinstance(workflow, dict):
            return {"error": "enqueue_workflow requires a workflow object", "exit_code": 1}
        if not prompt or prompt == "image":
            prompt = _prompt_from_workflow(workflow) or prompt
        enqueue = await mcp.call_tool(qualified_name, gen_args)
        checkpoint = checkpoint or _FLUX_UNET

    use_flux_workflow = (
        enqueue is None
        and tool_name == "generate_image"
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
            lora_name, lora_strength = lora_spec if lora_spec else ("", 0.8)
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
                lora_name=lora_name,
                lora_strength=lora_strength,
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
    elif enqueue is None:
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
