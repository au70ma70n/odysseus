#!/usr/bin/env python3
"""Download Flux Dev component models for ComfyUI and link the existing UNet checkpoint."""

from __future__ import annotations

import asyncio
import sys

FLUX_DOWNLOADS = [
    (
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "clip",
        "clip_l.safetensors",
    ),
    (
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "text_encoders",
        "t5xxl_fp8_e4m3fn.safetensors",
    ),
    (
        "https://huggingface.co/second-state/FLUX.1-dev-GGUF/resolve/main/ae.safetensors",
        "vae",
        "ae.safetensors",
    ),
]

MCP_SERVER_ID = "a72dccb9"


async def download_via_mcp(mcp, url: str, subfolder: str, filename: str) -> None:
    tool = f"mcp__{MCP_SERVER_ID}__download_model"
    result = await mcp.call_tool(
        tool,
        {"url": url, "target_subfolder": subfolder, "filename": filename},
    )
    if result.get("exit_code") not in (0, None):
        raise RuntimeError(result.get("error") or result.get("stderr") or result.get("stdout") or "download failed")
    print(result.get("stdout") or "ok")


async def main() -> int:
    from src.mcp_manager import McpManager
    from src.comfyui_mcp import flux_stack_status

    mcp = McpManager()
    await mcp.connect_all_enabled()
    if MCP_SERVER_ID not in mcp._sessions:
        print(f"ComfyUI MCP server {MCP_SERVER_ID} is not connected.", file=sys.stderr)
        return 1

    print("Updating TrueNAS sidecar (UNet symlink on startup) ...")
    from scripts.update_comfyui_mcp_sidecar import update_truenas_app

    if not await update_truenas_app():
        print("Warning: could not update TrueNAS app compose.", file=sys.stderr)

    ready, missing = await flux_stack_status()
    if ready:
        print("Flux model stack already complete.")
        return 0

    print("Missing components:", ", ".join(missing))
    for url, subfolder, filename in FLUX_DOWNLOADS:
        if any(filename in item for item in missing):
            print(f"Downloading {subfolder}/{filename} ...")
            await download_via_mcp(mcp, url, subfolder, filename)

    for attempt in range(30):
        ready, missing = await flux_stack_status()
        if ready:
            print("Flux model stack ready.")
            return 0
        if attempt == 0:
            print("Waiting for models in ComfyUI:", ", ".join(missing))
        await asyncio.sleep(5)

    print("Still missing:", ", ".join(missing), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
