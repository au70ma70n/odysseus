#!/usr/bin/env python3
"""Download Flux encoder/VAE files via ComfyUI MCP."""

from __future__ import annotations

import asyncio
import sys

MCP_SERVER_ID = "a72dccb9"
DOWNLOADS = [
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


async def main() -> int:
    from src.comfyui_mcp import flux_stack_status
    from src.mcp_manager import McpManager

    mcp = McpManager()
    await mcp.connect_all_enabled()
    if MCP_SERVER_ID not in mcp._sessions:
        print(f"MCP server {MCP_SERVER_ID} not connected", file=sys.stderr)
        return 1

    ready, missing = await flux_stack_status()
    if ready:
        print("Flux stack already complete.")
        return 0

    for url, subfolder, filename in DOWNLOADS:
        if not any(filename in item for item in missing):
            continue
        print(f"Downloading {subfolder}/{filename} ...", flush=True)
        result = await mcp.call_tool(
            f"mcp__{MCP_SERVER_ID}__download_model",
            {"url": url, "target_subfolder": subfolder, "filename": filename},
        )
        print(result.get("stdout") or result.get("error") or result, flush=True)
        if result.get("exit_code") not in (0, None):
            return 1

    for _ in range(24):
        ready, missing = await flux_stack_status()
        print(f"ready={ready} missing={missing}", flush=True)
        if ready:
            return 0
        await asyncio.sleep(5)

    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
