#!/usr/bin/env python3
"""Update TrueNAS ComfyUI app to add comfyui-mcp sidecar and repoint Odysseus MCP."""

import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timezone

COMPOSE = """services:
  comfyui:
    image: tommasopiantanida/comfyui-truenas:latest
    restart: unless-stopped
    privileged: true
    user: "0:0"
    ports:
      - "8188:8188"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - PORT=8188
    volumes:
      - /mnt/data/app_data/comfyui_data/models:/app/ComfyUI/models
      - /mnt/data/app_data/comfyui_data/output:/app/ComfyUI/output
      - /mnt/data/app_data/comfyui_data/input:/app/ComfyUI/input
      - /mnt/data/app_config/comfyui_config/custom_nodes:/app/ComfyUI/custom_nodes
      - /mnt/data/app_config/comfyui_config/user:/app/ComfyUI/user
  comfyui-mcp:
    image: node:22-bookworm-slim
    restart: unless-stopped
    depends_on:
      - comfyui
    ports:
      - "9100:9100"
    environment:
      - COMFYUI_PATH=/comfyui
    volumes:
      - /mnt/data/app_data/comfyui_data:/comfyui:rw
      - /mnt/data/app_config/comfyui_config/custom_nodes:/comfyui/custom_nodes:rw
      - /mnt/data/app_config/comfyui_config/user:/comfyui/user:rw
    command: bash -lc "mkdir -p /comfyui/models/diffusion_models /comfyui/models/clip /comfyui/models/text_encoders /comfyui/models/vae && if [ -f /comfyui/models/checkpoints/flux_dev.safetensors ] && [ ! -e /comfyui/models/diffusion_models/flux1-dev-fp8.safetensors ]; then ln -sf ../checkpoints/flux_dev.safetensors /comfyui/models/diffusion_models/flux1-dev-fp8.safetensors; fi && npm install -g comfyui-mcp@latest --ignore-scripts && exec comfyui-mcp --http --host 0.0.0.0 --port 9100 --comfyui-url http://comfyui:8188"
"""

MCP_SERVER_ID = "a72dccb9"
MCP_HTTP_URL = "http://server.lan:9100/mcp"


async def poll_job(job_id: int, label: str) -> bool:
    from src.integrations import execute_api_call

    for _ in range(60):
        r = await execute_api_call(
            "TrueNAS Scale", "GET", "/api/v2.0/core/get_jobs", params={"id": job_id}
        )
        out = r.get("output", "")
        if "FAILED" in out:
            print(f"{label} FAILED:\n{out}")
            return False
        if "SUCCESS" in out:
            print(f"{label} SUCCESS (job {job_id})")
            return True
        await asyncio.sleep(3)
    print(f"{label} timed out (job {job_id})")
    return False


async def update_truenas_app() -> bool:
    from src.integrations import execute_api_call

    body = {"custom_compose_config_string": COMPOSE}
    r = await execute_api_call("TrueNAS Scale", "PUT", "/api/v2.0/app/id/comfyui", body=body)
    out = r.get("output", "")
    print(out[:500])
    if not out.startswith("HTTP 200"):
        return False
    job_id = int(out.strip().splitlines()[-1])
    return await poll_job(job_id, "app.update")


def update_odysseus_mcp() -> None:
    db_path = "/app/data/app.db"
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE mcp_servers
        SET transport = 'http',
            command = '',
            args = '[]',
            env = '{}',
            url = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (MCP_HTTP_URL, now, MCP_SERVER_ID),
    )
    conn.commit()
    print(f"Updated MCP server {MCP_SERVER_ID} -> http {MCP_HTTP_URL} ({cur.rowcount} row)")
    conn.close()


async def main() -> int:
    if not await update_truenas_app():
        return 1
    update_odysseus_mcp()
    print("Done. Restart Odysseus to reconnect the ComfyUI MCP server.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
