#!/usr/bin/env python3
import asyncio
import httpx
from src.comfyui_mcp import build_flux_txt2img_workflow


async def main():
    wf = build_flux_txt2img_workflow(
        "sharp detailed anime girl with orange eyes, pink top, studio lighting, crisp focus",
        seed=12345,
        steps=20,
    )
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post("http://server.lan:8188/prompt", json={"prompt": wf})
        print("enqueue", response.status_code, response.text[:300])
        prompt_id = response.json()["prompt_id"]
        for _ in range(180):
            history = await client.get(f"http://server.lan:8188/history/{prompt_id}")
            data = history.json()
            if prompt_id not in data:
                await asyncio.sleep(2)
                continue
            entry = data[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                print("error", status)
                return 1
            outputs = entry.get("outputs") or {}
            for output in outputs.values():
                for image in output.get("images", []):
                    print("done", image["filename"])
                    return 0
            await asyncio.sleep(2)
    print("timeout")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
