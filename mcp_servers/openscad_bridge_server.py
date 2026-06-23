"""
openscad_bridge_server.py

Stdio MCP bridge to the OpenSCAD HTTP sidecar. Proxies tool calls and
downloads STL files into Odysseus data for user download links.
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import GENERATED_MODELS_DIR

server = Server("openscad")
_BASE_URL = os.environ.get("OPENSCAD_MCP_URL", "http://openscad-mcp:8000").rstrip("/")
_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)

_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "create_3d_model": {
        "description": "Create a parametric 3D model from a natural-language description using OpenSCAD.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What to model, including dimensions (e.g. 'hollow box 40mm wide, 30mm deep, 20mm tall')",
                },
            },
            "required": ["description"],
        },
    },
    "modify_3d_model": {
        "description": "Modify an existing OpenSCAD model by model_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "ID returned by create_3d_model"},
                "modifications": {"type": "string", "description": "Natural-language changes to apply"},
            },
            "required": ["model_id", "modifications"],
        },
    },
    "export_model": {
        "description": "Export a model to a file format. Use stl for 3D printing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Model ID to export"},
                "format": {
                    "type": "string",
                    "description": "Export format (stl, obj, scad, csg, 3mf, amf, ...)",
                    "default": "stl",
                },
            },
            "required": ["model_id"],
        },
    },
    "create_stl_for_printing": {
        "description": "Create a 3D model from a description and export STL for 3D printing in one step.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What to print, with dimensions in mm when possible",
                },
            },
            "required": ["description"],
        },
    },
}


async def _remote_tools() -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_BASE_URL}/")
            resp.raise_for_status()
            data = resp.json()
            tools = data.get("tools") or []
            return [t for t in tools if isinstance(t, str)]
    except Exception:
        return list(_TOOL_SCHEMAS.keys())


async def _call_remote(tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_BASE_URL}/tool_call",
            json={"tool_name": tool_name, "tool_params": tool_params},
        )
        if resp.status_code >= 400:
            detail = resp.text[:500]
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"{tool_name} failed ({resp.status_code}): {detail}")
        return resp.json()


def _public_base() -> str:
    try:
        from src.settings import get_setting

        return (get_setting("app_public_url", "") or "").rstrip("/")
    except Exception:
        return ""


async def _deliver_model_file(remote_result: Dict[str, Any], description: str = "") -> str:
    """Download model bytes from the sidecar and store under GENERATED_MODELS_DIR."""
    model_id = remote_result.get("model_id")
    fmt = (remote_result.get("format") or "stl").lower()
    if not model_id:
        raise RuntimeError("Remote result missing model_id")

    download_url = remote_result.get("download_url") or f"/download/{model_id}"
    if not download_url.startswith("http"):
        download_url = urljoin(f"{_BASE_URL}/", download_url.lstrip("/"))

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(download_url)
        resp.raise_for_status()
        content = resp.content

    if len(content) < 80:
        raise RuntimeError("Downloaded model file is unexpectedly small")

    out_dir = Path(GENERATED_MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:12]}.{fmt}"
    (out_dir / filename).write_bytes(content)

    pub = _public_base()
    link = f"{pub}/api/generated-model/{filename}"
    lines = [
        f"STL model ready for 3D printing.",
        f"Direct link: {link}",
        f"model_id: {model_id}",
        f"format: {fmt}",
    ]
    if description:
        lines.insert(1, f"description: {description[:200]}")
    return "\n".join(lines)


@server.list_tools()
async def list_tools() -> list[Tool]:
    names = await _remote_tools()
    tools: list[Tool] = []
    for name in names:
        meta = _TOOL_SCHEMAS.get(name)
        if meta:
            tools.append(Tool(name=name, description=meta["description"], inputSchema=meta["inputSchema"]))
        else:
            tools.append(
                Tool(
                    name=name,
                    description=f"Proxy to OpenSCAD sidecar tool {name}",
                    inputSchema={"type": "object", "properties": {}},
                )
            )
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = await _call_remote(name, arguments or {})
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]

    if name in ("export_model", "create_stl_for_printing") and (
        (arguments or {}).get("format", "stl") == "stl" or name == "create_stl_for_printing"
    ):
        try:
            text = await _deliver_model_file(result, description=(arguments or {}).get("description", ""))
            return [TextContent(type="text", text=text)]
        except Exception as exc:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Model exported on sidecar but delivery failed: {exc}\n"
                        f"Raw result: {json.dumps(result, indent=2)[:2000]}"
                    ),
                )
            ]

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
