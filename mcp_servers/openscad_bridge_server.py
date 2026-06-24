"""
openscad_bridge_server.py

Stdio MCP bridge to the OpenSCAD HTTP sidecar. Proxies tool calls, fetches
multi-angle preview PNGs, and downloads STL files into Odysseus data.
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

from src.constants import GENERATED_IMAGES_DIR, GENERATED_MODELS_DIR

server = Server("openscad")
_BASE_URL = os.environ.get("OPENSCAD_MCP_URL", "http://openscad-mcp:8000").rstrip("/")
_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)
_PREVIEW_VIEWS = ("perspective", "front", "top", "right")
_MODEL_TOOLS = frozenset({
    "create_3d_model",
    "modify_3d_model",
    "export_model",
    "create_stl_for_printing",
    "generate_custom_scad",
})

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
    "generate_custom_scad": {
        "description": (
            "Render arbitrary OpenSCAD code into a 3D model with preview images and "
            "optional STL export. Use this for complex geometry that the basic shape "
            "templates cannot produce — threaded parts, gears, multi-body assemblies, "
            "snap-fits, flanges, conduits, enclosures with features, etc. "
            "The caller writes complete, valid OpenSCAD source code. "
            "A thread helper module (iso_metric_thread) is available via "
            "'include <threads.scad>;'. "
            "Tip: assign key dimensions to top-level variables named width, height, "
            "depth, radius, or outer_diameter so the preview camera can frame the part."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scad_code": {
                    "type": "string",
                    "description": (
                        "Complete OpenSCAD source code. Must be syntactically valid. "
                        "Can use include <threads.scad>; for ISO metric threads."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable summary of what the code produces",
                },
                "export_stl": {
                    "type": "boolean",
                    "description": "Whether to also export an STL file (default true)",
                    "default": True,
                },
            },
            "required": ["scad_code"],
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


def _absolute_sidecar_url(path: str) -> str:
    if path.startswith("http"):
        return path
    return urljoin(f"{_BASE_URL}/", path.lstrip("/"))


async def _fetch_and_store_previews(
    client: httpx.AsyncClient,
    model_id: str,
    preview_urls: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Download preview PNGs from the sidecar into GENERATED_IMAGES_DIR."""
    urls = preview_urls or {view: f"/preview/{view}/{model_id}" for view in _PREVIEW_VIEWS}
    img_dir = Path(GENERATED_IMAGES_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)
    pub = _public_base()
    stored: List[str] = []

    for view in _PREVIEW_VIEWS:
        rel = urls.get(view) or f"/preview/{view}/{model_id}"
        try:
            resp = await client.get(_absolute_sidecar_url(rel))
            resp.raise_for_status()
            if len(resp.content) < 100:
                continue
            filename = f"{uuid.uuid4().hex[:12]}_{view}.png"
            (img_dir / filename).write_bytes(resp.content)
            stored.append(f"{pub}/api/generated-image/{filename}")
        except Exception:
            continue
    return stored


async def _fetch_and_store_stl(client: httpx.AsyncClient, remote_result: Dict[str, Any]) -> Optional[str]:
    model_id = remote_result.get("model_id")
    fmt = (remote_result.get("format") or "stl").lower()
    if not model_id:
        return None

    download_url = remote_result.get("download_url") or f"/download/{model_id}"
    resp = await client.get(_absolute_sidecar_url(download_url))
    resp.raise_for_status()
    content = resp.content
    if len(content) < 80:
        raise RuntimeError("Downloaded model file is unexpectedly small")

    out_dir = Path(GENERATED_MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:12]}.{fmt}"
    (out_dir / filename).write_bytes(content)
    pub = _public_base()
    return f"{pub}/api/generated-model/{filename}"


def _wants_stl(tool_name: str, arguments: Dict[str, Any], remote_result: Dict[str, Any]) -> bool:
    if tool_name == "create_stl_for_printing":
        return True
    if tool_name == "generate_custom_scad":
        return (arguments or {}).get("export_stl", True) is not False
    if tool_name == "export_model":
        fmt = (arguments or {}).get("format") or remote_result.get("format") or "stl"
        return str(fmt).lower() == "stl"
    return False


async def _deliver_model_artifacts(
    tool_name: str,
    remote_result: Dict[str, Any],
    arguments: Optional[Dict[str, Any]] = None,
) -> str:
    arguments = arguments or {}
    model_id = remote_result.get("model_id")
    description = (
        remote_result.get("description")
        or arguments.get("description")
        or arguments.get("modifications")
        or ""
    )
    fmt = (remote_result.get("format") or "stl").lower()

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        preview_links = await _fetch_and_store_previews(
            client,
            model_id,
            remote_result.get("preview_urls"),
        )
        stl_link = None
        if _wants_stl(tool_name, arguments, remote_result):
            stl_link = await _fetch_and_store_stl(client, remote_result)

    lines: List[str] = []
    if stl_link:
        lines.append("STL model ready for 3D printing.")
    else:
        lines.append("3D model created.")
    if description:
        lines.append(f"description: {description[:200]}")
    for i, link in enumerate(preview_links):
        view = _PREVIEW_VIEWS[i] if i < len(_PREVIEW_VIEWS) else f"view{i}"
        lines.append(f"Preview {view}: {link}")
    if stl_link:
        lines.append(f"Direct link: {stl_link}")
    if model_id:
        lines.append(f"model_id: {model_id}")
    if stl_link:
        lines.append(f"format: {fmt}")
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

    if name in _MODEL_TOOLS:
        try:
            text = await _deliver_model_artifacts(name, result, arguments or {})
            return [TextContent(type="text", text=text)]
        except Exception as exc:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Model created on sidecar but delivery failed: {exc}\n"
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
