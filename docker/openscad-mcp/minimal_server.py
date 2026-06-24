"""
Trimmed OpenSCAD HTTP API for Odysseus.

Exposes the text-to-OpenSCAD workflow (create / modify / export) without the
heavy CUDA / Gemini / SAM dependencies in upstream main.py.
"""

import logging
import os
import uuid
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from mcp_legacy import MCPServer
from src.models.code_generator import CodeGenerator
from src.nlp.parameter_extractor import ParameterExtractor
from src.openscad_wrapper.wrapper import OpenSCADWrapper
from src.utils.cad_exporter import CADExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenSCAD MCP Server (minimal)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("scad", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("output/models", exist_ok=True)
os.makedirs("output/preview", exist_ok=True)
os.makedirs("output/stl", exist_ok=True)

parameter_extractor = ParameterExtractor()
code_generator = CodeGenerator("scad", "output")
openscad_wrapper = OpenSCADWrapper("scad", "output")
cad_exporter = CADExporter()

# Upstream cad_exporter omits mesh formats; OpenSCAD can still emit them.
_EXTRA_FORMATS = {"stl": "STL mesh for 3D printing", "obj": "Wavefront OBJ mesh", "off": "OFF mesh"}
for _fmt, _desc in _EXTRA_FORMATS.items():
    cad_exporter.supported_formats.setdefault(_fmt, _desc)

models: Dict[str, Dict[str, Any]] = {}
mcp_server = MCPServer()

_PREVIEW_VIEWS = ("perspective", "front", "top", "right")


def _preview_urls_for(model_id: str) -> Dict[str, str]:
    return {view: f"/preview/{view}/{model_id}" for view in _PREVIEW_VIEWS}


def _model_response(
    model_id: str,
    *,
    model_type: str,
    parameters: Dict[str, Any],
    description: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "model_id": model_id,
        "model_type": model_type,
        "parameters": parameters,
        "supported_formats": cad_exporter.get_supported_formats(),
        "preview_urls": _preview_urls_for(model_id),
    }
    if description:
        out["description"] = description
    if extra:
        out.update(extra)
    return out


def _export_mesh(scad_file: str, model_id: str, fmt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    if fmt == "stl":
        return openscad_wrapper.generate_stl(scad_file, parameters)
    output_file = os.path.join("output", "models", f"{model_id}.{fmt}")
    cmd = ["openscad", "-o", output_file]
    if parameters:
        for key, value in parameters.items():
            cmd.extend(["-D", f"{key}={value}"])
    cmd.append(scad_file)
    import subprocess

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError(f"OpenSCAD failed to produce {fmt}")
    return output_file


@mcp_server.tool
def create_3d_model(description: str) -> Dict[str, Any]:
    """Create a parametric 3D model from a natural-language description."""
    model_type, parameters = parameter_extractor.extract_parameters(description)
    model_id = str(uuid.uuid4())
    generated_path = code_generator.generate_code(model_type, parameters)
    with open(generated_path, encoding="utf-8") as fh:
        scad_code = fh.read()
    scad_file = openscad_wrapper.generate_scad(scad_code, model_id)
    previews = openscad_wrapper.generate_multi_angle_previews(scad_file, parameters)
    success, model_file, _error = cad_exporter.export_model(
        scad_file,
        "csg",
        parameters,
        metadata={"description": description, "model_type": model_type},
    )
    models[model_id] = {
        "id": model_id,
        "type": model_type,
        "parameters": parameters,
        "description": description,
        "scad_file": scad_file,
        "model_file": model_file if success else None,
        "previews": previews,
        "format": "csg",
    }
    return {
        **_model_response(
            model_id,
            model_type=model_type,
            parameters=parameters,
            description=description,
        ),
    }


@mcp_server.tool
def modify_3d_model(model_id: str, modifications: str) -> Dict[str, Any]:
    """Modify an existing model using a natural-language change description."""
    if model_id not in models:
        raise ValueError(f"Model with ID {model_id} not found")
    model_info = models[model_id]
    _, new_parameters = parameter_extractor.extract_parameters(
        modifications,
        model_type=model_info["type"],
        existing_parameters=model_info["parameters"],
    )
    scad_code_path = code_generator.generate_code(model_info["type"], new_parameters)
    with open(scad_code_path, encoding="utf-8") as fh:
        scad_code = fh.read()
    scad_file = openscad_wrapper.generate_scad(scad_code, model_id)
    previews = openscad_wrapper.generate_multi_angle_previews(scad_file, new_parameters)
    success, model_file, _error = cad_exporter.export_model(
        scad_file,
        model_info["format"],
        new_parameters,
        metadata={
            "description": model_info["description"] + " | " + modifications,
            "model_type": model_info["type"],
        },
    )
    models[model_id] = {
        **model_info,
        "parameters": new_parameters,
        "description": model_info["description"] + " | " + modifications,
        "scad_file": scad_file,
        "model_file": model_file if success else None,
        "previews": previews,
    }
    return {
        **_model_response(
            model_id,
            model_type=model_info["type"],
            parameters=new_parameters,
        ),
    }


@mcp_server.tool
def export_model(model_id: str, format: str = "stl") -> Dict[str, Any]:
    """Export a model to a mesh or parametric format (stl recommended for printing)."""
    if model_id not in models:
        raise ValueError(f"Model with ID {model_id} not found")
    model_info = models[model_id]
    fmt = format.lower()
    supported = cad_exporter.get_supported_formats()
    if fmt not in supported:
        raise ValueError(f"Format {format} not supported. Supported: {', '.join(supported)}")
    if fmt in _EXTRA_FORMATS:
        model_file = _export_mesh(model_info["scad_file"], model_id, fmt, model_info["parameters"])
    else:
        success, model_file, error = cad_exporter.export_model(
            model_info["scad_file"],
            fmt,
            model_info["parameters"],
            metadata={
                "description": model_info["description"],
                "model_type": model_info["type"],
            },
        )
        if not success:
            raise ValueError(f"Failed to export model: {error}")
    models[model_id]["model_file"] = model_file
    models[model_id]["format"] = fmt
    return {
        **_model_response(
            model_id,
            model_type=model_info["type"],
            parameters=model_info["parameters"],
            extra={
                "format": fmt,
                "model_file": model_file,
                "download_url": f"/download/{model_id}",
            },
        ),
    }


@mcp_server.tool
def create_stl_for_printing(description: str) -> Dict[str, Any]:
    """Create a model from a description and export it as STL in one step."""
    created = create_3d_model(description)
    exported = export_model(created["model_id"], "stl")
    return {
        **created,
        **exported,
        "description": description,
    }


@mcp_server.tool
def generate_custom_scad(scad_code: str, description: str = "", export_stl: bool = True) -> Dict[str, Any]:
    """Render arbitrary OpenSCAD code into a 3D model with previews and optional STL export.

    Use this tool when the built-in shape templates are insufficient — for example
    threaded parts, multi-body assemblies, gears, snap-fits, or any geometry that
    requires raw OpenSCAD scripting.  The caller (typically the LLM) writes valid
    OpenSCAD code and passes it here.

    Args:
        scad_code: Complete, valid OpenSCAD source code to render.
        description: Human-readable summary of what the code produces.
        export_stl: If True (default), also export an STL for 3D printing.
    """
    model_id = str(uuid.uuid4())
    scad_file = openscad_wrapper.generate_scad(scad_code, model_id)

    # Try to infer bounding dimensions from the code for camera framing
    import re as _re
    dims: Dict[str, float] = {}
    for key in ("width", "depth", "height", "radius", "outer_radius", "outer_diameter",
                "pipe_od", "flange_diameter", "flange_od"):
        m = _re.search(rf'{key}\s*=\s*([0-9]+(?:\.[0-9]+)?)', scad_code)
        if m:
            dims[key] = float(m.group(1))

    previews = openscad_wrapper.generate_multi_angle_previews(scad_file, dims or None)

    model_file = None
    fmt = "scad"
    if export_stl:
        try:
            model_file = openscad_wrapper.generate_stl(scad_file)
            fmt = "stl"
        except Exception as exc:
            logger.warning("STL export failed for custom SCAD: %s", exc)

    models[model_id] = {
        "id": model_id,
        "type": "custom",
        "parameters": dims,
        "description": description,
        "scad_file": scad_file,
        "model_file": model_file,
        "previews": previews,
        "format": fmt,
    }

    resp = _model_response(
        model_id,
        model_type="custom",
        parameters=dims,
        description=description,
    )
    if model_file:
        resp["format"] = fmt
        resp["model_file"] = model_file
        resp["download_url"] = f"/download/{model_id}"
    return resp


@app.post("/tool_call")
async def handle_tool_call(request: Request) -> JSONResponse:
    data = await request.json()
    tool_name = data.get("tool_name")
    if not tool_name:
        raise HTTPException(status_code=400, detail="Tool name is required")
    if tool_name not in mcp_server.tools:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    try:
        result = mcp_server.tools[tool_name](**data.get("tool_params", {}))
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error("Error calling tool %s: %s", tool_name, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/preview/{view}/{model_id}")
async def get_preview(view: str, model_id: str) -> FileResponse:
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    previews = models[model_id].get("previews") or {}
    if view not in previews:
        raise HTTPException(status_code=404, detail=f"Preview for view {view} not found")
    preview_path = previews[view]
    if not preview_path or not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview image not found")
    return FileResponse(preview_path, media_type="image/png")


@app.get("/download/{model_id}")
async def download_model(model_id: str) -> FileResponse:
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    model_info = models[model_id]
    model_file = model_info.get("model_file")
    if not model_file or not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(model_file, filename=f"{model_id}.{model_info.get('format', 'stl')}")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "OpenSCAD MCP Server (minimal)",
        "version": "1.0.0",
        "description": "Text-to-OpenSCAD API for Odysseus",
        "tools": list(mcp_server.tools.keys()),
    }


if __name__ == "__main__":
    uvicorn.run("minimal_server:app", host="0.0.0.0", port=8000)
