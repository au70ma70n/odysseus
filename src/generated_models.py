import os
import re
from pathlib import Path

from fastapi import HTTPException

from src.constants import GENERATED_MODELS_DIR


GENERATED_MODEL_DIR = Path(GENERATED_MODELS_DIR)
GENERATED_MODEL_RE = re.compile(r"^[a-f0-9]{8,64}\.(stl|obj|3mf|amf|scad|csg|off)$")
GENERATED_MODEL_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
    "X-Content-Type-Options": "nosniff",
}


def resolve_generated_model_path(filename: str) -> Path:
    if not isinstance(filename, str) or not GENERATED_MODEL_RE.fullmatch(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    root = GENERATED_MODEL_DIR.resolve()
    path = (GENERATED_MODEL_DIR / filename).resolve()
    try:
        if os.path.commonpath([str(root), str(path)]) != str(root):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return path
