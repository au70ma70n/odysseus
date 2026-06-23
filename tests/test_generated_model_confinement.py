"""Tests for generated model path confinement."""

import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from src import generated_models


def test_resolve_generated_model_path_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(generated_models, "GENERATED_MODEL_DIR", tmp_path)
    filename = "abcd1234ef56.stl"
    model_path = tmp_path / filename
    model_path.write_bytes(b"solid test\nendsolid test\n")
    assert generated_models.resolve_generated_model_path(filename) == model_path


def test_resolve_generated_model_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(generated_models, "GENERATED_MODEL_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        generated_models.resolve_generated_model_path("../secrets.stl")
    assert exc.value.status_code == 400


def test_resolve_generated_model_path_rejects_bad_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(generated_models, "GENERATED_MODEL_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        generated_models.resolve_generated_model_path("abcd1234ef56.exe")
    assert exc.value.status_code == 400


def test_resolve_generated_model_path_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(generated_models, "GENERATED_MODEL_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        generated_models.resolve_generated_model_path("abcd1234ef56.stl")
    assert exc.value.status_code == 404
