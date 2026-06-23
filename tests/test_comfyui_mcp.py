"""Tests for ComfyUI MCP generation helpers."""

import json

from src.comfyui_mcp import (
    _parse_output_image_path,
    _parse_prompt_id,
    apply_flux_generation_defaults,
    build_flux_txt2img_workflow,
    is_comfyui_generation_tool,
)


def test_is_comfyui_generation_tool():
    assert is_comfyui_generation_tool("mcp__a72dccb9__generate_image")
    assert not is_comfyui_generation_tool("mcp__a72dccb9__list_local_models")


def test_parse_prompt_id_from_json():
    stdout = json.dumps({"status": "enqueued", "prompt_id": "abc-123"})
    assert _parse_prompt_id(stdout) == "abc-123"


def test_parse_output_image_path_plain_filename():
    text = "### Outputs (1 nodes)\n- Node 7: images → **ComfyUI_00001_.png**"
    assert _parse_output_image_path(text) == ("", "ComfyUI_00001_.png")


def test_parse_output_image_path_with_subfolder():
    text = "- Node 7: images → **batch/ComfyUI_00002_.png**"
    assert _parse_output_image_path(text) == ("batch", "ComfyUI_00002_.png")


def test_apply_flux_generation_defaults_when_checkpoint_omitted(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_COMFYUI_ASSUME_FLUX", "1")
    out = apply_flux_generation_defaults({"prompt": "cat"})
    assert out["guidance"] == 3.5
    assert out["scheduler"] == "simple"
    assert out["steps"] == 20
    assert out["checkpoint"] == "flux_dev.safetensors"


def test_apply_flux_generation_defaults_respects_explicit_values():
    out = apply_flux_generation_defaults({
        "prompt": "cat",
        "checkpoint": "flux_dev.safetensors",
        "guidance": 4.0,
        "steps": 24,
    })
    assert out["guidance"] == 4.0
    assert out["steps"] == 24
    assert out["scheduler"] == "simple"


def test_build_flux_txt2img_workflow_uses_cfg_one():
    wf = build_flux_txt2img_workflow("a cat", seed=42, steps=20)
    assert wf["9"]["inputs"]["cfg"] == 1.0
    assert wf["9"]["inputs"]["seed"] == 42
    assert wf["4"]["inputs"]["clip_l"] == "a cat"


def test_apply_flux_generation_defaults_skips_non_flux(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_COMFYUI_ASSUME_FLUX", "0")
    out = apply_flux_generation_defaults({
        "prompt": "cat",
        "checkpoint": "sd_xl_base_1.0.safetensors",
    })
    assert "cfg" not in out
