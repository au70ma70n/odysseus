"""Tests for ComfyUI MCP generation helpers."""

import json

from src.comfyui_mcp import (
    _build_source_load_node,
    _parse_output_image_path,
    _parse_prompt_id,
    apply_flux_generation_defaults,
    build_flux_img2img_workflow,
    build_flux_kontext_workflow,
    build_flux_txt2img_workflow,
    comfyui_virtual_tools,
    is_comfyui_generation_tool,
    parse_lora_from_args,
)


def test_is_comfyui_generation_tool():
    assert is_comfyui_generation_tool("mcp__a72dccb9__generate_image")
    assert is_comfyui_generation_tool("mcp__a72dccb9__edit_image")
    assert is_comfyui_generation_tool("mcp__a72dccb9__enqueue_workflow")
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
    assert wf["9"]["inputs"]["model"] == ["1", 0]
    assert wf["4"]["inputs"]["clip_l"] == "a cat"
    assert "12" not in wf


def test_build_flux_txt2img_workflow_with_lora():
    wf = build_flux_txt2img_workflow(
        "portrait",
        seed=7,
        lora_name="NSFW_master.safetensors",
        lora_strength=0.8,
    )
    assert wf["12"]["class_type"] == "LoraLoaderModelOnly"
    assert wf["12"]["inputs"]["lora_name"] == "NSFW_master.safetensors"
    assert wf["12"]["inputs"]["strength_model"] == 0.8
    assert wf["9"]["inputs"]["model"] == ["12", 0]


def test_parse_lora_from_args_name_and_strength():
    assert parse_lora_from_args({
        "lora": "test_lora.safetensors",
        "lora_strength": 0.6,
    }) == ("test_lora.safetensors", 0.6)


def test_parse_lora_from_args_list_of_dicts():
    assert parse_lora_from_args({
        "loras": [{"name": "foo.safetensors", "strength": 0.75}],
    }) == ("foo.safetensors", 0.75)


def test_parse_lora_from_args_missing():
    assert parse_lora_from_args({"prompt": "cat"}) is None


def test_apply_flux_generation_defaults_skips_non_flux(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_COMFYUI_ASSUME_FLUX", "0")
    out = apply_flux_generation_defaults({
        "prompt": "cat",
        "checkpoint": "sd_xl_base_1.0.safetensors",
    })
    assert "cfg" not in out


def test_build_source_load_node_output_annotation():
    node = _build_source_load_node("ComfyUI_00025_.png", "output")
    assert node["class_type"] == "LoadImageOutput"
    assert node["inputs"]["image"] == "ComfyUI_00025_.png [output]"


def test_build_flux_kontext_workflow_reference_latent():
    wf = build_flux_kontext_workflow("change her hair to black", init_image="ComfyUI_00025_.png")
    assert wf["4"]["inputs"]["image"] == "ComfyUI_00025_.png [output]"
    assert wf["9"]["class_type"] == "ReferenceLatent"
    assert wf["7"]["inputs"]["text"] == "change her hair to black"


def test_build_flux_img2img_workflow_denoise():
    wf = build_flux_img2img_workflow("black hair", init_image="ComfyUI_00025_.png", denoise=0.55)
    assert wf["10"]["inputs"]["denoise"] == 0.55
    assert wf["13"]["inputs"]["filename_prefix"] == "ComfyUI_edit"


def test_comfyui_virtual_tools_injected_when_generate_present():
    tools = [{"name": "generate_image", "description": "gen"}]
    virtual = comfyui_virtual_tools(tools)
    assert len(virtual) == 1
    assert virtual[0]["name"] == "edit_image"


def test_comfyui_virtual_tools_skipped_when_already_present():
    tools = [{"name": "generate_image"}, {"name": "edit_image"}]
    assert comfyui_virtual_tools(tools) == []
