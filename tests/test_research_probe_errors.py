"""Regression tests for Deep Research model probe error messages.

Deep Research probes the selected model before starting a long run. When the
upstream returned a concrete model/API error, the probe used to collapse it into
"Cannot reach model", hiding the real issue from the UI.
"""
import pytest
from fastapi import HTTPException

from src.research_handler import ResearchHandler, _format_probe_failure, _probe_call_budget


def test_probe_failure_preserves_upstream_model_errors():
    exc = HTTPException(
        status_code=400,
        detail="OpenAI returned HTTP 400: Unsupported parameter: temperature",
    )

    msg = _format_probe_failure("o3-mini", exc)

    assert msg == (
        "Model 'o3-mini' probe failed: "
        "OpenAI returned HTTP 400: Unsupported parameter: temperature"
    )


def test_probe_failure_keeps_api_key_guidance():
    exc = HTTPException(status_code=401, detail="OpenAI authentication failed")

    assert _format_probe_failure("gpt-4o", exc) == (
        "Model 'gpt-4o' requires an API key. Check your endpoint configuration."
    )


def test_probe_call_budget_ollama_openai_compat_gets_cold_load_time():
    timeout, retries = _probe_call_budget(
        "http://host.docker.internal:11434/v1/chat/completions"
    )
    assert timeout == 180
    assert retries == 2


def test_probe_call_budget_ollama_native_gets_cold_load_time():
    timeout, retries = _probe_call_budget("http://localhost:11434/api/chat")
    assert timeout == 180
    assert retries == 2


def test_probe_call_budget_cloud_api_stays_fast():
    timeout, retries = _probe_call_budget("https://api.openai.com/v1/chat/completions")
    assert timeout == 15
    assert retries == 1


def test_probe_failure_keeps_reachability_guidance_for_plain_errors():
    msg = _format_probe_failure("local-model", RuntimeError("connection refused"))

    assert msg == "Cannot reach model 'local-model' — connection refused"


@pytest.mark.asyncio
async def test_probe_endpoint_uses_ollama_cold_load_budget(monkeypatch):
    captured = {}

    async def _capture(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.llm_core.llm_call_async", _capture)

    await ResearchHandler._probe_endpoint(
        "http://host.docker.internal:11434/v1/chat/completions",
        "huihui_ai/Qwen3.6-abliterated:35b",
        None,
    )

    assert captured["timeout"] == 180
    assert captured["max_retries"] == 2


@pytest.mark.asyncio
async def test_probe_endpoint_surfaces_http_exception_detail(monkeypatch):
    async def _raise(*args, **kwargs):
        raise HTTPException(
            status_code=400,
            detail="OpenAI returned HTTP 400: max_tokens is not supported",
        )

    monkeypatch.setattr("src.llm_core.llm_call_async", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        await ResearchHandler._probe_endpoint(
            "https://api.openai.com/v1/chat/completions",
            "o3-mini",
            {"Authorization": "Bearer test"},
        )

    msg = str(excinfo.value)
    assert "Model 'o3-mini' probe failed" in msg
    assert "max_tokens is not supported" in msg
    assert "Cannot reach model" not in msg
