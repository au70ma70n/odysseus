"""Tests for the proseless-turn synthesis guard.

When an agent turn runs multiple tools but never writes user-facing prose,
the end-of-loop guard should force a synthesis pass instead of leaving the
user staring at a wall of tool-call blocks.
"""
import pytest

from src.agent_loop import _has_meaningful_prose, _proseless_agent_synthesis


# ── _has_meaningful_prose ─────────────────────────────────────────────────

def test_detects_all_empty_rounds():
    assert _has_meaningful_prose(["", "", ""]) is False


def test_detects_think_only_rounds():
    rounds = [
        "<think>thinking hard</think>",
        "<think>\nmore thinking\n</think>",
        "",
    ]
    assert _has_meaningful_prose(rounds) is False


def test_detects_whitespace_only():
    assert _has_meaningful_prose(["   ", "\n\n", "\t"]) is False


def test_single_prose_round_is_meaningful():
    rounds = ["", "", "Here is the summary of what I found."]
    assert _has_meaningful_prose(rounds) is True


def test_prose_mixed_with_empty_is_meaningful():
    rounds = ["", "The server is running TrueNAS 25.10.", ""]
    assert _has_meaningful_prose(rounds) is True


def test_empty_list_not_meaningful():
    assert _has_meaningful_prose([]) is False


def test_none_entries_tolerated():
    assert _has_meaningful_prose([None, None]) is False


def test_think_with_trailing_prose_is_meaningful():
    rounds = ["<think>plan</think>Here's what I found:"]
    assert _has_meaningful_prose(rounds) is True


# ── _proseless_agent_synthesis ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesis_returns_clean_text(monkeypatch):
    """Synthesis call should return cleaned prose, stripping think/tool blocks."""
    async def fake_llm(*args, **kwargs):
        return "<think>planning</think>Here is the final answer."

    monkeypatch.setattr("src.llm_core.llm_call_async", fake_llm)

    result = await _proseless_agent_synthesis(
        messages=[{"role": "user", "content": "list apps"}],
        endpoint_url="http://localhost:11434/v1/chat/completions",
        model="test-model",
        headers=None,
        max_tokens=4096,
    )
    assert result == "Here is the final answer."


@pytest.mark.asyncio
async def test_synthesis_returns_empty_on_failure(monkeypatch):
    """On LLM failure, synthesis should return empty string, not raise."""
    async def failing_llm(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("src.llm_core.llm_call_async", failing_llm)

    result = await _proseless_agent_synthesis(
        messages=[{"role": "user", "content": "list apps"}],
        endpoint_url="http://localhost:11434/v1/chat/completions",
        model="test-model",
        headers=None,
        max_tokens=4096,
    )
    assert result == ""


@pytest.mark.asyncio
async def test_synthesis_prompt_asks_for_no_tools(monkeypatch):
    """The synthesis prompt should tell the model not to call tools."""
    captured_messages = []

    async def capture_llm(*args, **kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return "Summary goes here."

    monkeypatch.setattr("src.llm_core.llm_call_async", capture_llm)

    await _proseless_agent_synthesis(
        messages=[
            {"role": "user", "content": "deploy comfyui"},
            {"role": "assistant", "content": "```api_call\n...\n```"},
        ],
        endpoint_url="http://example.com/v1/chat/completions",
        model="test-model",
        headers=None,
        max_tokens=4096,
    )
    last_msg = captured_messages[-1]["content"]
    assert "Do NOT call any" in last_msg
    assert "tool" in last_msg.lower()
