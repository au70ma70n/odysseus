"""Unit tests for `_promote_image_fields` and duplicate-image markdown stripping."""
from src.tool_execution import (
    _promote_image_fields,
    strip_duplicate_generated_image_markdown,
    format_tool_result,
)


def _result(stdout, exit_code=0):
    return {"exit_code": exit_code, "stdout": stdout}


def test_absolute_url_promoted_with_fields():
    """An absolute https URL in stdout is lifted into image_url, along with the
    prompt/model/size lines."""
    r = _result(
        "Generated image for: a red fox in snow\n"
        "Direct link: https://odysseus.example.com/api/generated-image/abc123.png\n"
        "model: qwen-image\n"
        "size: 1024x1024"
    )
    _promote_image_fields(r)
    assert r["image_url"] == "https://odysseus.example.com/api/generated-image/abc123.png"
    assert r["image_prompt"] == "a red fox in snow"
    assert r["image_model"] == "qwen-image"
    assert r["image_size"] == "1024x1024"


def test_relative_url_promoted():
    """A relative /api/generated-image/... path (no host) is still matched."""
    r = _result(
        "Generated image for: a cat\n"
        "Direct link: /api/generated-image/def456.png"
    )
    _promote_image_fields(r)
    assert r["image_url"] == "/api/generated-image/def456.png"
    assert r["image_prompt"] == "a cat"


def test_no_url_leaves_result_unchanged():
    """No generated-image URL anywhere -> no image_url key is added."""
    r = _result("Generated image for: a dog\n(no link produced)")
    _promote_image_fields(r)
    assert "image_url" not in r
    assert "image_prompt" not in r


def test_nonzero_exit_not_promoted():
    """A non-success result is never promoted, even if stdout contains a URL."""
    r = _result("https://host/api/generated-image/zzz.png", exit_code=1)
    _promote_image_fields(r)
    assert "image_url" not in r


def test_strip_duplicate_markdown_image():
    """Markdown embeds for already-delivered URLs are removed from prose."""
  tool_events = [{"image_url": "/api/generated-image/abc123.png"}]
  text = (
      "Here is your image:\n\n"
      "![Generated Image](/api/generated-image/abc123.png)\n\n"
      "Hope you like it."
  )
  assert strip_duplicate_generated_image_markdown(text, tool_events) == (
      "Here is your image:\n\n\n\nHope you like it."
  )


def test_strip_leaves_unrelated_markdown():
    """Images not in tool_events are left intact."""
    text = "![other](/api/generated-image/other.png)"
    assert strip_duplicate_generated_image_markdown(text, []) == text


def test_format_tool_result_image_guidance():
    """format_tool_result tells the model not to echo image markdown."""
    r = _result(
        "Generated image for: a cat\n"
        "Direct link: /api/generated-image/cat.png\n"
        "model: flux\n"
        "size: 1024x1024",
    )
    _promote_image_fields(r)
    out = format_tool_result("generate_image", r)
    assert "shown automatically in chat" in out
    assert "Direct link:" not in out
    assert "do not embed" in out.lower() or "Do not embed" in out
