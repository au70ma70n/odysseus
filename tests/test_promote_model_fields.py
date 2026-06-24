"""Unit tests for `_promote_model_fields` (OpenSCAD STL delivery)."""

from src.tool_execution import _promote_model_fields


def _result(stdout, exit_code=0):
    return {"exit_code": exit_code, "stdout": stdout}


def test_relative_model_url_promoted():
    r = _result(
        "3D model created.\n"
        "description: a solid cube 20 mm wide\n"
        "Preview perspective: /api/generated-image/aa11_perspective.png\n"
        "Preview front: /api/generated-image/aa11_front.png\n"
        "Direct link: /api/generated-model/55a71e83d9c4.stl\n"
        "format: stl\n"
    )
    _promote_model_fields(r)
    assert r["model_url"] == "/api/generated-model/55a71e83d9c4.stl"
    assert r["model_preview_urls"] == [
        "/api/generated-image/aa11_perspective.png",
        "/api/generated-image/aa11_front.png",
    ]
    assert r["model_preview_url"] == "/api/generated-image/aa11_perspective.png"


def test_no_url_no_promotion():
    r = _result("Model created but no link")
    _promote_model_fields(r)
    assert "model_url" not in r
