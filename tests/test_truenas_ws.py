"""Tests for TrueNAS WebSocket log helpers."""
import json

from src.truenas_ws import container_log_event_name, is_truenas_integration


def test_container_log_event_name_format():
    name = container_log_event_name("comfyui", "abc123", tail_lines=50)
    assert name.startswith("app.container_log_follow:")
    payload = json.loads(name.split(":", 1)[1])
    assert payload == {
        "app_name": "comfyui",
        "container_id": "abc123",
        "tail_lines": 50,
    }


def test_is_truenas_integration_by_name_or_preset():
    assert is_truenas_integration({"name": "TrueNAS Scale"})
    assert is_truenas_integration({"preset": "truenas", "name": "NAS"})
    assert not is_truenas_integration({"name": "Miniflux"})
