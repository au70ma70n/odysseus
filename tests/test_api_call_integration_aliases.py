"""api_call accepts integration_id / id aliases models commonly emit."""

import asyncio
import json

import pytest

from src import tool_implementations as ti


@pytest.fixture
def truenas_integration(monkeypatch):
  integ = {
      "id": "c97660b25fe0",
      "name": "TrueNAS Scale",
      "base_url": "https://server.lan:444",
      "auth_type": "bearer",
      "api_key": "test-key",
      "verify_ssl": False,
      "enabled": True,
  }
  monkeypatch.setattr(ti, "load_integrations", lambda: [integ])

  async def _fake_execute(integration_id, method, path, params=None, body=None, extra_headers=None):
      assert integration_id == "c97660b25fe0"
      assert method == "GET"
      assert path == "/api/v2.0/system/info"
      return {"output": "HTTP 200\n{}", "exit_code": 0}

  # execute_api_call lives in integrations module; patch where do_api_call imports it.
  import src.integrations as integrations_mod
  monkeypatch.setattr(integrations_mod, "execute_api_call", _fake_execute)
  monkeypatch.setattr(integrations_mod, "load_integrations", lambda: [integ])
  return integ


@pytest.mark.parametrize("payload", [
    {"integration_id": "c97660b25fe0", "method": "GET", "path": "/api/v2.0/system/info"},
    {"id": "c97660b25fe0", "method": "GET", "path": "/api/v2.0/system/info"},
    {"integration": "TrueNAS Scale", "method": "GET", "path": "/api/v2.0/system/info"},
])
def test_api_call_resolves_integration_aliases(truenas_integration, payload):
    result = asyncio.run(ti.do_api_call(json.dumps(payload)))
    assert result.get("exit_code") == 0, result
