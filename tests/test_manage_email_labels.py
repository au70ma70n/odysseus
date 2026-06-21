"""Tests for manage_email_labels MCP tool and agent wiring."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.tool_parsing import parse_tool_blocks
from src.tool_schemas import function_call_to_tool_block


def test_parse_manage_email_labels_fence():
    body = '{"action": "create", "label": "Banking"}'
    blocks = parse_tool_blocks(f"```manage_email_labels\n{body}\n```")
    assert len(blocks) == 1
    assert blocks[0].tool_type == "manage_email_labels"
    assert json.loads(blocks[0].content)["action"] == "create"


def test_function_call_routes_manage_email_labels_to_mcp():
    block = function_call_to_tool_block(
        "manage_email_labels",
        json.dumps({"action": "merge", "source_label": "A", "dest_label": "B"}),
    )
    assert block is not None
    assert block.tool_type == "mcp__email__manage_email_labels"
    assert json.loads(block.content)["action"] == "merge"


def test_normalize_label_path_adds_labels_prefix():
    from mcp_servers import email_server as es

    conn = MagicMock()
    conn.list.return_value = ("OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Labels/Banking"'])
    assert es._normalize_label_path("Banking", conn) == "Labels/Banking"
    assert es._normalize_label_path("Labels/Banking", conn) == "Labels/Banking"


def test_assert_label_mutable_blocks_inbox():
    from mcp_servers import email_server as es

    with pytest.raises(ValueError, match="protected"):
        es._assert_label_mutable("INBOX")


@pytest.mark.asyncio
async def test_manage_email_labels_routes_to_mcp(monkeypatch):
    import src.tool_execution as tool_execution
    from src.tool_execution import execute_tool_block

    class FakeMcp:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return {"output": "Created label `Labels/Test`", "exit_code": 0}

    fake = FakeMcp()
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: True)
    monkeypatch.setattr(tool_execution, "get_mcp_manager", lambda: fake)

    desc, result = await execute_tool_block(
        SimpleNamespace(
            tool_type="manage_email_labels",
            content='{"action": "create", "label": "Test", "account": "ProtonMail"}',
        ),
        owner="admin",
    )
    assert desc == "mcp: mcp__email__manage_email_labels"
    assert result["exit_code"] == 0
    assert fake.calls == [
        ("mcp__email__manage_email_labels", {"action": "create", "label": "Test", "account": "ProtonMail", "_odysseus_owner": "admin"}),
    ]
