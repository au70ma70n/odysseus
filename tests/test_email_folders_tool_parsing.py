"""Fenced-block parsing for email folder/label tools."""

from src.tool_parsing import parse_tool_blocks


def test_parse_empty_list_email_folders_fence():
    blocks = parse_tool_blocks("```list_email_folders\n```")
    assert len(blocks) == 1
    assert blocks[0].tool_type == "list_email_folders"
    assert blocks[0].content == "{}"


def test_parse_empty_list_email_labels_fence():
    blocks = parse_tool_blocks("```list_email_labels\n```")
    assert len(blocks) == 1
    assert blocks[0].tool_type == "list_email_labels"
    assert blocks[0].content == "{}"


def test_parse_list_email_folders_with_json_args():
    blocks = parse_tool_blocks('```list_email_folders\n{"account": "ProtonMail"}\n```')
    assert len(blocks) == 1
    assert blocks[0].tool_type == "list_email_folders"
    assert blocks[0].content == '{"account": "ProtonMail"}'


def test_manage_email_labels_in_tool_tags():
    from src.agent_tools import TOOL_TAGS
    assert "manage_email_labels" in TOOL_TAGS


def test_parse_manage_email_labels_fence():
    body = '{"action": "merge", "source_label": "Labels/A", "dest_label": "Labels/B"}'
    blocks = parse_tool_blocks(f"```manage_email_labels\n{body}\n```")
    assert len(blocks) == 1
    assert blocks[0].tool_type == "manage_email_labels"
    assert "merge" in blocks[0].content


def test_parse_multiple_manage_email_labels_fences():
    text = (
        '```manage_email_labels\n{"action": "merge", "source_label": "Labels/A", "dest_label": "Labels/B"}\n```\n\n'
        '```manage_email_labels\n{"action": "delete", "label": "Labels/A"}\n```'
    )
    blocks = parse_tool_blocks(text)
    assert len(blocks) == 2
    assert all(b.tool_type == "manage_email_labels" for b in blocks)
