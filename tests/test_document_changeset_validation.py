from pathlib import Path

from pixie_core import tools
from pixie_core._api import Engine


def _engine(workspace: Path) -> Engine:
    return Engine(context=object(), state=object(), workspace=str(workspace))


def test_valid_code_and_multiple_documents_share_one_changeset(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src/api.py").write_bytes(b"VERSION = 1\n")
    (tmp_path / "docs/spec.md").write_bytes(b"# API\nREQ-API-1 stable\n")
    (tmp_path / "docs/index.md").write_bytes(b"# Index\n[spec](spec.md#api)\n")
    result = _engine(tmp_path).validate_changeset({"changes": [
        {"path": "src/api.py", "operations": [
            {"kind": "search_replace", "search": "VERSION = 1", "replace": "VERSION = 2"},
        ]},
        {"path": "docs/spec.md", "operations": [
            {"kind": "replace_section", "heading": "API", "content": "REQ-API-1 stable\n"},
        ]},
        {"path": "docs/index.md", "operations": [
            {"kind": "insert_after_heading", "heading": "Index", "content": "[spec](spec.md#api)\n"},
        ]},
    ]})

    assert result["ok"] is True
    assert len(result["document_validation"]["matrix"]) == 2


def test_broken_link_anchor_and_mermaid_block_validation(tmp_path):
    (tmp_path / "doc.md").write_bytes(b"# Doc\n")
    result = _engine(tmp_path).validate_changeset({"changes": [{
        "path": "doc.md", "operations": [{"kind": "write_file", "content": (
            "# Doc\n[missing](none.md)\n[self](#unknown)\n"
            "```mermaid\nA[broken --> B\n"
        )}],
    }]})

    kinds = {item["kind"] for item in result["document_validation"]["errors"]}
    assert result["ok"] is False
    assert {"link_missing", "anchor_missing", "mermaid_unclosed"} <= kinds


def test_requirement_and_term_conflicts_across_changed_documents(tmp_path):
    (tmp_path / "a.md").write_bytes(b"# A\n")
    (tmp_path / "b.md").write_bytes(b"# B\n")
    result = _engine(tmp_path).validate_changeset({"changes": [
        {"path": "a.md", "operations": [{"kind": "write_file", "content":
            "# A\nREQ-X-1 must be fast\n**Timeout**: 10 seconds\n"}]},
        {"path": "b.md", "operations": [{"kind": "write_file", "content":
            "# B\nREQ-X-1 may be slow\n**Timeout**: 30 seconds\n"}]},
    ]})

    kinds = {item["kind"] for item in result["document_validation"]["errors"]}
    assert result["ok"] is False
    assert kinds == {"requirement_conflict", "term_conflict"}


def test_duplicate_headings_are_visible_warning_not_apply_blocker(tmp_path):
    (tmp_path / "doc.md").write_bytes(b"# Old\n")
    result = _engine(tmp_path).validate_changeset({"changes": [{
        "path": "doc.md", "operations": [{"kind": "write_file", "content": "# Same\n# Same\n"}],
    }]})

    assert result["ok"] is True
    assert result["document_validation"]["warnings"][0]["kind"] == "duplicate_heading"


def test_markdown_section_tool_applies_through_journaled_changeset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "doc.md"
    path.write_bytes(b"# Intro\nold\n# Next\nkeep\n")

    message = tools.replace_markdown_section("doc.md", "Intro", "new\n")

    assert message.startswith("Success:")
    assert path.read_bytes() == b"# Intro\nnew\n# Next\nkeep\n"
    journals = list((tmp_path / ".pixie_notes" / "changesets").glob("*.json"))
    assert len(journals) == 1
