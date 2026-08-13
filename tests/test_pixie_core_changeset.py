import hashlib
import json
from pathlib import Path

from pixie_core._api import Engine


def _engine(workspace: Path) -> Engine:
    return Engine(context=object(), state=object(), workspace=str(workspace))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_preview_composes_multiple_files_and_operations_without_writing(tmp_path):
    py = tmp_path / "src" / "app.py"
    md = tmp_path / "docs" / "guide.md"
    py.parent.mkdir()
    md.parent.mkdir()
    py.write_bytes(b"value = 1\nlast = True\n")
    md.write_bytes(b"# Guide\nold\n\n# Next\nkeep\n")
    change = {
        "id": "chg_preview",
        "changes": [
            {"path": "src/app.py", "operations": [
                {"kind": "search_replace", "search": "value = 1", "replace": "value = 2"},
                {"kind": "append", "content": "added = True\n"},
            ]},
            {"path": "docs/guide.md", "operations": [
                {"kind": "replace_section", "heading": "Guide", "content": "new text\n\n"},
            ]},
        ],
    }

    result = _engine(tmp_path).preview_changeset(change)

    assert result["ok"] is True
    assert result["id"] == "chg_preview"
    assert result["changes"][0]["after"] == "value = 2\nlast = True\nadded = True\n"
    assert result["changes"][1]["after"] == "# Guide\nnew text\n# Next\nkeep\n"
    assert py.read_text(encoding="utf-8") == "value = 1\nlast = True\n"


def test_validate_detects_base_hash_conflict(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"new disk\n")
    result = _engine(tmp_path).validate_changeset({"changes": [{
        "path": "a.py", "base_hash": _sha(b"old disk\n"),
        "operations": [{"kind": "write_file", "content": "after\n"}],
    }]})

    assert result["ok"] is False
    assert result["conflicts"][0]["path"] == "a.py"
    assert path.read_text(encoding="utf-8") == "new disk\n"


def test_apply_and_revert_restore_modified_created_and_deleted_files(tmp_path):
    modified = tmp_path / "modified.txt"
    deleted = tmp_path / "deleted.txt"
    modified.write_bytes(b"before\n")
    deleted.write_bytes(b"remove me\n")
    change = {"id": "chg_roundtrip", "changes": [
        {"path": "modified.txt", "base_hash": _sha(b"before\n"),
         "operations": [{"kind": "write_file", "content": "after\n"}]},
        {"path": "created.txt", "base_hash": _sha(b""),
         "operations": [{"kind": "write_file", "content": "created\n"}]},
        {"path": "deleted.txt", "base_hash": _sha(b"remove me\n"),
         "operations": [{"kind": "delete_file"}]},
    ]}
    engine = _engine(tmp_path)

    applied = engine.apply_changeset(change)

    assert applied["applied"] is True
    assert modified.read_text(encoding="utf-8") == "after\n"
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created\n"
    assert not deleted.exists()
    journal = tmp_path / applied["journal"]
    assert json.loads(journal.read_text(encoding="utf-8"))["status"] == "applied"

    reverted = engine.revert_changeset("chg_roundtrip")

    assert reverted["reverted"] is True
    assert modified.read_text(encoding="utf-8") == "before\n"
    assert not (tmp_path / "created.txt").exists()
    assert deleted.read_text(encoding="utf-8") == "remove me\n"
    assert json.loads(journal.read_text(encoding="utf-8"))["status"] == "reverted"


def test_revert_refuses_to_overwrite_external_change(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"before")
    engine = _engine(tmp_path)
    assert engine.apply_changeset({"id": "chg_conflict", "changes": [{
        "path": "a.txt", "operations": [{"kind": "write_file", "content": "after"}],
    }]})["applied"]
    path.write_bytes(b"external")

    result = engine.revert_changeset("chg_conflict")

    assert result["reverted"] is False
    assert result["conflicts"][0]["path"] == "a.txt"
    assert path.read_text(encoding="utf-8") == "external"


def test_document_frontmatter_and_heading_operations(tmp_path):
    path = tmp_path / "doc.md"
    path.write_bytes(b"---\ntitle: Old\n---\n# Intro\nbody\n")
    result = _engine(tmp_path).preview_changeset({"changes": [{
        "path": "doc.md", "operations": [
            {"kind": "update_frontmatter", "values": {"title": "New", "draft": False}},
            {"kind": "insert_after_heading", "heading": "Intro", "content": "inserted\n"},
        ],
    }]})

    assert result["ok"]
    assert "title: New" in result["changes"][0]["after"]
    assert "draft: false" in result["changes"][0]["after"]
    assert "# Intro\ninserted\nbody" in result["changes"][0]["after"]
