import hashlib
import json
from pathlib import Path

from pixie_core._api import Engine
from state import AgentState


def _engine(workspace: Path) -> Engine:
    return Engine(context=object(), state=object(), workspace=str(workspace))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_approval_applied_changeset_allows_post_edit_read(tmp_path, monkeypatch):
    import engine as graph
    from test_acceptance_gate_engine import _MockLLM, _context, _tc

    path = tmp_path / "README.md"
    path.write_text("before\n", encoding="utf-8")
    read = _tc("read_file", {"path": str(path)})
    write = _tc("write_file", {"path": str(path), "content": "after\n"})
    llm = _MockLLM([(None, read), (None, write), (None, read), ("DONE", None)])
    context = _context(llm)
    context.code_mode = True
    state = AgentState()
    state.chat_history.add("user", "Read README.md, change before to after, then read it to verify.")
    api = Engine(context=context, state=state, workspace=str(tmp_path))
    monkeypatch.setattr(graph, "derive_acceptance", lambda *a: [])
    monkeypatch.setattr(graph, "LESSONS_ENABLED", False)
    monkeypatch.setattr(graph, "BEST_OF_ANSWER_ENABLED", False)
    output = []

    def approve(calls, content):
        if calls[0]["function"]["name"] == "write_file":
            state.loop_warn_count = 2
            applied = api.apply_changeset({"id": "chg_hook", "changes": [{
                "path": "README.md", "operations": [{
                    "kind": "write_file", "content": "after\n",
                }],
            }]})
            assert applied["applied"]
            assert state.loop_warn_count == 0
            return [], "The approved edit has been applied. Read the file to verify."
        return calls, None

    graph.run_graph(context, state, interactive_fn=approve,
                    output_fn=lambda text="", **kwargs: output.append(text))
    assert path.read_text(encoding="utf-8") == "after\n"
    assert sum(a.startswith("read_file:") for a in state.executed_actions) == 2
    assert "apply_changeset:chg_hook" in state.executed_actions
    assert not any("連続ループ" in text for text in output)
    assert any("after" in str(m) for m in llm.captured[-1])


def test_changeset_preview_and_failed_apply_do_not_record_success(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("before", encoding="utf-8")
    state = AgentState()
    state.executed_actions.append("read_file:previous")
    state.loop_warn_count = 2
    api = Engine(context=object(), state=state, workspace=str(tmp_path))
    change = {"id": "chg_unapproved", "changes": [{
        "path": "README.md", "base_hash": _sha(b"before"),
        "operations": [{"kind": "write_file", "content": "after"}],
    }]}
    assert api.preview_changeset(change)["ok"]
    # A rejected approval ends at the preview; it must not create a mutation.
    assert state.executed_actions == ["read_file:previous"]
    assert path.read_text(encoding="utf-8") == "before"
    path.write_text("external", encoding="utf-8")
    assert not api.apply_changeset(change)["applied"]
    assert state.executed_actions == ["read_file:previous"]
    assert state.loop_warn_count == 2


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
