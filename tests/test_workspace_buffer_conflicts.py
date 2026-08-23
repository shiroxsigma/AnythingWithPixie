"""未保存WorkspaceSnapshotをdisk競合から保護する契約。"""

import contextvars
import hashlib
import os

import pytest

import pixie_core
import tools
from paths import (
    build_working_file_injection,
    get_canonical_working_file_content,
    get_canonical_working_file_snapshots,
    get_workspace_buffer,
    get_workspace_snapshot_buffer,
)


_SERVER = {"base_url": "http://localhost:1/v1", "model": "test-model"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _engine(tmp_path):
    return pixie_core.create_engine(_SERVER, str(tmp_path))


def _bound(engine, fn):
    context = contextvars.copy_context()

    def run():
        engine._bind_context()
        return fn()

    return context.run(run)


def _edit(tool_name: str, path) -> str:
    if tool_name == "write_file":
        return tools.write_file(str(path), "written = True\n")
    if tool_name == "append_to_file":
        return tools.append_to_file(str(path), "appended = True\n")
    if tool_name == "replace_lines":
        return tools.replace_lines(str(path), 1, 1, "editor = 2")
    if tool_name == "search_and_replace":
        return tools.search_and_replace(str(path), "editor = 1", "editor = 2")
    raise AssertionError(tool_name)


@pytest.mark.parametrize(
    "tool_name",
    ["write_file", "append_to_file", "replace_lines", "search_and_replace"],
)
def test_buffer_edit_tools_reject_external_disk_change_without_writing(tmp_path, tool_name):
    target = tmp_path / "app.py"
    original = b"disk = 1\r\n"
    external = b"external = 3\r\n"
    target.write_bytes(original)
    engine = _engine(tmp_path)
    engine.set_workspace_snapshot({
        "buffers": [{"path": "app.py", "content": "editor = 1\nsecond = 2\n"}],
    })
    assert engine._workspace_buffers[str(target.resolve())]["base_hash"] == _sha(original)
    target.write_bytes(external)

    result = _bound(engine, lambda: _edit(tool_name, target))

    assert result.startswith("Error: CONFLICT:")
    assert target.read_bytes() == external
    assert engine._workspace_buffers[str(target.resolve())]["content"] == (
        "editor = 1\nsecond = 2\n"
    )


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("write_file", "written = True\n"),
        ("append_to_file", "editor = 1\nsecond = 2\nappended = True\n"),
        ("replace_lines", "editor = 2\nsecond = 2\n"),
        ("search_and_replace", "editor = 2\nsecond = 2\n"),
    ],
)
def test_buffer_edit_tools_write_when_disk_base_is_unchanged_and_advance_hash(
    tmp_path, tool_name, expected,
):
    target = tmp_path / "app.py"
    target.write_bytes(b"disk = 1\n")
    engine = _engine(tmp_path)
    engine.set_workspace_snapshot({
        "buffers": [{"path": "app.py", "content": "editor = 1\nsecond = 2\n"}],
    })

    result = _bound(engine, lambda: _edit(tool_name, target))

    assert result.startswith("Success:")
    assert target.read_text(encoding="utf-8") == expected
    current = engine._workspace_buffers[str(target.resolve())]
    assert current["content"] == expected
    assert current["base_hash"] == _sha(target.read_bytes())


def test_two_buffer_edits_in_same_turn_do_not_self_conflict(tmp_path):
    target = tmp_path / "app.py"
    target.write_bytes(b"disk = 1\n")
    engine = _engine(tmp_path)
    engine.set_workspace_snapshot({
        "buffers": [{"path": "app.py", "content": "value = 1\n"}],
    })

    def edit_twice():
        first = tools.search_and_replace(str(target), "value = 1", "value = 2")
        after_first = engine._workspace_buffers[str(target.resolve())]["base_hash"]
        first_disk = target.read_bytes()
        second = tools.append_to_file(str(target), "done = True\n")
        return first, after_first, first_disk, second

    first, after_first, first_disk, second = _bound(engine, edit_twice)

    assert first.startswith("Success:")
    assert after_first == _sha(first_disk)
    assert second.startswith("Success:")
    assert target.read_text(encoding="utf-8") == "value = 2\ndone = True\n"
    assert engine._workspace_buffers[str(target.resolve())]["base_hash"] == _sha(
        target.read_bytes()
    )


def test_omitted_base_hash_for_new_file_uses_empty_bytes(tmp_path):
    target = tmp_path / "new.txt"
    engine = _engine(tmp_path)
    engine.set_workspace_snapshot({
        "buffers": [{"path": r"new.txt", "content": "unsaved\n"}],
    })

    assert engine._workspace_buffers[str(target.resolve())]["base_hash"] == _sha(b"")
    result = _bound(engine, lambda: tools.append_to_file(str(target), "added\n"))

    assert result.startswith("Success:")
    assert target.read_text(encoding="utf-8") == "unsaved\nadded\n"


def test_document_changeset_uses_buffer_base_hash_and_preserves_disk_on_conflict(tmp_path):
    target = tmp_path / "doc.md"
    target.write_bytes(b"# Intro\ndisk body\n")
    engine = _engine(tmp_path)
    engine.set_workspace_snapshot({
        "buffers": [{"path": "doc.md", "content": "# Intro\neditor body\n"}],
    })
    target.write_bytes(b"# Intro\nexternal body\n")

    result = _bound(
        engine,
        lambda: tools.replace_markdown_section(str(target), "Intro", "agent body\n"),
    )

    assert result.startswith("Error:")
    assert "conflict" in result.lower() or "expected" in result.lower()
    assert target.read_bytes() == b"# Intro\nexternal body\n"


def test_document_changeset_advances_buffer_hash_after_success(tmp_path):
    target = tmp_path / "doc.md"
    target.write_bytes(b"# Intro\ndisk body\n")
    engine = _engine(tmp_path)
    engine.set_workspace_snapshot({
        "buffers": [{"path": "doc.md", "content": "# Intro\neditor body\n"}],
    })

    result = _bound(
        engine,
        lambda: tools.replace_markdown_section(str(target), "Intro", "agent body\n"),
    )

    assert result.startswith("Success:")
    assert target.read_bytes() == b"# Intro\nagent body\n"
    assert engine._workspace_buffers[str(target.resolve())]["base_hash"] == _sha(
        target.read_bytes()
    )


def test_delete_marks_buffer_tombstone_without_discarding_ui_content(tmp_path):
    target = tmp_path / "note.md"
    target.write_bytes(b"disk\n")
    engine = _engine(tmp_path)
    engine.set_working_files(["note.md"])
    engine.set_workspace_snapshot({
        "buffers": [{"path": "note.md", "content": "unsaved editor\n"}],
    })

    def delete_and_inspect():
        result = tools.delete_file(str(target))
        saved = get_workspace_snapshot_buffer(str(target))
        return result, saved, get_workspace_buffer(str(target)), get_canonical_working_file_snapshots()

    result, saved, readable, snapshots = _bound(engine, delete_and_inspect)

    assert result.startswith("Success:")
    assert not target.exists()
    assert saved == {
        "content": "unsaved editor\n",
        "base_hash": _sha(b""),
        "deleted": True,
    }
    assert readable is None
    assert snapshots["note.md"]["content"] == "unsaved editor\n"
    assert snapshots["note.md"]["status"] == "deleted"


def test_move_migrates_buffer_and_workset_without_flushing_unsaved_content(tmp_path):
    source = tmp_path / "old.md"
    destination = tmp_path / "new.md"
    source.write_bytes(b"disk\n")
    engine = _engine(tmp_path)
    engine.set_working_files(["old.md"])
    engine.set_workspace_snapshot({
        "buffers": [{"path": r"old.md", "content": "unsaved editor\n"}],
    })

    def move_and_inspect():
        result = tools.move_file(str(source), str(destination))
        return (
            result,
            get_workspace_snapshot_buffer(str(source)),
            get_workspace_snapshot_buffer(str(destination)),
            get_canonical_working_file_snapshots(),
        )

    result, old_buffer, new_buffer, snapshots = _bound(engine, move_and_inspect)

    assert result.startswith("Success:")
    assert not source.exists()
    assert destination.read_bytes() == b"disk\n"
    assert old_buffer is None
    assert new_buffer["content"] == "unsaved editor\n"
    assert new_buffer["base_hash"] == _sha(b"disk\n")
    assert "old.md" not in snapshots
    assert snapshots["new.md"]["content"] == "unsaved editor\n"


def test_run_command_disk_change_makes_canonical_verification_fail_closed(tmp_path):
    target = tmp_path / "tracked.txt"
    target.write_bytes(b"disk\n")
    engine = _engine(tmp_path)
    engine.set_working_files(["tracked.txt"])
    engine.set_workspace_snapshot({
        "buffers": [{"path": "tracked.txt", "content": "unsaved editor\n"}],
    })
    command = (
        "Set-Content -LiteralPath tracked.txt -Value command_changed -NoNewline"
        if os.name == "nt"
        else "printf command_changed > tracked.txt"
    )

    def run_and_inspect():
        result = tools.run_command(command)
        return (
            result,
            get_canonical_working_file_content("tracked.txt"),
            get_workspace_snapshot_buffer("tracked.txt"),
        )

    result, canonical, saved = _bound(engine, run_and_inspect)

    assert not result.startswith(("Error", "Execution Failed"))
    assert canonical is not None
    assert canonical["content"] == "unsaved editor\n"
    assert canonical["status"] == "conflict"
    assert canonical["conflict"]["expected"] == _sha(b"disk\n")
    assert saved["content"] == "unsaved editor\n"
    assert saved["base_hash"] == _sha(b"disk\n")


def test_recreated_empty_file_clears_tombstone_and_refreshes_revision(tmp_path):
    target = tmp_path / "note.md"
    target.write_bytes(b"disk\n")
    engine = _engine(tmp_path)
    engine.set_working_files(["note.md"])
    engine.set_workspace_snapshot({
        "buffers": [{"path": "note.md", "content": "unsaved editor\n"}],
    })

    def delete_recreate_and_inspect():
        assert tools.delete_file(str(target)).startswith("Success:")
        deleted_revision = next(iter(engine._working_file_context["entries"].values()))[
            "revision"
        ]
        target.write_bytes(b"")
        revived = get_workspace_buffer(str(target))
        entry = next(iter(engine._working_file_context["entries"].values()))
        canonical = get_canonical_working_file_content("note.md")
        injection = build_working_file_injection()
        return deleted_revision, revived, entry, canonical, injection

    deleted_revision, revived, entry, canonical, injection = _bound(
        engine, delete_recreate_and_inspect
    )

    assert revived["content"] == "unsaved editor\n"
    assert "deleted" not in revived
    assert "deleted" not in entry
    assert entry["revision"] > deleted_revision
    assert canonical is not None
    assert canonical["content"] == "unsaved editor\n"
    assert "status" not in canonical
    assert "unsaved editor" in injection
