import hashlib
import os
from pathlib import Path

import pytest

from paths import (
    bind_working_file_context,
    bind_workspace_buffers,
    build_working_file_injection,
    create_working_file_context,
    get_canonical_working_file_content,
    get_canonical_working_file_snapshots,
    get_workspace_buffer,
    get_working_file_snapshots,
    reset_working_file_context,
    reset_workspace_buffers,
    update_workspace_buffer,
)


def _bind(tmp_path: Path, relative_path: str, buffer_content: str | None = None):
    context = create_working_file_context(tmp_path, [relative_path])
    working_token = bind_working_file_context(context)
    target = (tmp_path / relative_path).resolve()
    buffers = (
        {str(target): {"content": buffer_content, "base_hash": None}}
        if buffer_content is not None else {}
    )
    buffer_token = bind_workspace_buffers(buffers)
    return target, working_token, buffer_token


def test_turn_snapshot_uses_unsaved_buffer_as_acceptance_baseline(tmp_path):
    target = tmp_path / "version.py"
    target.write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    target, working_token, buffer_token = _bind(
        tmp_path, "version.py", '__version__ = "1.0.1-dev"\n'
    )
    try:
        snapshots = get_canonical_working_file_snapshots()
        expected = '__version__ = "1.0.1-dev"\n'
        assert snapshots["version.py"] == {
            "content": expected,
            "initial_hash": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
        }
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


def test_canonical_content_advances_after_buffer_edit(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    target, working_token, buffer_token = _bind(tmp_path, "app.py", "VALUE = 1\n")
    try:
        update_workspace_buffer(target, "VALUE = 2\n")

        current = get_canonical_working_file_content("app.py")
        assert current is not None
        assert current["content"] == "VALUE = 2\n"
        assert current["source"] == "buffer"
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


def test_windows_separator_is_normalized_to_posix_relative_path(tmp_path):
    nested = tmp_path / "src"
    nested.mkdir()
    target = nested / "app.py"
    target.write_text("pass\n", encoding="utf-8")
    _, working_token, buffer_token = _bind(tmp_path, "src/app.py")
    try:
        current = get_canonical_working_file_content(r"src\app.py")
        assert current is not None
        assert current["path"] == "src/app.py"
        assert current["content"] == "pass\n"
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


def test_external_disk_change_does_not_replace_or_write_unsaved_buffer(tmp_path):
    target = tmp_path / "note.md"
    target.write_text("disk before\n", encoding="utf-8")
    target, working_token, buffer_token = _bind(tmp_path, "note.md", "unsaved editor\n")
    try:
        target.write_text("external disk change\n", encoding="utf-8")

        current = get_canonical_working_file_content("note.md")
        assert current is not None
        assert current["content"] == "unsaved editor\n"
        assert current["source"] == "buffer"
        assert target.read_text(encoding="utf-8") == "external disk change\n"
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


def test_disk_change_refreshes_workset_only_when_no_buffer_exists(tmp_path):
    target = tmp_path / "note.md"
    target.write_text("before\n", encoding="utf-8")
    target, working_token, buffer_token = _bind(tmp_path, "note.md")
    try:
        target.write_text("external current\n", encoding="utf-8")

        current = get_canonical_working_file_content("note.md")
        assert current is not None
        assert current["content"] == "external current\n"
        assert current["source"] == "working_file"
        assert get_canonical_working_file_snapshots()["note.md"]["content"] == "external current\n"
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


def test_legacy_working_file_snapshot_remains_registration_time_content(tmp_path):
    target = tmp_path / "note.md"
    target.write_text("initial\n", encoding="utf-8")
    target, working_token, buffer_token = _bind(tmp_path, "note.md")
    try:
        target.write_text("later\n", encoding="utf-8")

        assert get_working_file_snapshots()["note.md"]["content"] == "initial\n"
        assert get_canonical_working_file_snapshots()["note.md"]["content"] == "later\n"
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


def test_deleted_disk_file_is_not_resurrected_from_stale_workset(tmp_path):
    target = tmp_path / "gone.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    target, working_token, buffer_token = _bind(tmp_path, "gone.py")
    try:
        target.unlink()

        assert get_canonical_working_file_content("gone.py") is None
        assert "gone.py" not in get_canonical_working_file_snapshots()
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


def test_buffer_lookup_and_update_accept_windows_separator(tmp_path):
    nested = tmp_path / "src"
    nested.mkdir()
    target = nested / "app.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    _, working_token, buffer_token = _bind(
        tmp_path, "src/app.py", "VALUE = 2\n"
    )
    try:
        assert get_workspace_buffer(r"src\app.py")["content"] == "VALUE = 2\n"
        update_workspace_buffer(r"src\app.py", "VALUE = 3\n")
        assert get_workspace_buffer("src/app.py")["content"] == "VALUE = 3\n"
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


@pytest.mark.skipif(os.name != "nt", reason="Windows path identity semantics")
def test_buffer_lookup_accepts_windows_case_difference(tmp_path):
    nested = tmp_path / "src"
    nested.mkdir()
    target = nested / "app.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    _, working_token, buffer_token = _bind(
        tmp_path, "src/app.py", "VALUE = 2\n"
    )
    try:
        assert get_workspace_buffer(r"SRC\APP.PY")["content"] == "VALUE = 2\n"
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)


@pytest.mark.skipif(os.name != "nt", reason="Windows path identity semantics")
def test_workset_injection_uses_case_insensitive_buffer_identity(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("DISK = 1\n", encoding="utf-8")
    context = create_working_file_context(tmp_path, ["app.py"])
    working_token = bind_working_file_context(context)
    buffer_token = bind_workspace_buffers({
        str(target.resolve()).upper(): {"content": "BUFFER = 2\n", "base_hash": None}
    })
    try:
        rendered = build_working_file_injection()
        assert "BUFFER = 2" in rendered
        assert "DISK = 1" not in rendered
    finally:
        reset_workspace_buffers(buffer_token)
        reset_working_file_context(working_token)
