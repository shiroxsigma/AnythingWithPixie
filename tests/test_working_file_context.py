from pathlib import Path

from paths import (
    bind_working_file_context,
    build_working_file_injection,
    create_working_file_context,
    reset_working_file_context,
)
from tools import read_file, search_and_replace
from engine import _build_dynamic_suffix
from state import AgentState


def _bound_context(tmp_path: Path, files: list[str]):
    context = create_working_file_context(tmp_path, files)
    token = bind_working_file_context(context)
    return context, token


def test_edit_replaces_old_content_in_injected_workset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "app.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    context, token = _bound_context(tmp_path, ["app.py"])
    try:
        before = build_working_file_injection()
        assert "VALUE = 1" in before
        assert search_and_replace("app.py", "VALUE = 1", "VALUE = 2").startswith("Success")

        after = build_working_file_injection()
        assert "VALUE = 2" in after
        assert "VALUE = 1" not in after
        assert "revision=2" in after
        assert len(context["entries"]) == 1
    finally:
        reset_working_file_context(token)


def test_external_change_refreshes_revision_and_rejects_stale_edit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "app.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    _, token = _bound_context(tmp_path, ["app.py"])
    try:
        path.write_text("VALUE = 3\n", encoding="utf-8")
        result = search_and_replace("app.py", "VALUE = 1", "VALUE = 2")
        assert result.startswith("Error")
        injection = build_working_file_injection()
        assert "VALUE = 3" in injection
        assert "VALUE = 1" not in injection
        assert "revision=2" in injection
    finally:
        reset_working_file_context(token)


def test_large_file_injection_is_bounded(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("x" * 20000, encoding="utf-8")
    _, token = _bound_context(tmp_path, ["large.txt"])
    try:
        injection = build_working_file_injection(max_chars=1000, per_file_chars=500)
        assert len(injection) < 1000
        assert "以降省略" in injection
    finally:
        reset_working_file_context(token)


def test_full_reread_of_fully_injected_working_file_returns_short_notice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "app.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    context, token = _bound_context(tmp_path, ["app.py"])
    try:
        build_working_file_injection()
        result = read_file("app.py")
        assert result.startswith("[Workset]")
        assert "VALUE = 1" not in result
        assert context["entries"][str(path.resolve())]["reread_count"] == 1
    finally:
        reset_working_file_context(token)


def test_explicit_range_can_reread_working_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "app.py"
    path.write_text("A = 1\nB = 2\n", encoding="utf-8")
    _, token = _bound_context(tmp_path, ["app.py"])
    try:
        build_working_file_injection()
        result = read_file("app.py", start_line=2, end_line=2)
        assert "2: B = 2" in result
    finally:
        reset_working_file_context(token)


def test_truncated_working_file_can_still_be_read_in_full(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "large.txt"
    path.write_text("x" * 2000, encoding="utf-8")
    _, token = _bound_context(tmp_path, ["large.txt"])
    try:
        build_working_file_injection(per_file_chars=100)
        result = read_file("large.txt")
        assert not result.startswith("[Workset]")
        assert "x" * 2000 in result
    finally:
        reset_working_file_context(token)


def test_file_omitted_from_latest_injection_is_not_suppressed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first body", encoding="utf-8")
    second.write_text("second body", encoding="utf-8")
    _, token = _bound_context(tmp_path, ["first.txt", "second.txt"])
    try:
        build_working_file_injection(max_chars=2000)
        assert read_file("second.txt").startswith("[Workset]")

        build_working_file_injection(max_chars=200, per_file_chars=200)
        result = read_file("second.txt")
        assert not result.startswith("[Workset]")
        assert "second body" in result
    finally:
        reset_working_file_context(token)


def test_working_files_are_dynamic_and_not_added_to_history(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("current only\n", encoding="utf-8")
    _, token = _bound_context(tmp_path, ["note.md"])
    try:
        state = AgentState()
        state.chat_history.add("user", "更新して")
        original = list(state.chat_history.messages)
        suffix = _build_dynamic_suffix(
            state, available_tools={"read_file"}, jit_input="更新して",
            thinking_mode="shallow", usage_ratio=0.1,
        )
        assert "current only" in suffix
        assert state.chat_history.messages == original
        assert "current only" not in str(state.chat_history.messages)
    finally:
        reset_working_file_context(token)

