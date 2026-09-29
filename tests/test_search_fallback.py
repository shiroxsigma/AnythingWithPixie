"""Windows without rg still has a bounded, usable grep_search tool."""

import pytest

from pixie_core import search_fallback, tools
from pixie_core.turn_control import TurnControl, TurnStopped, active_control


@pytest.fixture
def no_rg(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tools, "get_bundled_path", lambda name: str(tmp_path / "missing-rg.exe"))
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)


def test_literal_regex_extensions_context_and_exclusions(no_rg, tmp_path):
    (tmp_path / "a.py").write_text("before\nTODO. TODOx\nafter\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("TODO.\n", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("TODO.\n", encoding="utf-8")
    for ignored in (".git", "node_modules", ".venv"):
        folder = tmp_path / ignored
        folder.mkdir()
        (folder / "hidden.py").write_text("TODO.\n", encoding="utf-8")
    (tmp_path / "binary.py").write_bytes(b"\x00TODO.")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(tmp_path / "a.py")
    except OSError:
        pass  # 権限のない Windows 環境では symlink を作れない。

    literal = tools.grep_search("TODO.", path=str(tmp_path), file_extensions=".py", context_lines=1)
    assert "2:TODO. TODOx" in literal
    assert "1-before" in literal and "3-after" in literal
    assert "total_matches=1" in literal
    assert "Search summary (authoritative)" in literal
    assert all(name not in literal for name in ("b.md", "hidden.py", "binary.py", "linked.py"))

    regex = tools.grep_search("TODO.", path=str(tmp_path), is_regex=True, context_lines=0)
    assert "total_matches=3" in regex
    assert "b.md" in regex


def test_scan_and_output_limits_are_explicit(no_rg, tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(search_fallback, "MAX_FILES", 1)
    partial = tools.grep_search("needle", path=str(tmp_path), context_lines=0)
    assert "Search summary (partial): total_matches=1" in partial
    assert "truncated=true (scan=true" in partial
    assert "Use total_matches as the answer" not in partial

    monkeypatch.setattr(search_fallback, "MAX_FILES", 2_000)
    (tmp_path / "long.txt").write_text(("needle " + "x" * 100 + "\n") * 150, encoding="utf-8")
    result = tools.grep_search("needle", path=str(tmp_path), context_lines=0)
    assert "total_matches=152" in result
    assert "truncated=true (scan=false, output=true)" in result
    assert len(result) <= search_fallback.MAX_OUTPUT_CHARS


def test_cancelled_turn_stops_python_search(no_rg, tmp_path):
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
    control = TurnControl()
    control.cancel()
    token = active_control.set(control)
    try:
        with pytest.raises(TurnStopped):
            tools.grep_search("needle", path=str(tmp_path))
    finally:
        active_control.reset(token)


def test_match_and_file_size_limits_mark_partial(no_rg, tmp_path, monkeypatch):
    (tmp_path / "many.txt").write_text("needle needle\n", encoding="utf-8")
    monkeypatch.setattr(search_fallback, "MAX_MATCHES", 1)
    capped = tools.grep_search("needle", path=str(tmp_path), context_lines=0)
    assert "Search summary (partial): total_matches=1" in capped
    assert "truncated=true (scan=true" in capped

    monkeypatch.setattr(search_fallback, "MAX_MATCHES", 10_000)
    monkeypatch.setattr(search_fallback, "MAX_FILE_BYTES", 4)
    skipped = tools.grep_search("needle", path=str(tmp_path), context_lines=0)
    assert "No matches found" in skipped
    assert "truncated=true; scan limit reached" in skipped
