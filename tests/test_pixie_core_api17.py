"""pixie_core API 1.7 の WorkspaceSnapshot 契約。"""

import contextvars
from pathlib import Path

import pytest

import pixie_core
import tools


_SERVER = {"base_url": "http://localhost:1/v1", "model": "test-model"}


def _engine(tmp_path):
    return pixie_core.create_engine(_SERVER, str(tmp_path))


def _bound(eng, fn):
    """run_turn と同じく、束縛が呼出スレッドへ漏れない独立 context で実行する。"""
    ctx = contextvars.copy_context()

    def run():
        eng._bind_context()
        return fn()

    return ctx.run(run)


def test_workspace_snapshot_read_file_prefers_unsaved_buffer(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("disk version\n", encoding="utf-8")
    eng = _engine(tmp_path)
    eng.set_workspace_snapshot({
        "buffers": [{"path": "note.md", "content": "editor version\n"}],
    })

    result = _bound(eng, lambda: tools.read_file(str(path)))

    assert "editor version" in result
    assert "disk version" not in result
    assert "未保存バッファ" in result


def test_workspace_snapshot_can_be_cleared(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("disk version\n", encoding="utf-8")
    eng = _engine(tmp_path)
    eng.set_workspace_snapshot({
        "buffers": [{"path": "note.md", "content": "editor version\n"}],
    })
    eng.set_workspace_snapshot(None)

    assert "disk version" in _bound(eng, lambda: tools.read_file(str(path)))


def test_set_working_files_registers_replaceable_latest_context(tmp_path):
    path = tmp_path / "target.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    eng = _engine(tmp_path)
    eng.set_working_files(["target.py"])

    def inspect():
        from paths import build_working_file_injection
        return build_working_file_injection()

    assert "VALUE = 1" in _bound(eng, inspect)
    path.write_text("VALUE = 2\n", encoding="utf-8")
    refreshed = _bound(eng, inspect)
    assert "VALUE = 2" in refreshed
    assert "VALUE = 1" not in refreshed


def test_workspace_snapshot_rejects_path_outside_workspace(tmp_path):
    eng = _engine(tmp_path)
    outside = Path(tmp_path).parent / "outside.txt"

    with pytest.raises(ValueError, match="workspace 外"):
        eng.set_workspace_snapshot({
            "buffers": [{"path": str(outside), "content": "secret"}],
        })


def test_workspace_snapshot_is_isolated_between_engines(tmp_path):
    path = tmp_path / "same.txt"
    path.write_text("disk\n", encoding="utf-8")
    first = _engine(tmp_path)
    second = _engine(tmp_path)
    first.set_workspace_snapshot({"buffers": [{"path": "same.txt", "content": "first\n"}]})
    second.set_workspace_snapshot({"buffers": [{"path": "same.txt", "content": "second\n"}]})

    assert "first" in _bound(first, lambda: tools.read_file(str(path)))
    assert "second" in _bound(second, lambda: tools.read_file(str(path)))


def test_search_and_replace_uses_and_preserves_unsaved_buffer(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("disk = True\n", encoding="utf-8")
    eng = _engine(tmp_path)
    eng.set_workspace_snapshot({
        "buffers": [{"path": "code.py", "content": "user_edit = True\nvalue = 1\n"}],
    })
    def edit_and_read():
        result = tools.search_and_replace(str(path), "value = 1", "value = 2")
        return result, tools.read_file(str(path))

    result, current = _bound(eng, edit_and_read)

    assert result.startswith("Success:")
    assert path.read_text(encoding="utf-8") == "user_edit = True\nvalue = 2\n"
    assert "value = 2" in current
