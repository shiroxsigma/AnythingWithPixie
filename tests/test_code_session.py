"""コードセッション強化のテスト: outline AST化、get_code_outline、
project_structure 永続化、ruff検証、code_tool.py 移行登録。"""

import pytest

from code_index import outline
from code_tool import get_code_outline
from config import CODE_TOOL_SET
from state import AgentStateBoard
from tools import TOOL_REGISTRY

# =====================================================
# code_index.outline (AST)
# =====================================================

def test_outline_nested_class_methods(tmp_path):
    """ネストクラス+メソッド+デコレータをASTで正確に抽出。"""
    src = tmp_path / "sample.py"
    src.write_text(
        "import functools\n"
        "\n"
        "@functools.lru_cache\n"
        "def top_func():\n"
        "    return 1\n"
        "\n"
        "class Outer:\n"
        "    def method_a(self):\n"
        "        return 2\n"
        "    class Inner:\n"
        "        def inner_method(self):\n"
        "            return 3\n",
        encoding="utf-8",
    )
    syms = outline(src)
    names = [(s["name"], s["kind"]) for s in syms]
    assert ("top_func", "function") in names
    assert ("Outer", "class") in names
    assert ("method_a", "method") in names
    assert ("Inner", "class") in names
    assert ("inner_method", "method") in names
    qualnames = [s["qualname"] for s in syms]
    assert "Outer.method_a" in qualnames
    assert "Outer.Inner.inner_method" in qualnames
    top = next(s for s in syms if s["name"] == "top_func")
    assert any("lru_cache" in d for d in top.get("decorators", []))
    assert all(s.get("end_lineno") and s["end_lineno"] >= s["lineno"] for s in syms)


def test_outline_non_py_returns_empty(tmp_path):
    src = tmp_path / "a.js"
    src.write_text("function f() {}", encoding="utf-8")
    assert outline(src) == []


def test_outline_missing_file_returns_empty(tmp_path):
    assert outline(tmp_path / "nope.py") == []


# =====================================================
# get_code_outline (AST for .py, regex for JS/TS)
# =====================================================

def test_get_code_outline_py_uses_ast(tmp_path):
    src = tmp_path / "m.py"
    src.write_text(
        "class C:\n"
        "    def m(self):\n"
        "        pass\n",
        encoding="utf-8",
    )
    out = get_code_outline(str(src))
    assert "C" in out
    assert "m" in out  # ネストメソッドが AST で抽出される


def test_get_code_outline_js_kept_regex(tmp_path):
    src = tmp_path / "a.ts"
    src.write_text(
        "export function foo() {}\n"
        "const bar = () => {}\n",
        encoding="utf-8",
    )
    out = get_code_outline(str(src))
    assert "foo" in out
    assert "bar" in out


def test_get_code_outline_fallback_on_syntax_error(tmp_path):
    """構文エラーでも正規表現フォールバックでトップレベル定義を抽出。"""
    src = tmp_path / "bad.py"
    src.write_text("def good():\n    return 1\n", encoding="utf-8")
    out = get_code_outline(str(src))
    assert "good" in out


# =====================================================
# project_structure 永続化 (AgentStateBoard)
# =====================================================

def test_state_board_project_structure_persists(tmp_path):
    board = AgentStateBoard(file_path=str(tmp_path / "sb.json"))
    board.project_structure = "[Tree]\nfoo"
    board._save()
    reloaded = AgentStateBoard(file_path=str(tmp_path / "sb.json"))
    assert reloaded.project_structure == "[Tree]\nfoo"


def test_state_board_injection_truncates_project_structure(tmp_path):
    board = AgentStateBoard(file_path=str(tmp_path / "sb.json"))
    board.project_structure = "x" * 2000
    text = board.to_injection_text()
    assert "プロジェクト構造" in text
    assert "省略" in text
    assert len(text) < 2000  # 1500 chars 切詰め + ヘッダで 2000 未満


def test_state_board_clear_resets_project_structure(tmp_path):
    board = AgentStateBoard(file_path=str(tmp_path / "sb.json"))
    board.project_structure = "foo"
    board.clear()
    assert board.project_structure == ""


def test_state_board_not_empty_with_project_structure(tmp_path):
    board = AgentStateBoard(file_path=str(tmp_path / "sb.json"))
    assert board.is_empty()
    board.project_structure = "foo"
    assert not board.is_empty()


# =====================================================
# ruff 自動検証 (_run_ruff_check)
# =====================================================

def test_run_ruff_check_clean_py(tmp_path):
    import engine
    src = tmp_path / "clean.py"
    src.write_text("x = 1\n", encoding="utf-8")
    assert engine._run_ruff_check(str(src)) == ""


def test_run_ruff_check_undefined_name(tmp_path):
    pytest.importorskip("ruff")  # ruff 未環境は skip
    import engine
    src = tmp_path / "bad.py"
    src.write_text("print(undefined_var)\n", encoding="utf-8")
    out = engine._run_ruff_check(str(src))
    assert out != ""
    assert "undefined_var" in out or "F821" in out


def test_run_ruff_check_non_py_returns_empty(tmp_path):
    import engine
    src = tmp_path / "a.txt"
    src.write_text("hello", encoding="utf-8")
    assert engine._run_ruff_check(str(src)) == ""


def test_run_ruff_check_missing_file_returns_empty():
    import engine
    assert engine._run_ruff_check("/nonexistent/path.py") == ""


# =====================================================
# code_tool.py 移行登録（回帰ガード）
# =====================================================

def test_code_tools_registered():
    """code_tool.py の7ツールが TOOL_REGISTRY に登録されている。"""
    expected = {
        "get_code_outline", "research_code_paths", "gather_project_info",
        "map_codebase", "detect_dead_code", "read_symbol", "get_file_stats",
    }
    assert expected <= set(TOOL_REGISTRY.keys())


def test_get_code_outline_in_code_tool_set():
    assert "get_code_outline" in CODE_TOOL_SET
