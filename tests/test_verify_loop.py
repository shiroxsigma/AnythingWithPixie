"""verify → fix（実行ベース検証 → 自動再編集）ループのテスト（pure・モデル不要）。

/verify トグルで起動する run_verify_fix_loop と、その構成要素
（.venv 解決・py_compile・段階ゲート・修正 JSON 生成・ループ制御・execute_tool フック）
を、決定的な値とスクリプト化モック LLM で検証する。
"""

import os
import sys

import pytest

from config import VERIFY_MAX_ROUNDS
from engine import execute_tool
from paths import resolve_venv_python
from subagent import (
    _generate_fix_edit,
    _resolve_verify_python,
    _run_execution_verification,
    _run_import_check,
    _run_py_compile,
    run_fast_gate_check,
    run_verify_fix_loop,
)

# =====================================================
# ヘルパ
# =====================================================

class _FixMockLLM:
    """create_chat_completion が指定 content を1回返すモック（LM Studio 互換 generator）。"""

    def __init__(self, content):
        self.content = content

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream, **kw):
        c = self.content

        def _gen():
            yield {
                "choices": [{
                    "delta": {"content": c},
                    "finish_reason": "stop",
                }]
            }

        return _gen()


class _ExplodingLLM:
    """create_chat_completion が常に例外を投げるモック（例外安全テスト用）。"""

    def create_chat_completion(self, *a, **kw):
        raise RuntimeError("LLM boom")


class _Ctx:
    """run_verify_fix_loop 用の最小 context（llm のみ必要）。"""

    def __init__(self, llm):
        self.llm = llm
        self.supports_tool_role = False


class _FullCtx:
    """execute_tool 用の最小 context。"""

    def __init__(self, *, verify_mode=False, review_mode=False):
        self.llm = None
        self.use_vision = False
        self.supports_tool_role = False
        self.verify_mode = verify_mode
        self.review_mode = review_mode


def _outfn():
    """output_fn 互換（end/flush 等のキーワードを受け付ける）バッファ。

    本番の output_fn は print（end/flush を取る）なので、モックもそれに合わせる。
    戻り値の関数は .buf に蓄積した出力リストを持つ。
    """
    buf = []

    def _fn(text="", end="", flush=True, **kw):
        buf.append(text)

    _fn.buf = buf
    return _fn


# =====================================================
# resolve_venv_python / _resolve_verify_python
# =====================================================
# os.name を偽装すると Path が PosixPath/WindowsPath を誤生成して落ちるため、
# プラットフォームごとに実環境でテストする（実装は os.name に基づき正しく分岐）。

@pytest.mark.skipif(os.name != "nt", reason="Windows path layout")
def test_resolve_venv_python_windows(tmp_path):
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("fake")
    target = tmp_path / "proj" / "main.py"
    target.parent.mkdir(parents=True)
    target.write_text("x")

    result = resolve_venv_python(str(target))
    assert result is not None
    assert result.endswith(os.path.join(".venv", "Scripts", "python.exe"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX path layout")
def test_resolve_venv_python_posix(tmp_path):
    bindir = tmp_path / ".venv" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("fake")
    target = tmp_path / "main.py"
    target.write_text("x")

    result = resolve_venv_python(str(target))
    assert result is not None
    assert result.endswith(os.path.join(".venv", "bin", "python"))


def test_resolve_venv_python_fallback_none(tmp_path):
    # .venv / venv が無い → None
    target = tmp_path / "main.py"
    target.write_text("x")
    assert resolve_venv_python(str(target)) is None


def test_resolve_verify_python_falls_back_to_sys_executable(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("x")
    # .venv 無し環境では sys.executable にフォールバック
    assert _resolve_verify_python(str(target)) == sys.executable


# =====================================================
# _run_py_compile（実際のインタープリタで決定的）
# =====================================================

def test_run_py_compile_syntax_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n    pass\n", encoding="utf-8")
    err = _run_py_compile(str(f), sys.executable)
    assert err != ""
    assert "py_compile" in err


def test_run_py_compile_clean(tmp_path):
    f = tmp_path / "good.py"
    f.write_text("def f():\n    return 1\n", encoding="utf-8")
    assert _run_py_compile(str(f), sys.executable) == ""


def test_run_py_compile_non_py_skipped(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hi", encoding="utf-8")
    assert _run_py_compile(str(f), sys.executable) == ""


# =====================================================
# _run_import_check（AST + find_spec・依存欠落検出）
# =====================================================

def test_run_import_check_missing(tmp_path):
    f = tmp_path / "needs.py"
    f.write_text("import __definitely_not_installed_xyz__\n", encoding="utf-8")
    err = _run_import_check(str(f), sys.executable)
    assert err != ""
    assert "import check" in err
    assert "__definitely_not_installed_xyz__" in err


def test_run_import_check_clean(tmp_path):
    f = tmp_path / "stdlib.py"
    f.write_text("import os\nimport sys\nfrom pathlib import Path\n", encoding="utf-8")
    assert _run_import_check(str(f), sys.executable) == ""


def test_run_import_check_non_py_skipped(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello", encoding="utf-8")
    assert _run_import_check(str(f), sys.executable) == ""


# =====================================================
# _run_execution_verification（段階ゲート・short-circuit）
# =====================================================

def test_execution_verification_short_circuits_at_compile(tmp_path, monkeypatch):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n    pass\n", encoding="utf-8")
    called = {"ruff": False}

    def _fake_ruff(path, python_exe=None):
        called["ruff"] = True
        return ""

    monkeypatch.setattr("subagent._run_ruff_check", _fake_ruff)
    err = _run_execution_verification(str(f), sys.executable)
    assert err != ""            # py_compile で失敗
    assert not called["ruff"]   # ruff ゲートは呼ばれない（short-circuit）


def test_execution_verification_passes_clean(tmp_path):
    f = tmp_path / "good.py"
    f.write_text("ANSWER = 42\n", encoding="utf-8")
    # ruff 未導入でも導入済でも（違反なければ）"" になるはず
    assert _run_execution_verification(str(f), sys.executable) == ""


def test_execution_verification_non_py_skipped(tmp_path):
    f = tmp_path / "data.json"
    f.write_text("{}", encoding="utf-8")
    assert _run_execution_verification(str(f), sys.executable) == ""


# =====================================================
# _generate_fix_edit（JSON 生成・パース）
# =====================================================

def test_generate_fix_edit_parses_json():
    good = ('{"tool": "search_and_replace", "args": '
            '{"path": "x.py", "search_block": "a", "replace_block": "b"}}')
    fix = _generate_fix_edit(_FixMockLLM(good), "x.py", "a", "boom", "goal")
    assert fix == {
        "tool": "search_and_replace",
        "args": {"path": "x.py", "search_block": "a", "replace_block": "b"},
    }


def test_generate_fix_edit_handles_bad_json():
    assert _generate_fix_edit(_FixMockLLM("これは JSON ではない"), "x.py", "a", "err", "g") is None


def test_generate_fix_edit_extracts_json_from_prose():
    mixed = ('考えます。\n{"tool": "write_file", "args": '
             '{"path": "y.py", "content": "z"}}\n以上です。')
    fix = _generate_fix_edit(_FixMockLLM(mixed), "y.py", "old", "err", "")
    assert fix is not None
    assert fix["tool"] == "write_file"
    assert fix["args"]["content"] == "z"


def test_generate_fix_edit_rejects_unknown_tool():
    bad = '{"tool": "delete_file", "args": {"path": "x.py"}}'
    assert _generate_fix_edit(_FixMockLLM(bad), "x.py", "a", "err", "g") is None


def test_generate_fix_edit_fills_missing_path():
    no_path = '{"tool": "search_and_replace", "args": {"search_block": "a", "replace_block": "b"}}'
    fix = _generate_fix_edit(_FixMockLLM(no_path), "filled.py", "a", "err", "g")
    assert fix is not None
    assert fix["args"]["path"] == "filled.py"


def test_generate_fix_edit_swallows_llm_exception():
    # LLM 呼出が例外 → None（ループ側で安全スキップ）
    assert _generate_fix_edit(_ExplodingLLM(), "x.py", "a", "err", "g") is None


# =====================================================
# run_verify_fix_loop（ループ制御・例外安全）
# =====================================================

def test_verify_fix_loop_converges(tmp_path, monkeypatch):
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    # 検証: 1回目失敗 → 2回目成功
    results = iter(["[py_compile]\nSyntaxError", ""])
    monkeypatch.setattr("subagent._run_execution_verification",
                        lambda path, exe: next(results))
    # 修正生成: 有効な search_and_replace
    llm = _FixMockLLM('{"tool": "search_and_replace", "args": '
                      '{"path": "mod.py", "search_block": "x", "replace_block": "y"}}')
    applied = []
    monkeypatch.setattr("tools.execute_builtin_tool",
                        lambda name, args: applied.append((name, args)) or "Success: applied")

    ofn = _outfn()
    summary = run_verify_fix_loop(_Ctx(llm), str(f), "search_and_replace", {}, "goal", ofn)
    assert "成功" in summary
    assert applied  # 修正が1回適用された


def test_verify_fix_loop_respects_max_rounds(tmp_path, monkeypatch):
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    # 常に失敗
    monkeypatch.setattr("subagent._run_execution_verification",
                        lambda path, exe: "[py_compile]\nerr")
    llm = _FixMockLLM('{"tool": "search_and_replace", "args": '
                      '{"path": "mod.py", "search_block": "x", "replace_block": "y"}}')
    monkeypatch.setattr("tools.execute_builtin_tool", lambda name, args: "Success")

    ofn = _outfn()
    summary = run_verify_fix_loop(_Ctx(llm), str(f), "search_and_replace", {}, "goal", ofn)
    assert "未解決" in summary
    assert f"({VERIFY_MAX_ROUNDS}ラウンド)" in summary


def test_verify_fix_loop_breaks_when_fix_generation_fails(tmp_path, monkeypatch):
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("subagent._run_execution_verification",
                        lambda path, exe: "[py_compile]\nerr")
    # 修正生成が例外 → None → break
    monkeypatch.setattr("tools.execute_builtin_tool", lambda name, args: "Success")

    ofn = _outfn()
    summary = run_verify_fix_loop(_Ctx(_ExplodingLLM()), str(f),
                                  "search_and_replace", {}, "goal", ofn)
    assert "未解決" in summary


def test_verify_fix_loop_outer_exception_safety(tmp_path, monkeypatch):
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")

    def _boom(path, exe):
        raise RuntimeError("verify exploded")

    monkeypatch.setattr("subagent._run_execution_verification", _boom)

    ofn = _outfn()
    summary = run_verify_fix_loop(_Ctx(_FixMockLLM("{}")), str(f),
                                  "search_and_replace", {}, "goal", ofn)
    # 外側 try/except で捕捉 → 未解決サマリ（編集結果は壊さない）
    assert "未解決" in summary
    assert any("例外" in s for s in ofn.buf)


def test_verify_fix_loop_skips_non_py():
    ofn = _outfn()
    # .py 以外は "" を返し、何もしない（検証対象外）
    summary = run_verify_fix_loop(_Ctx(None), "notes.md", "write_file", {}, "goal", ofn)
    assert summary == ""


# =====================================================
# execute_tool フック（/verify モード連動）
# =====================================================

def test_execute_tool_invokes_verify_when_mode_on(tmp_path, monkeypatch):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("tools.execute_builtin_tool",
                        lambda name, args: "Success: written")
    monkeypatch.setattr("engine._run_ruff_check", lambda path, python_exe=None: "")
    spy = {"path": None}

    def _fake_verify(ctx, path, tn, ta, goal, ofn):
        spy["path"] = path
        return "[検証結果]\n実行検証: 成功"

    monkeypatch.setattr("engine.run_verify_fix_loop", _fake_verify)

    ctx = _FullCtx(verify_mode=True)
    ofn = _outfn()
    result = execute_tool(ctx, "search_and_replace",
                          {"path": str(f), "search_block": "x", "replace_block": "y"},
                          ofn)
    assert spy["path"] == str(f)
    assert "検証結果" in result


def test_execute_tool_skips_verify_when_mode_off(tmp_path, monkeypatch):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("tools.execute_builtin_tool", lambda name, args: "Success")
    monkeypatch.setattr("engine._run_ruff_check", lambda path, python_exe=None: "")
    spy = {"called": False}

    def _spy_verify(*a, **kw):
        spy["called"] = True
        return ""

    monkeypatch.setattr("engine.run_verify_fix_loop", _spy_verify)

    ctx = _FullCtx(verify_mode=False)
    ofn = _outfn()
    execute_tool(ctx, "search_and_replace",
                 {"path": str(f), "search_block": "x", "replace_block": "y"},
                 ofn)
    assert spy["called"] is False


# =====================================================
# run_fast_gate_check（VERIFY_FAST_GATE_ALWAYS: 常時高速ゲート・LLM不使用）
# =====================================================
# 「生成は1回、検証は安価な決定的チェックで」の実装。py_compile + import解決 + ruff の
# 検出のみを行う（LLMを使う自動修正ループとは別物）。/verify トグルの状態に関わらず、
# engine.execute_tool から破壊的編集の直後に毎回呼ばれる。

def test_run_fast_gate_check_detects_syntax_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n    pass\n", encoding="utf-8")
    out = run_fast_gate_check(str(f))
    assert "[py_compile]" in out


def test_run_fast_gate_check_detects_missing_import(tmp_path):
    f = tmp_path / "needs.py"
    f.write_text("import __definitely_not_installed_xyz__\n", encoding="utf-8")
    out = run_fast_gate_check(str(f))
    assert "[import check]" in out


def test_run_fast_gate_check_clean(tmp_path):
    f = tmp_path / "good.py"
    f.write_text("ANSWER = 42\n", encoding="utf-8")
    assert run_fast_gate_check(str(f)) == ""


def test_run_fast_gate_check_non_py_skipped(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hi", encoding="utf-8")
    assert run_fast_gate_check(str(f)) == ""


def test_run_fast_gate_check_missing_file_returns_empty():
    assert run_fast_gate_check("Z:/definitely/not/here.py") == ""


# =====================================================
# execute_tool フック（VERIFY_FAST_GATE_ALWAYS: /verify トグル非依存の常時ゲート）
# =====================================================
# 決定的チェック（py_compile/import解決/ruff）は toggle 不要でデフォルト実行する一方、
# LLM を使う自動再編集（run_verify_fix_loop の修正ループ）は従来通り verify_mode
# （/verify トグル）が有効な時だけ動く、という境界を検証する。

def test_execute_tool_fast_gate_runs_even_when_verify_mode_off(tmp_path, monkeypatch):
    """verify_mode オフでも、構文エラーを含む編集の直後に高速ゲートが検出結果を返す。

    LLM を使う自動修正ループ (run_verify_fix_loop) は verify_mode オフのままなので
    呼ばれてはならない（決定的チェックは常時・LLM修正は opt-in、という境界の検証）。
    """
    f = tmp_path / "m.py"
    bad_code = "def f(:\n    pass\n"
    f.write_text(bad_code, encoding="utf-8")

    monkeypatch.setattr("tools.execute_builtin_tool",
                        lambda name, args: "Success: written")
    spy = {"called": False}

    def _spy_verify(*a, **kw):
        spy["called"] = True
        return "[検証結果]\nLLM修正ループは呼ばれてはいけない"

    monkeypatch.setattr("engine.run_verify_fix_loop", _spy_verify)

    ctx = _FullCtx(verify_mode=False)
    ofn = _outfn()
    result = execute_tool(ctx, "write_file", {"path": str(f), "content": bad_code}, ofn)

    assert "[py_compile]" in result  # 高速ゲートは常時実行され構文エラーを検出する
    assert spy["called"] is False    # LLMベースの自動修正ループは verify_mode オフでは動かない


def test_execute_tool_fast_gate_disabled_falls_back_to_ruff_only(tmp_path, monkeypatch):
    """VERIFY_FAST_GATE_ALWAYS=False なら旧来通り ruff のみが常時実行される。"""
    f = tmp_path / "m.py"
    bad_code = "def f(:\n    pass\n"
    f.write_text(bad_code, encoding="utf-8")

    monkeypatch.setattr("tools.execute_builtin_tool",
                        lambda name, args: "Success: written")
    monkeypatch.setattr("engine.VERIFY_FAST_GATE_ALWAYS", False)
    monkeypatch.setattr("engine._run_ruff_check",
                        lambda path, python_exe=None: "[ruff check --select E,F]\nfake\n")

    ctx = _FullCtx(verify_mode=False)
    ofn = _outfn()
    result = execute_tool(ctx, "write_file", {"path": str(f), "content": bad_code}, ofn)

    assert "[py_compile]" not in result  # 高速ゲート無効時は py_compile を実行しない
    assert "[ruff check" in result       # 旧来通り ruff のみは実行される
