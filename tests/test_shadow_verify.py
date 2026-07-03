"""分岐点限定 lazy best-of-2 のテスト（shadow_verify.py + run_graph 連携）。

- shadow_apply: 各編集ツールの適用後コンテンツが実適用と一致し、実ファイルは無変更であること。
- shadow_gate: 構文エラーの検出、クリーンな編集の通過、非 .py の対象外を検証する。
- run_graph 統合: scripted mock LLM（test_run_graph_blackbox.py と同じパターン）で
  「1本目クリーン→再サンプルなし」「1本目が壊れた編集→再サンプルが呼ばれ2本目採用」
  「final answer のギリギリスコア→best-of-2」「明確な高スコア→再サンプルなし」を検証する。
"""

import copy
import json
import types

import tools
from engine import run_graph
from shadow_verify import shadow_apply, shadow_gate
from state import AgentState

# =====================================================
# shadow_apply
# =====================================================


def test_shadow_apply_write_file_returns_content_without_writing(tmp_path):
    f = tmp_path / "new.py"
    content, reason = shadow_apply("write_file", {"path": str(f), "content": "x = 1\n"})
    assert reason == ""
    assert content == "x = 1\n"
    assert not f.exists()  # shadow_apply は実ファイルに一切触れない


def test_shadow_apply_append_to_file_matches_real_apply(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("line1\n", encoding="utf-8")
    args = {"path": str(f), "content": "line2\n"}

    shadow_content, reason = shadow_apply("append_to_file", args)
    assert reason == ""
    assert f.read_text(encoding="utf-8") == "line1\n"  # 無変更のまま

    tools.append_to_file(**args)
    assert shadow_content == f.read_text(encoding="utf-8")  # 実適用結果と一致


def test_shadow_apply_replace_lines_matches_real_apply(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    args = {"path": str(f), "start_line": 2, "end_line": 2, "new_content": "b = 20\n"}

    shadow_content, reason = shadow_apply("replace_lines", args)
    assert reason == ""
    assert f.read_text(encoding="utf-8") == "a = 1\nb = 2\nc = 3\n"  # 無変更のまま

    tools.replace_lines(**args)
    assert shadow_content == f.read_text(encoding="utf-8")


def test_shadow_apply_replace_lines_invalid_range_returns_reason(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("a = 1\n", encoding="utf-8")
    content, reason = shadow_apply(
        "replace_lines", {"path": str(f), "start_line": 5, "end_line": 6, "new_content": "x"}
    )
    assert content is None
    assert "Error" in reason


def test_shadow_apply_search_and_replace_matches_real_apply(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    args = {"path": str(f), "search_block": "return 1", "replace_block": "return 2"}

    shadow_content, reason = shadow_apply("search_and_replace", args)
    assert reason == ""
    assert f.read_text(encoding="utf-8") == "def foo():\n    return 1\n"  # 無変更のまま

    tools.search_and_replace(**args)
    assert shadow_content == f.read_text(encoding="utf-8")


def test_shadow_apply_search_and_replace_not_found_returns_reason(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    content, reason = shadow_apply(
        "search_and_replace",
        {"path": str(f), "search_block": "totally nonexistent content xyz", "replace_block": "y"},
    )
    assert content is None
    assert reason


def test_shadow_apply_unknown_tool_returns_none():
    content, reason = shadow_apply("delete_file", {"path": "x"})
    assert content is None
    assert reason


# =====================================================
# shadow_gate
# =====================================================


def test_shadow_gate_detects_syntax_error(tmp_path):
    f = tmp_path / "broken.py"
    result = shadow_gate("write_file", {"path": str(f), "content": "def foo(:\n    pass\n"})
    assert result != ""
    assert "py_compile" in result


def test_shadow_gate_detects_ruff_violation(tmp_path):
    f = tmp_path / "undefined.py"
    # 構文は正しいが未定義名参照（pyflakes F821）
    result = shadow_gate("write_file", {"path": str(f), "content": "print(undefined_variable_xyz)\n"})
    assert result != ""
    assert "ruff" in result


def test_shadow_gate_clean_edit_passes(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    result = shadow_gate("write_file", {"path": str(f), "content": "def foo():\n    return 2\n"})
    assert result == ""


def test_shadow_gate_non_py_skipped(tmp_path):
    f = tmp_path / "notes.txt"
    result = shadow_gate("write_file", {"path": str(f), "content": "this is not python ((( broken"})
    assert result == ""


def test_shadow_gate_shadow_apply_unable_skips(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    result = shadow_gate(
        "search_and_replace",
        {"path": str(f), "search_block": "nonexistent xyz content", "replace_block": "y"},
    )
    assert result == ""  # 適用不能はゲート対象外（事後の fast gate / エラーFBに委ねる）


def test_shadow_gate_never_writes_real_file(tmp_path):
    f = tmp_path / "target.py"
    original = "def foo():\n    return 1\n"
    f.write_text(original, encoding="utf-8")
    shadow_gate("write_file", {"path": str(f), "content": "def foo(:\n"})
    assert f.read_text(encoding="utf-8") == original


# =====================================================
# run_graph 統合（scripted mock LLM）
# =====================================================
# test_run_graph_blackbox.py と同じヘルパパターンを自己完結で複製する。


class _MockLLM:
    """create_chat_completion の戻り値をスクリプト化したモック（LM Studio 互換）。"""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.captured = []
        self.n_ctx = 32768

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream,
                               tools=None, tool_choice=None, **kw):
        self.captured.append(copy.deepcopy(messages))
        item = self.scripts.pop(0)
        if len(item) == 3:
            content, tool_calls, finish_reason = item
        else:
            content, tool_calls = item
            finish_reason = "tool_calls" if tool_calls else "stop"

        def _gen():
            yield {
                "choices": [{
                    "delta": {"content": content, "tool_calls": tool_calls},
                    "finish_reason": finish_reason,
                }]
            }
        return _gen()

    def estimate_token_count(self, text):
        return len(text) // 3


def _tc(name, args=None, idx=0):
    return [{
        "index": idx,
        "id": f"call_{idx}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }]


def _make_ctx(llm, **overrides):
    defaults = dict(
        llm=llm,
        code_mode=False,
        force_deep=False,
        supports_tool_role=False,
        debug_mode=False,
        phase="EXECUTING",
        review_mode=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _output_buf():
    buf = []

    def _fn(text="", end="", flush=True, **kw):
        buf.append(text)
    return _fn, buf


def _fresh_state(user_text="バグを直して"):
    state = AgentState()
    state.chat_history.add("user", user_text)
    return state


def _run(ctx, state, *, out_fn=None):
    def _sys_builder(context, state_board, **kw):
        return "You are a helpful test assistant."
    return run_graph(ctx, state, output_fn=out_fn, system_msg_builder=_sys_builder)


# --- 編集のシャドウ検証: 1本目クリーン → 再サンプルなし ---

def test_run_graph_edit_no_resample_when_clean(monkeypatch, tmp_path):
    # final answer の best-of-2 は本テストの対象外なので無効化して干渉を防ぐ
    monkeypatch.setattr("engine.BEST_OF_ANSWER_ENABLED", False)

    target = tmp_path / "clean_out.py"
    calls = []

    def _fake_execute(context, tool_name, tool_args, output_fn):
        calls.append((tool_name, tool_args))
        return f"Success: {tool_args.get('path')} に書き込みました。"

    monkeypatch.setattr("engine.execute_tool", _fake_execute)

    clean_args = {"path": str(target), "content": "def foo():\n    return 1\n"}
    final_text = ("対応案として、ファイルへの書き込みが完了しました。"
                  "以下の通り、関数を正しく実装済みです。out.pyへの変更が反映されています。")

    llm = _MockLLM([
        ("編集します。", _tc("write_file", clean_args)),
        (final_text, None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("out.py に関数を書いて")
    out_fn, buf = _output_buf()

    result = _run(ctx, state, out_fn=out_fn)

    assert len(llm.captured) == 2  # 再サンプルなし（追加コストゼロ）
    assert calls == [("write_file", clean_args)]
    assert not state.failure_signals
    assert not any("編集候補が検証に失敗" in s for s in buf)
    assert "final_answer" in state.exit_reason
    assert result == final_text


# --- 編集のシャドウ検証: 1本目が壊れた編集 → 再サンプルが呼ばれ2本目採用 ---

def test_run_graph_edit_resample_on_dirty_first_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr("engine.BEST_OF_ANSWER_ENABLED", False)

    target = tmp_path / "resample_out.py"
    calls = []

    def _fake_execute(context, tool_name, tool_args, output_fn):
        calls.append((tool_name, tool_args))
        return f"Success: {tool_args.get('path')} に書き込みました。"

    monkeypatch.setattr("engine.execute_tool", _fake_execute)

    dirty_args = {"path": str(target), "content": "def foo(:\n    pass\n"}  # 構文エラー
    clean_args = {"path": str(target), "content": "def foo():\n    return 1\n"}  # クリーン
    final_text = ("対応案として、ファイルへの書き込みが完了しました。"
                  "以下の通り、関数を正しく実装済みです。out.pyへの変更が反映されています。")

    llm = _MockLLM([
        ("編集します。", _tc("write_file", dirty_args)),          # 1本目: 壊れた候補
        ("修正して再生成します。", _tc("write_file", clean_args)),  # 再サンプル: クリーン
        (final_text, None),                                       # 最終回答
        ("", None),                                                # 教訓ストアの reflection 呼出
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("resample_out.py に関数を書いて")
    out_fn, buf = _output_buf()

    result = _run(ctx, state, out_fn=out_fn)

    assert len(llm.captured) == 4  # 1本目 + 再サンプル + 最終回答 + reflection
    assert calls == [("write_file", clean_args)]  # 実行されたのはクリーンな2本目のみ
    assert any("編集候補が検証に失敗したため再生成" in s for s in buf)
    assert any("再生成された編集候補は検証をクリア" in s for s in buf)
    assert state.failure_signals  # 教訓ストア連携: 失敗信号が記録される
    assert "final_answer" in state.exit_reason
    assert result == final_text


# --- final answer の lazy best-of-2: ギリギリスコア → 2本目生成し高スコアを採用 ---

def test_run_graph_final_answer_margin_triggers_resample_and_picks_higher(monkeypatch):
    first_content = ("対応案として、ファイルへの書き込みが完了しました。"
                      "以下の通り、関数を正しく実装済みです。out.pyへの変更が反映されています。")  # score 58
    second_content = (
        "## 結論\n調査の結果、原因と対応案は以下の通りです。\n\n"
        "- ファイル src/example.py の構文エラーを修正しました。\n"
        "- 関数の戻り値を正しい値に変更し、動作確認を行いました。\n"
        "- 該当箇所は行番号12付近です。\n\n"
        "以上で対応は完了しました。"
    )  # score 76（1本目より高い）

    llm = _MockLLM([
        (first_content, None),
        (second_content, None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("バグを直して")
    out_fn, buf = _output_buf()

    result = _run(ctx, state, out_fn=out_fn)

    assert len(llm.captured) == 2  # 1本目 + 再サンプル1回のみ
    assert result == second_content  # 高スコアの2本目を採用
    assert any("もう1候補を生成して比較" in s for s in buf)
    assert any("2本目の回答を採用しました" in s for s in buf)
    assert "final_answer" in state.exit_reason


# --- final answer の lazy best-of-2: 明確に高スコア → 再サンプルなし ---

def test_run_graph_final_answer_high_score_skips_resample(monkeypatch):
    content = (
        "## 結論\n調査の結果、原因と対応案は以下の通りです。\n\n"
        "- ファイル src/example.py の構文エラーを修正しました。\n"
        "- 関数の戻り値を正しい値に変更し、動作確認を行いました。\n"
        "- 該当箇所は行番号12付近です。\n\n"
        "以上で対応は完了しました。"
    )  # score 76 > 閾値(50) + マージン(15)

    llm = _MockLLM([
        (content, None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("バグを直して")
    out_fn, buf = _output_buf()

    result = _run(ctx, state, out_fn=out_fn)

    assert len(llm.captured) == 1  # 再サンプルなし（追加コストゼロ）
    assert result == content
    assert not any("もう1候補を生成して比較" in s for s in buf)
    assert "final_answer" in state.exit_reason
