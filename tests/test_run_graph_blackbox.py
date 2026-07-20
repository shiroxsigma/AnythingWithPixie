"""run_graph の制御フロー遷移をブラックボックステスト（scripted mock LLM で駆動）。

test_delegate_research.py の _MockLLM / _tc パターンを finish_reason 指定可能に
拡張して再利用。LM Studio 互換（create_chat_completion は常に generator を返す）。

カバーする終了遷移:
  - final_answer（ツール実行後 / no-tool 直接回答）
  - max_tool_calls_reached（node_observe で DONE）
  - continuation → final_answer（finish_reason="length" で NEEDS_CONTINUATION）
  - continuation_limit（継続8回到達）
  - empty_response（空応答）
  - user_rejected（interactive_fn が全ツール却下）

※ iteration_limit（全体反復上限）と loop_force_exit の純粋遷移は、通常パスでは他の
   終了条件が先に発火するため到達困難。状態注入や条件設計で独立再現でき次第追加する
   （TODO: 残課題）。run_graph の主要出口は本ファイルで保護される。
"""

import copy
import json
import types

from config import EMPTY_RESPONSE_MAX_RETRY
from engine import run_graph
from state import AgentState

# =====================================================
# ヘルパ: scripted mock LLM（finish_reason 指定可能拡張版）
# =====================================================

class _MockLLM:
    """create_chat_completion の戻り値をスクリプト化したモック。

    scripts: list of
      - (content, tool_calls_or_None)               # 従来形式（finish_reason 自動）
      - (content, tool_calls_or_None, finish_reason) # length 等を明示指定
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.captured = []  # 呼出ごとの messages スナップショット
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
    """_accumulate_tool_calls が期待する delta.tool_calls 要素1件を生成。"""
    return [{
        "index": idx,
        "id": f"call_{idx}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }]


def _make_ctx(llm, **overrides):
    """run_graph / node_plan が getattr で参照する最小 context。"""
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
    """output_fn をバッファ蓄積版で返す（検証には使わないが、print 副作用を吸収）。"""
    buf = []

    def _fn(text="", end="", flush=True, **kw):
        buf.append(text)
    return _fn, buf


def _fresh_state(user_text="現在の状況について詳しく教えてください"):
    state = AgentState()
    state.chat_history.add("user", user_text)
    return state


def _run(ctx, state, *, out_fn=None, interactive_fn=None):
    """run_graph を system_msg_builder 付きで呼ぶラッパー。

    node_plan は budget_hint 生成時（コンテキスト使用率のしきい値超過）に system_text
    を参照するため、system_msg_builder なしでは UnboundLocalError になる（本番は常に
    build_system_text が渡されるため発現しない）。テストでは最小ビルダーを注入する。
    """
    def _sys_builder(context, state_board, **kw):
        return "You are a helpful test assistant."
    return run_graph(
        ctx, state,
        output_fn=out_fn,
        system_msg_builder=_sys_builder,
        interactive_fn=interactive_fn,
    )


# =====================================================
# 遷移 C: ツール実行 → 最終回答
# =====================================================

def test_transition_c_tool_then_final(monkeypatch):
    """ツール1回実行後、最終回答で終了する標準パス。"""
    calls = []

    def _fake_execute(context, tool_name, tool_args, output_fn):
        calls.append((tool_name, tool_args))
        return "[get_cwd の実行結果: /home/user]"

    monkeypatch.setattr("engine.execute_tool", _fake_execute)

    llm = _MockLLM([
        ("現在のディレクトリを確認します。", _tc("get_cwd")),
        ("現在のディレクトリは /home/user でした。これで質問への回答は完了し、十分な情報を提供できました。", None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("現在のディレクトリを教えて")
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert calls == [("get_cwd", {})]
    assert state.tool_call_count == 1
    assert "final_answer" in state.exit_reason


def test_max_tool_calls_reached(monkeypatch):
    """max_tool_calls 到達で node_observe が DONE を返し終了する。"""
    monkeypatch.setattr("engine.execute_tool",
                        lambda ctx, name, args, out: "[結果]")

    llm = _MockLLM([
        ("ツールを使用します。", _tc("get_cwd")),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("ディレクトリは？")
    state.max_tool_calls = 1
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert "max_tool_calls_reached" in state.exit_reason


# =====================================================
# 遷移 E: LLMバックエンド接続/APIエラー（__llm_error__ マーカー）
# =====================================================

class _ErrorLLM:
    """llm_client の接続/APIエラー時の yield を再現するモック。

    エラー文字列を content として流しつつ、チャンク top-level に __llm_error__
    マーカーを付ける（llm_client.py の URLError/HTTPError 分岐と同形）。
    """

    def __init__(self, error_detail="接続エラー: <urlopen error [WinError 10061]>"):
        self.error_detail = error_detail
        self.n_ctx = 32768

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream,
                               tools=None, tool_choice=None, **kw):
        def _gen():
            yield {
                "choices": [{"delta": {"content": f"\n(API Error: {self.error_detail})"}}],
                "__llm_error__": self.error_detail,
            }
        return _gen()

    def estimate_token_count(self, text):
        return len(text) // 3


def test_llm_connection_error_not_final_answer(monkeypatch):
    """接続断が final_answer に偽装されず、llm_connection_error で終了する（回帰）。"""
    llm = _ErrorLLM()
    ctx = _make_ctx(llm)
    state = _fresh_state("test")
    out_fn, _ = _output_buf()

    result = _run(ctx, state, out_fn=out_fn)

    # 異常系 exit_reason になっており、final_answer に偽装されていない
    assert "llm_connection_error" in state.exit_reason
    assert "final_answer" not in state.exit_reason
    # ツールは1回も実行されていない
    assert state.tool_call_count == 0
    # 返却メッセージはクリーンな案内文（生のエラー文字列そのままではない）
    assert "接続できませんでした" in result
    # 履歴はエラー文字列で汚染されていない（assistant メッセージが追加されていない）
    assert not any(m.get("role") == "assistant" for m in state.chat_history.messages)


# =====================================================
# 遷移 A: 継続（finish_reason="length"）→ 最終回答 / 上限
# =====================================================

def test_continuation_then_final(monkeypatch):
    """length で途切れた後、続きを生成して最終回答で終了する。"""
    llm = _MockLLM([
        ("これは非常に長い説明の始まりであり、出力が最大文字数に達して途中で切れてしまった内容です。", None, "length"),
        ("続きの文章を生成し、最終的に十分な長さのまとまった回答として完結しました。これで完了です。", None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("詳しく説明してください")
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert state.continuation_count >= 1
    assert "final_answer" in state.exit_reason


def test_continuation_limit(monkeypatch):
    """継続8回到達で continuation_limit 強制終了する。"""
    llm = _MockLLM([
        ("長い出力が途切れました。", None, "length"),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("説明してください")
    state.continuation_count = 8  # 上限到達状態を注入
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert "continuation_limit" in state.exit_reason


# =====================================================
# 遷移 D: ツールなし最終回答 / 空応答 / ユーザー却下
# =====================================================

def test_no_tool_final_answer(monkeypatch):
    """ツールを一度も呼ばず、直接の最終回答で終了する。"""
    llm = _MockLLM([
        ("これは最終回答です。ユーザーの質問に対して十分な情報を含んでおり、明確に完結しています。以上で完了します。", None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("質問に答えて")
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert "final_answer" in state.exit_reason
    assert state.tool_call_count == 0


def test_empty_response_break(monkeypatch):
    """空応答を再試行上限まで繰り返し、empty_response 終了する。"""
    n_empty = EMPTY_RESPONSE_MAX_RETRY + 1  # 初回 + 再試行分
    llm = _MockLLM([("", None)] * n_empty)
    ctx = _make_ctx(llm)
    state = _fresh_state("質問")
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert "empty_response" in state.exit_reason


def test_empty_response_retry_then_recover(monkeypatch):
    """空応答後に再試行し、ツール実行で回復する（max_tool_calls=1 で確実終了）。"""
    calls = []

    def _fake_execute(context, tool_name, tool_args, output_fn):
        calls.append((tool_name, tool_args))
        return "[get_cwd の実行結果: /home/user]"

    monkeypatch.setattr("engine.execute_tool", _fake_execute)

    llm = _MockLLM([
        ("", None),  # 1回目: 空応答 → 再試行
        ("ディレクトリを確認します。", _tc("get_cwd")),  # 2回目: 再試行でツール実行
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("ディレクトリは？")
    state.max_tool_calls = 1  # ツール1回で max_tool_calls_reached で確実終了
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert calls == [("get_cwd", {})]
    assert state.tool_call_count == 1
    assert "max_tool_calls_reached" in state.exit_reason


def test_empty_response_retry_then_fallback(monkeypatch):
    """空応答再試行上限到達後、last_substantive_content があればフォールバックする。"""
    monkeypatch.setattr("engine.execute_tool",
                        lambda ctx, name, args, out: "[結果]")
    # 先に有意 content(>=50字) で last_substantive_content を蓄積 → ツール実行 →
    # その後、再試行上限+1回の空応答で fallback_response に至る
    llm = _MockLLM([
        ("これは十分な長さのある有意な回答テキストです。" * 3, _tc("get_cwd")),
    ] + [("", None)] * (EMPTY_RESPONSE_MAX_RETRY + 1))
    ctx = _make_ctx(llm)
    state = _fresh_state("詳細を教えて")
    out_fn, _ = _output_buf()

    result = _run(ctx, state, out_fn=out_fn)

    assert "fallback_response" in state.exit_reason
    assert result


def test_user_rejected(monkeypatch):
    """interactive_fn が全ツールを却下した場合、user_rejected で終了する。"""
    llm = _MockLLM([
        ("ツールを使おうと思います。", _tc("get_cwd")),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state("ディレクトリは？")
    out_fn, _ = _output_buf()

    def _reject_all(tool_calls, content):
        return [], None  # 全ツール却下

    _run(ctx, state, out_fn=out_fn, interactive_fn=_reject_all)

    assert "user_rejected" in state.exit_reason
