"""delegate_research サブエージェントのテスト（pure・モデル不要）。

run_agent_subquery の軽量 ReAct ループを、スクリプト化モック LLM で検証する。
モックは LM Studio 互換（create_chat_completion は常に generator を返す）。
"""

import copy
import json

import pytest

from config import (
    CODE_TOOL_SET,
    DELEGATE_SUBAGENT_TOOLS,
    DESTRUCTIVE_TOOLS,
    READONLY_TOOLS,
)
from state import AgentState
from subagent import run_agent_subquery
from tools import TOOL_REGISTRY, score_tools

# =====================================================
# モック LLM（LM Studio 互換: 常に generator）
# =====================================================

class _MockLLM:
    """create_chat_completion の戻り値をスクリプト化したモック。

    scripts: list of (content, tool_calls_or_None)。
    tool_calls は _accumulate_tool_calls が期待する delta 形式のリスト。
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.captured = []  # 呼出ごとの messages スナップショット
        self.n_ctx = 32768

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream,
                               tools=None, tool_choice=None, **kw):
        self.captured.append(copy.deepcopy(messages))
        content, tool_calls = self.scripts.pop(0)

        def _gen():
            yield {
                "choices": [{
                    "delta": {"content": content, "tool_calls": tool_calls},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
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


@pytest.fixture
def fake_execute(monkeypatch):
    """execute_builtin_tool を記録付きの fake に差し替え（実際の副作用を避ける）。"""
    calls = []

    def _fake(name, args):
        calls.append((name, args))
        return f"[{name} の実行結果]"

    monkeypatch.setattr("tools.execute_builtin_tool", _fake)
    return calls


# =====================================================
# run_agent_subquery コア動作
# =====================================================

def test_subagent_returns_conclusion(fake_execute):
    # step1: get_cwd を呼ぶ → step2: 結論
    mock = _MockLLM([
        ("調べます", _tc("get_cwd")),
        ("結論: 現在地は分かりました", None),
    ])
    result = run_agent_subquery(mock, question="現在地を教えて")
    assert result == "結論: 現在地は分かりました"
    assert len(mock.captured) == 2  # LLM 呼び出し2回
    assert fake_execute == [("get_cwd", {})]


def test_subagent_does_not_touch_chat_history(fake_execute):
    # run_agent_subquery は state を持たず、外部の AgentState を汚さない
    state = AgentState()
    before = list(state.chat_history.messages)
    mock = _MockLLM([("結論のみ", None)])
    run_agent_subquery(mock, question="x")
    assert state.chat_history.messages == before


def test_only_delegate_tools_invoked(fake_execute):
    # 書き込み系ツールは拒否され、execute_builtin_tool に渡らない
    mock = _MockLLM([
        ("", _tc("write_file", {"path": "evil.txt", "content": "x"})),
        ("結論", None),
    ])
    result = run_agent_subquery(mock, question="x")
    assert result == "結論"
    assert all(name != "write_file" for name, _ in fake_execute)
    # step2 の messages に拒否メッセージが記録されている
    step2_msgs = mock.captured[1]
    rejected = [
        m for m in step2_msgs
        if "読み取り専用ツールのみ使用可能" in m.get("content", "")
    ]
    assert rejected, "拒否メッセージが記録されていない"


def test_step_limit_enforced(fake_execute):
    # 常に tool_call を返す → max_steps 到達 → 強制サマライズ
    mock = _MockLLM([
        ("s1", _tc("get_cwd")),
        ("s2", _tc("get_cwd")),
        ("s3", _tc("get_cwd")),
        ("強制サマライズ結論", None),
    ])
    result = run_agent_subquery(mock, question="x", max_steps=3)
    assert result == "強制サマライズ結論"
    assert len(mock.captured) == 4  # 3ステップ + 強制サマライズ1回


def test_native_tool_call_text_fallback(fake_execute):
    # 構造化 tool_calls なし・<tool_call> テキスト → _parse_native_tool_calls で抽出
    content = '<tool_call\n{"name":"get_cwd","arguments":{}}\n</tool_call>'
    mock = _MockLLM([
        (content, None),
        ("結論", None),
    ])
    result = run_agent_subquery(mock, question="x")
    assert result == "結論"
    assert ("get_cwd", {}) in fake_execute  # ネイティブパース経路が動いた


def test_thinking_stripped(fake_execute):
    mock = _MockLLM([("<think>長い推論...</think>\n実テキスト", None)])
    result = run_agent_subquery(mock, question="x")
    assert result == "実テキスト"
    assert "<think>" not in result


def test_supports_tool_role_false(fake_execute):
    mock = _MockLLM([
        ("調べます", _tc("get_cwd")),
        ("結論", None),
    ])
    run_agent_subquery(mock, question="x", supports_tool_role=False)
    step2_msgs = mock.captured[1]
    # tool result が role="user" + [ツール結果] に変換されている
    converted = [
        m for m in step2_msgs
        if m["role"] == "user" and "[ツール結果]" in m.get("content", "")
    ]
    assert converted
    assert not any(m["role"] == "tool" for m in step2_msgs)


def test_supports_tool_role_true(fake_execute):
    mock = _MockLLM([
        ("調べます", _tc("get_cwd")),
        ("結論", None),
    ])
    run_agent_subquery(mock, question="x", supports_tool_role=True)
    step2_msgs = mock.captured[1]
    assert any(m["role"] == "tool" for m in step2_msgs)


def test_budget_timeout(monkeypatch, fake_execute):
    # time.monotonic を呼出ごとに大きく増加させ、予算超過 → 強制サマライズへ
    counter = [0]

    def fake_monotonic():
        counter[0] += 1
        return float(counter[0] * 1_000_000)

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    mock = _MockLLM([("予算超過フォールバック結論", None)])
    result = run_agent_subquery(mock, question="x")
    assert result == "予算超過フォールバック結論"
    assert len(mock.captured) == 1  # ループ即 break → 強制サマライズ1回のみ


def test_file_hints_and_focus_injected(fake_execute):
    mock = _MockLLM([("結論", None)])
    run_agent_subquery(
        mock,
        question="本体",
        file_hints=["src/a.py", "src/b.py"],
        focus="シグネチャ変更の影響",
    )
    first_user = next(m for m in mock.captured[0] if m["role"] == "user")
    assert "本体" in first_user["content"]
    assert "src/a.py" in first_user["content"]
    assert "シグネチャ変更の影響" in first_user["content"]


# =====================================================
# 登録・分類（pure）
# =====================================================

def test_registered_in_tool_registry():
    assert "delegate_research" in TOOL_REGISTRY


def test_score_tools_surfaces_delegate():
    # ツール名直接マッチで確実にスコア上位に浮上
    assert "delegate_research" in score_tools("delegate_research で調査して")


def test_classification():
    assert "delegate_research" in READONLY_TOOLS
    assert "delegate_research" in CODE_TOOL_SET
    assert "delegate_research" not in DESTRUCTIVE_TOOLS


def test_subagent_tools_excludes_analyze_file():
    # ネストLLM回避: analyze_file はサブエージェントの許可セットに含まれない
    assert "analyze_file" not in DELEGATE_SUBAGENT_TOOLS
    assert "read_file" in DELEGATE_SUBAGENT_TOOLS


# =====================================================
# サーバーラウンドロビン選択（並列サブ×2サーバー分散）
# =====================================================

def test_execute_delegate_round_robin(monkeypatch):
    # delegate_llm 設定時、メイン/サブが交互に選ばれる（スレッドセーフなラウンドロビン）
    import subagent

    seen = []

    def spy(llm, **kw):
        seen.append(llm)
        return "結論"

    monkeypatch.setattr(subagent, "run_agent_subquery", spy)

    class _Ctx:
        llm = "MAIN"
        delegate_llm = "SUB"
        supports_tool_role = False

    def out(*a, **k):
        pass

    subagent._execute_delegate_research(_Ctx(), {"question": "x"}, out)
    subagent._execute_delegate_research(_Ctx(), {"question": "x"}, out)
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert set(seen) == {"MAIN", "SUB"}


def test_execute_delegate_no_sub_uses_main(monkeypatch):
    # delegate_llm=None のとき常にメインを使用
    import subagent

    seen = []

    def spy(llm, **kw):
        seen.append(llm)
        return "結論"

    monkeypatch.setattr(subagent, "run_agent_subquery", spy)

    class _Ctx:
        llm = "MAIN"
        delegate_llm = None
        supports_tool_role = False

    def out(*a, **k):
        pass

    subagent._execute_delegate_research(_Ctx(), {"question": "x"}, out)
    assert seen == ["MAIN"]


def test_execute_analyze_round_robin(monkeypatch, tmp_path):
    """analyze_file も delegate_llm 設定時にメイン/サブを交互に使用（並列分散）。"""
    import subagent

    seen = []

    def spy(llm, file_path, file_content, prompt=None):
        seen.append(llm)
        return "要約"

    monkeypatch.setattr(subagent, "run_text_subquery", spy)
    # キャッシュ I/O と state_board 副作用を回避
    monkeypatch.setattr(subagent, "_lookup_analysis_cache", lambda *a, **k: None)
    monkeypatch.setattr(subagent, "_save_analysis_cache", lambda *a, **k: None)

    src = tmp_path / "sample.py"
    src.write_text("def f():\n    pass\n", encoding="utf-8")

    class _Ctx:
        llm = "MAIN"
        delegate_llm = "SUB"
        use_vision = False

    def out(*a, **k):
        pass

    subagent._execute_analyze_file(_Ctx(), {"path": str(src)}, out)
    subagent._execute_analyze_file(_Ctx(), {"path": str(src)}, out)
    assert len(seen) == 2
    assert set(seen) == {"MAIN", "SUB"}
