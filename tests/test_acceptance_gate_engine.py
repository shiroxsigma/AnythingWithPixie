"""run_graph の正常終了候補が共通 acceptance gate を通ることの回帰テスト。"""

import copy
import json
import types

from engine import run_graph
from state import AgentState


class _MockLLM:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.captured = []
        self.n_ctx = 32768

    def create_chat_completion(
        self,
        messages,
        *,
        max_tokens,
        temperature,
        stream,
        tools=None,
        tool_choice=None,
        **kwargs,
    ):
        self.captured.append(copy.deepcopy(messages))
        script = self.scripts.pop(0)
        content, tool_calls = script[:2]
        finish_reason = script[2] if len(script) > 2 else (
            "tool_calls" if tool_calls else "stop"
        )

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


def _tc(name, args=None):
    return [{
        "index": 0,
        "id": "call_0",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }]


def _context(llm):
    return types.SimpleNamespace(
        llm=llm,
        code_mode=False,
        force_deep=False,
        supports_tool_role=False,
        debug_mode=False,
        phase="EXECUTING",
        review_mode=False,
    )


def _state(user_text):
    state = AgentState()
    state.chat_history.add("user", user_text)
    return state


def _run(llm, state):
    def _system_builder(context, state_board, **kwargs):
        return "You are a helpful test assistant."

    output = []
    result = run_graph(
        _context(llm),
        state,
        output_fn=lambda text="", **kwargs: output.append(text),
        system_msg_builder=_system_builder,
    )
    return result, "".join(output)


def _install_acceptance(monkeypatch, validation_results):
    results = list(validation_results)
    monkeypatch.setattr("engine.derive_acceptance", lambda *args: [{"kind": "test"}])
    monkeypatch.setattr(
        "engine.validate_acceptance", lambda *args, **kwargs: results.pop(0)
    )
    monkeypatch.setattr("engine.BEST_OF_ANSWER_ENABLED", False)
    monkeypatch.setattr("engine.LESSONS_ENABLED", False)
    return results


def test_normal_final_warns_and_uses_unresolved_exit_after_two_retries(monkeypatch):
    remaining = _install_acceptance(
        monkeypatch,
        [["CHANGELOG.md に期待値がありません"]] * 3,
    )
    answer = (
        "結論: 更新作業は完了しました。理由と確認結果を含む十分な報告です。"
        "対象の状態も確認済みであり、以上が最終的なまとめです。"
    )
    llm = _MockLLM([(answer, None), (answer, None), (answer, None)])
    state = _state("明示した条件どおりに更新してください")

    result, output = _run(llm, state)

    assert remaining == []
    assert state.acceptance_retry_count == 2
    assert state.exit_reason == "final_answer_acceptance_unresolved"
    assert "【警告: 受け入れ条件未解決】" in result
    assert "この回答は正常完了ではありません" in result
    assert "未解決の警告を明示して終了" in output
    feedback = [
        message for message in state.chat_history.messages
        if str(message.get("content", "")).startswith("【システム: 受け入れ条件")
    ]
    assert len(feedback) == 2


def test_state_only_completion_failure_returns_to_planning_without_duplicate(monkeypatch):
    remaining = _install_acceptance(monkeypatch, [["条件が未達です"], []])

    def _execute(context, tool_name, tool_args, output_fn):
        if tool_name == "run_command":
            return "1 passed in 0.01s"
        return "Success: 更新しました"

    monkeypatch.setattr("engine.execute_tool", _execute)
    state_only_answer = (
        "結論: 指定された編集とテストは完了しました。変更内容と検証結果を確認済みで、"
        "必要な状態も保存しました。以上で依頼された作業は完了です。"
    )
    final_answer = (
        "結論: 再確認後、受け入れ条件を満たしました。理由と検証結果も確認済みです。"
        "必要な変更はすべて反映され、作業は正常に完了しました。"
    )
    llm = _MockLLM([
        ("ファイルを編集します。", _tc("write_file", {"path": "a.py", "content": "x = 1\n"})),
        ("テストします。", _tc("run_command", {"command": "pytest -q"})),
        (state_only_answer, _tc("update_state", {"current_step": "完了"})),
        (final_answer, None),
    ])
    state = _state("条件どおりにファイルを更新してください")

    result, _ = _run(llm, state)

    assert remaining == []
    assert result == final_answer
    assert state.acceptance_retry_count == 1
    assert state.exit_reason.startswith("final_answer (")
    occurrences = sum(
        state_only_answer in str(message.get("content", ""))
        for message in state.chat_history.messages
        if message.get("role") == "assistant"
    )
    assert occurrences == 1


def test_simple_direct_answer_cannot_bypass_acceptance_gate(monkeypatch):
    remaining = _install_acceptance(monkeypatch, [["条件が未達です"], []])
    monkeypatch.setattr(
        "engine.execute_tool",
        lambda context, tool_name, tool_args, output_fn: "[get_cwd の実行結果: /home/user]",
    )
    direct_answer = "現在のディレクトリは /home/user です。"
    llm = _MockLLM([
        ("現在のディレクトリを確認します。", _tc("get_cwd")),
        (direct_answer, None),
        (direct_answer, None),
    ])
    state = _state("現在のディレクトリを教えて")

    result, _ = _run(llm, state)

    assert remaining == []
    assert result == direct_answer
    assert state.acceptance_retry_count == 1
    assert state.exit_reason.startswith("final_answer_simple_direct")
    assert len(llm.captured) == 3


def test_no_acceptance_condition_keeps_simple_direct_behavior(monkeypatch):
    monkeypatch.setattr("engine.derive_acceptance", lambda *args: [])
    monkeypatch.setattr(
        "engine.validate_acceptance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("条件なしでは検証しない")
        ),
    )
    monkeypatch.setattr(
        "engine.execute_tool",
        lambda context, tool_name, tool_args, output_fn: "[get_cwd の実行結果: /home/user]",
    )
    monkeypatch.setattr("engine.BEST_OF_ANSWER_ENABLED", False)
    monkeypatch.setattr("engine.LESSONS_ENABLED", False)
    direct_answer = "現在のディレクトリは /home/user です。"
    llm = _MockLLM([
        ("現在のディレクトリを確認します。", _tc("get_cwd")),
        (direct_answer, None),
    ])
    state = _state("現在のディレクトリを教えて")

    result, _ = _run(llm, state)

    assert result == direct_answer
    assert state.exit_reason.startswith("final_answer_simple_direct")


def test_state_only_completion_does_not_mask_max_tool_calls(monkeypatch):
    monkeypatch.setattr("engine.derive_acceptance", lambda *args: [])
    monkeypatch.setattr("engine.FORCE_FINAL_ANSWER_ON_LIMIT", False)
    monkeypatch.setattr("engine.LESSONS_ENABLED", False)
    def _execute(context, tool_name, tool_args, output_fn):
        if tool_name == "run_command":
            return "1 passed in 0.01s"
        return "Success: 更新しました"

    monkeypatch.setattr("engine.execute_tool", _execute)
    state_only_answer = (
        "結論: 指定された編集とテストは完了しました。変更内容と検証結果を確認済みで、"
        "必要な状態も保存しました。以上で依頼された作業は完了です。"
    )
    llm = _MockLLM([
        ("ファイルを編集します。", _tc("write_file", {"path": "a.py", "content": "x = 1\n"})),
        ("テストします。", _tc("run_command", {"command": "pytest -q"})),
        (state_only_answer, _tc("update_state", {"current_step": "完了"})),
    ])
    state = _state("条件どおりにファイルを更新してください")
    state.max_tool_calls = 3

    result, _ = _run(llm, state)

    assert result == ""
    assert state.exit_reason.startswith("max_tool_calls_reached")


def test_forced_final_warns_about_limit_and_unresolved_acceptance(monkeypatch):
    remaining = _install_acceptance(monkeypatch, [["CHANGELOG.md が未達です"]])
    monkeypatch.setattr("engine.FORCE_FINAL_ANSWER_ON_LIMIT", True)
    monkeypatch.setattr(
        "engine.execute_tool",
        lambda context, tool_name, tool_args, output_fn: "[get_cwd result] /workspace",
    )
    forced = "現時点の情報をまとめました。"
    llm = _MockLLM([
        ("ディレクトリを確認します。", _tc("get_cwd")),
        (forced, None),
    ])
    state = _state("明示条件どおりに更新してください")
    state.max_tool_calls = 1

    result, _ = _run(llm, state)

    assert remaining == []
    assert state.exit_reason.startswith("max_tool_calls_reached_with_final")
    assert "【警告: ツール実行上限到達】" in result
    assert "【警告: 受け入れ条件未解決】" in result
    assistants = [
        str(message.get("content", ""))
        for message in state.chat_history.messages
        if message.get("role") == "assistant"
    ]
    assert any("【警告: ツール実行上限到達】" in text for text in assistants)
    assert "終了時点でも確認できませんでした" in result
    assert "2回の再試行後" not in result


def test_empty_response_fallback_cannot_hide_unresolved_acceptance(monkeypatch):
    remaining = _install_acceptance(
        monkeypatch,
        [["CHANGELOG.md が未達です"], ["CHANGELOG.md が未達です"]],
    )
    candidate = (
        "結論: 更新作業は完了しました。変更内容と検証結果を確認し、"
        "指定された成果物へ必要な情報を反映したという最終報告です。"
    )
    llm = _MockLLM([(candidate, None), ("", None), ("", None), ("", None)])
    state = _state("明示条件どおりに更新してください")
    state.tool_evidence_guardrail_count = 1

    result, _ = _run(llm, state)

    assert remaining == []
    assert state.exit_reason.startswith("fallback_response")
    assert candidate in result
    assert "【警告: 空応答フォールバック】" in result
    assert "【警告: 受け入れ条件未解決】" in result
    assert "この回答は正常完了ではありません" in result
    assistants = [
        str(message.get("content", ""))
        for message in state.chat_history.messages
        if message.get("role") == "assistant"
    ]
    assert any("【警告: 受け入れ条件未解決】" in text for text in assistants)


def test_continuation_limit_always_marks_partial_answer(monkeypatch):
    remaining = _install_acceptance(monkeypatch, [["条件が未達です"]])
    chunks = [
        (f"継続出力の断片{i}です。", None, "length")
        for i in range(8)
    ]
    llm = _MockLLM(chunks)
    state = _state("明示条件どおりに長い成果物を作成してください")

    result, _ = _run(llm, state)

    assert remaining == []
    assert state.exit_reason.startswith("continuation_limit")
    assert "【警告: 出力継続上限到達】" in result
    assert "作業完了を保証しません" in result
    assert "【警告: 受け入れ条件未解決】" in result
