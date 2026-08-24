"""編集世代に紐づくテスト成功証跡と、完了報告の高速受理を検証する。"""

import copy
import json
import types

import pytest

from engine import _EditVerificationTracker, run_graph
from state import AgentState


def _observe_edit(tracker, path="a.py", result="Success: updated"):
    tracker.observe_tool_result("write_file", {"path": path}, result)


def _observe_pytest(tracker, result="2 passed in 0.02s"):
    tracker.observe_tool_result(
        "run_command", {"command": "python -m pytest -q"}, result
    )


def test_multiple_edits_are_verified_as_one_generation_snapshot():
    tracker = _EditVerificationTracker()
    _observe_edit(tracker, "a.py")
    _observe_edit(tracker, "b.py")
    _observe_pytest(tracker)

    assert tracker.edit_generation == 2
    assert tracker.verified_edit_generation == 2
    assert tracker.accepts_completion_report("2ファイルの修正とテストが完了しました。")


def test_explicit_pytest_file_verifies_all_workset_tests():
    tracker = _EditVerificationTracker(
        working_snapshots={
            "app.py": {"content": ""},
            "test_app.py": {"content": ""},
        }
    )
    _observe_edit(tracker, "app.py")
    tracker.observe_tool_result(
        "run_command",
        {"command": "python -m pytest -q test_app.py"},
        "1 passed in 0.02s",
    )

    assert tracker.has_current_verification()


def test_explicit_pytest_file_does_not_verify_other_workset_tests():
    tracker = _EditVerificationTracker(
        working_snapshots={
            "test_app.py": {"content": ""},
            "test_other.py": {"content": ""},
        }
    )
    _observe_edit(tracker, "app.py")
    tracker.observe_tool_result(
        "run_command",
        {"command": "python -m pytest -q test_app.py"},
        "1 passed in 0.02s",
    )

    assert not tracker.has_current_verification()


def test_edit_after_test_invalidates_verification():
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)
    _observe_edit(tracker, "b.py")

    assert tracker.edit_generation == 2
    assert tracker.verified_edit_generation == -1
    assert not tracker.accepts_completion_report("修正が完了しました。")


@pytest.mark.parametrize(
    ("tool_name", "tool_args"),
    [
        ("delete_file", {"path": "old.py"}),
        ("move_file", {"src": "a.py", "dst": "b.py"}),
        ("make_directory", {"path": "generated"}),
        ("run_command", {"command": "python format_generated.py"}),
        ("run_python", {"code": "generate_files()"}),
    ],
)
def test_post_test_mutation_invalidates_until_reverified(tool_name, tool_args):
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)

    tracker.observe_tool_result(tool_name, tool_args, "Success: done")
    assert not tracker.accepts_completion_report("修正が完了しました。")

    _observe_pytest(tracker)
    assert tracker.accepts_completion_report("修正と再テストが完了しました。")


def test_unknown_extension_tool_invalidates_verification_until_retested():
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)

    tracker.observe_tool_result(
        "format_project", {"path": "."}, "Success: formatted project"
    )

    assert not tracker.has_current_verification()
    _observe_pytest(tracker)
    assert tracker.has_current_verification()


def test_failed_test_invalidates_verification_until_a_later_test_passes():
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)
    _observe_pytest(tracker, "Error (1): 1 failed")

    assert not tracker.accepts_completion_report("修正が完了しました。")

    # 一時的な失敗でもその時点の証跡は失効するが、後続の成功で同じ編集世代を再検証できる。
    _observe_pytest(tracker)
    assert tracker.accepts_completion_report("修正と再テストが完了しました。")


@pytest.mark.parametrize(
    "result",
    [
        "Execution Timeout: コマンドの実行が30秒を超えました。",
        "Execution Failed: PowerShell could not start",
        "Error (1): 1 failed",
    ],
)
def test_command_failure_forms_never_create_verification_proof(result):
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker, result)

    assert tracker.verified_edit_generation == -1
    assert not tracker.accepts_completion_report("修正が完了しました。")


def test_unresolved_tool_error_and_fast_gate_failure_invalidate_verification():
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    tracker.observe_tool_result("read_file", {"path": "missing.py"}, "Error: missing")
    _observe_pytest(tracker)
    assert not tracker.accepts_completion_report("修正が完了しました。")

    tracker = _EditVerificationTracker()
    tracker.observe_tool_result(
        "write_file",
        {"path": "a.py"},
        "Success: updated\n[ruff check failed]",
        validation_failed=True,
    )
    _observe_pytest(tracker)
    assert not tracker.accepts_completion_report("修正が完了しました。")


@pytest.mark.parametrize("content", ["", "次にテスト結果を確認します。"])
def test_empty_or_action_promise_is_not_a_completion_report(content):
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)

    assert not tracker.accepts_completion_report(content)


@pytest.mark.parametrize(
    "content",
    [
        "テストは失敗しました。",
        "修正できたかは不明です。",
        "未解決の問題があります。",
        "修正は完了しましたが、テストは未実施です。",
        "修正しましたが、テストは成功しませんでした。",
        "修正しましたが、エラーが残っています。",
        "更新しましたが、問題があります。",
        "Changes are complete, but tests failed.",
        "Changes are complete but not tested.",
    ],
)
def test_failure_report_is_not_accepted_as_completion(content):
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)

    assert not tracker.accepts_completion_report(content)


def test_negated_failure_in_a_completion_report_is_accepted():
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)

    assert tracker.accepts_completion_report(
        "修正とテストは完了し、失敗はありません。"
    )
    assert tracker.accepts_completion_report(
        "修正とテストは完了し、エラーや問題はありません。"
    )


@pytest.mark.parametrize(
    "report",
    [
        "CHANGELOG.mdに新しい節を追加しました。",
        "リリース情報を追記しました。",
        "指定位置へ見出しを挿入しました。",
        "必要な内容を反映しました。",
        "新しい成果物を作成しました。",
    ],
)
def test_concrete_edit_verbs_are_completion_reports(report):
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)

    assert tracker.accepts_completion_report(report)


def test_report_saying_more_work_is_needed_is_not_accepted():
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    _observe_pytest(tracker)

    assert not tracker.accepts_completion_report(
        "一部の更新は完了しましたが、まだ追加対応が必要です。"
    )


@pytest.mark.parametrize(
    ("command", "result"),
    [
        ("pytest -q", "0 passed in 0.01s"),
        ("npm test", "Tests: 0 passed"),
        ("cargo test", "test result: ok. 0 passed; 0 failed"),
    ],
)
def test_zero_executed_tests_do_not_create_completion_proof(command, result):
    tracker = _EditVerificationTracker()
    _observe_edit(tracker)
    tracker.observe_tool_result("run_command", {"command": command}, result)

    assert not tracker.accepts_completion_report("修正が完了しました。")


def test_test_run_outside_workspace_does_not_verify_the_edit(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "other-project"
    workspace.mkdir()
    outside.mkdir()
    tracker = _EditVerificationTracker(workspace)
    _observe_edit(tracker, "a.py")

    tracker.observe_tool_result(
        "run_command",
        {"command": "pytest -q", "working_directory": str(outside)},
        "1 passed in 0.01s",
    )

    assert not tracker.has_current_verification()


def test_specific_unrelated_test_does_not_verify_the_whole_edit(tmp_path):
    tracker = _EditVerificationTracker(tmp_path)
    _observe_edit(tracker, "a.py")

    tracker.observe_tool_result(
        "run_command",
        {"command": "pytest -q tests/test_unrelated.py"},
        "1 passed in 0.01s",
    )

    assert not tracker.has_current_verification()


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q -k=unrelated",
        "pytest -q --ignore=tests/integration",
        "pytest -q --deselect=tests/test_slow.py::test_case",
        "pytest -q --lf",
        "npm test -- --runTestsByPath=tests/unrelated.test.js",
        "cargo test --test=unrelated",
        "go test -run=Unrelated ./...",
        "dotnet test --filter=Unrelated",
    ],
)
def test_scope_limiting_test_flags_do_not_verify_the_whole_edit(tmp_path, command):
    tracker = _EditVerificationTracker(tmp_path)
    _observe_edit(tracker, "a.py")

    tracker.observe_tool_result(
        "run_command", {"command": command}, "2 passed in 0.01s"
    )

    assert not tracker.has_current_verification()


def test_broad_test_in_workspace_still_verifies_the_edit(tmp_path):
    tracker = _EditVerificationTracker(tmp_path)
    _observe_edit(tracker, "a.py")

    tracker.observe_tool_result(
        "run_command",
        {"command": "pytest --maxfail=1 -q", "working_directory": str(tmp_path)},
        "2 passed in 0.01s",
    )

    assert tracker.has_current_verification()


class _MockLLM:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.captured = []
        self.captured_tools = []
        self.n_ctx = 32768

    def create_chat_completion(
        self, messages, *, max_tokens, temperature, stream, tools=None,
        tool_choice=None, **kwargs,
    ):
        self.captured.append(copy.deepcopy(messages))
        self.captured_tools.append(copy.deepcopy(tools))
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


def _tool_call(name, args, index):
    return [{
        "index": 0,
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }]


def _tool_calls(*items):
    calls = []
    for index, (name, args) in enumerate(items):
        call = _tool_call(name, args, index + 1)[0]
        call["index"] = index
        calls.append(call)
    return calls


def _run_verified_completion(monkeypatch, tmp_path, final_report):
    monkeypatch.setattr("engine.BEST_OF_EDIT_ENABLED", False)
    monkeypatch.setattr("engine.BEST_OF_ANSWER_ENABLED", True)
    monkeypatch.setattr("engine.LESSONS_ENABLED", False)
    monkeypatch.setattr("engine.derive_acceptance", lambda *args, **kwargs: [])
    monkeypatch.setattr("engine.validate_acceptance", lambda *args, **kwargs: [])

    def _execute(context, tool_name, tool_args, output_fn):
        if tool_name == "run_command":
            return "2 passed in 0.02s"
        return f"Success: {tool_args['path']} updated"

    monkeypatch.setattr("engine.execute_tool", _execute)

    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    llm = _MockLLM([
        ("最初のファイルを編集します。", _tool_call(
            "write_file", {"path": str(first), "content": "A = 1\n"}, 1
        )),
        ("次のファイルを編集します。", _tool_call(
            "write_file", {"path": str(second), "content": "B = 2\n"}, 2
        )),
        ("テストを実行します。", _tool_call(
            "run_command", {"command": "python -m pytest -q"}, 3
        )),
        (final_report, None),
    ])
    context = types.SimpleNamespace(
        llm=llm,
        code_mode=False,
        force_deep=False,
        supports_tool_role=False,
        debug_mode=False,
        phase="EXECUTING",
        review_mode=False,
    )
    state = AgentState()
    state.chat_history.add("user", "二つの成果物を更新してテストしてください")

    def _system_builder(context, state_board, **kwargs):
        return "You are a helpful test assistant."

    answer = run_graph(
        context,
        state,
        output_fn=lambda *args, **kwargs: None,
        system_msg_builder=_system_builder,
    )
    return answer, state, llm


def test_verified_short_report_skips_short_answer_regeneration(monkeypatch, tmp_path):
    report = "2ファイルの修正とテストが完了しました。"
    answer, state, llm = _run_verified_completion(
        monkeypatch, tmp_path, report
    )

    assert answer == report
    assert len(llm.captured) == 4
    assert "final_answer" in state.exit_reason
    assert not any("short_answer_guardrail" in signal for signal in state.failure_signals)


def test_verified_margin_report_skips_best_of_resampling(monkeypatch, tmp_path):
    # 「対応案」で完全性スコアが best-of 対象帯に入るが、テスト証跡があるため再生成しない。
    report = "対応案として2ファイルを修正しました。"
    answer, state, llm = _run_verified_completion(
        monkeypatch, tmp_path, report
    )

    assert answer == report
    assert len(llm.captured) == 4
    assert "final_answer" in state.exit_reason


def test_test_then_edit_in_same_batch_does_not_force_the_next_plan(monkeypatch, tmp_path):
    monkeypatch.setattr("engine.BEST_OF_EDIT_ENABLED", False)
    monkeypatch.setattr("engine.BEST_OF_ANSWER_ENABLED", False)
    monkeypatch.setattr("engine.LESSONS_ENABLED", False)
    monkeypatch.setattr("engine.derive_acceptance", lambda *args, **kwargs: [])

    def _execute(context, tool_name, tool_args, output_fn):
        if tool_name == "run_command":
            return "2 passed in 0.02s"
        return "Success: updated"

    monkeypatch.setattr("engine.execute_tool", _execute)
    llm = _MockLLM([
        (
            "編集と途中確認を行います。",
            _tool_calls(
                ("write_file", {"path": str(tmp_path / "a.py"), "content": "A=1\n"}),
                ("run_command", {"command": "pytest -q"}),
                ("write_file", {"path": str(tmp_path / "b.py"), "content": "B=2\n"}),
            ),
        ),
        ("最後の編集後に再テストします。", _tool_call(
            "run_command", {"command": "pytest -q"}, 4
        )),
        ("2ファイルの修正と再テストが完了しました。", None),
    ])
    context = types.SimpleNamespace(
        llm=llm,
        code_mode=False,
        force_deep=False,
        supports_tool_role=False,
        debug_mode=False,
        phase="EXECUTING",
        review_mode=False,
    )
    state = AgentState()
    state.chat_history.add("user", "二つのファイルを更新してテストしてください")

    answer = run_graph(
        context,
        state,
        output_fn=lambda *args, **kwargs: None,
        system_msg_builder=lambda *args, **kwargs: "test system",
    )

    assert answer == "2ファイルの修正と再テストが完了しました。"
    assert llm.captured_tools[1] is not None
    assert "finalize_after_verification" not in state.__dict__
