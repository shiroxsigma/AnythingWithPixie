"""Embedded ChangeSet receipts must participate in normal edit verification."""
import os
import sys
from pathlib import Path

import pytest
from test_acceptance_gate_engine import _context
from test_verified_completion import _MockLLM, _tool_call

import engine
import paths
from pixie_core._api import Engine
from state import AgentState


def _change(content="VALUE = 2\n", change_id="chg_receipt"):
    return {"id": change_id, "changes": [{
        "path": "module.py", "operations": [{"kind": "write_file", "content": content}],
    }]}


def _api(tmp_path):
    return Engine(context=object(), state=AgentState(), workspace=str(tmp_path))


def test_applied_receipt_is_not_verification_and_new_edit_invalidates(tmp_path):
    api = _api(tmp_path)
    tracker = engine._EditVerificationTracker(tmp_path)
    assert api.apply_changeset(_change())["applied"]
    assert engine._observe_applied_changesets(api.state, tracker)
    assert not engine._observe_applied_changesets(api.state, tracker)
    assert tracker.edit_generation == 1
    assert not tracker.has_current_verification()
    args = {"command": "pytest -q", "working_directory": str(tmp_path)}
    tracker.observe_tool_result("run_command", args, "1 passed in 0.01s")
    assert tracker.has_current_verification()
    assert api.apply_changeset(_change("VALUE = 3\n", "chg_second"))["applied"]
    engine._observe_applied_changesets(api.state, tracker)
    assert tracker.edit_generation == 2
    assert not tracker.has_current_verification()
    tracker.observe_tool_result("run_command", args, "Error: 1 failed")
    assert not tracker.accepts_completion_report("修正が完了しました。")


def test_receipts_do_not_leak_between_sessions_or_turns(tmp_path):
    first, second = _api(tmp_path), _api(tmp_path)
    first.apply_changeset(_change())
    assert first.state.pending_applied_changesets
    assert second.state.pending_applied_changesets == []
    first.state.reset_for_new_turn()
    assert first.state.pending_applied_changesets == []
    assert not engine._observe_applied_changesets(first.state, engine._EditVerificationTracker(tmp_path))


def test_failed_or_unapproved_changeset_does_not_create_receipt(tmp_path):
    api = _api(tmp_path)
    assert api.preview_changeset(_change())["ok"]
    assert api.state.pending_applied_changesets == []
    invalid = _change()
    invalid["changes"][0]["base_hash"] = "0" * 64
    assert not api.apply_changeset(invalid)["applied"]
    assert api.state.pending_applied_changesets == []


def test_atomic_batch_preserves_multiple_edit_generations(tmp_path):
    api = _api(tmp_path)
    change = _change()
    change["changes"][0]["operations"].append({"kind": "append", "content": "MORE = 3\n"})
    change["changes"].append({"path": "other.py", "operations": [
        {"kind": "write_file", "content": "OTHER = 4\n"},
    ]})
    assert api.apply_changeset(change)["applied"]
    tracker = engine._EditVerificationTracker(tmp_path)
    engine._observe_applied_changesets(api.state, tracker)
    assert tracker.edit_generation == 3
    assert not tracker.has_current_verification()


@pytest.mark.parametrize("passes", [True, False])
def test_approval_edit_and_real_pytest_control_finalization(tmp_path, monkeypatch, passes):
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    (tmp_path / "test_module.py").write_text(
        "from module import VALUE\ndef test_value():\n    assert VALUE == 2\n", encoding="utf-8"
    )
    content = "VALUE = 2\n" if passes else "VALUE = 3\n"
    llm = _MockLLM([
        ("", _tool_call("write_file", {"path": "module.py", "content": content}, 1)),
        ("", _tool_call("run_command", {"command": "python -m pytest -q"}, 2)),
        ("修正とテストが完了しました。" if passes else "テストが失敗しました。修正は未完了です。", None),
    ])
    context = _context(llm)
    context.code_mode = True
    state = AgentState()
    state.chat_history.add("user", "Update module.py and run the tests.")
    api = Engine(context=context, state=state, workspace=str(tmp_path))
    monkeypatch.setattr(engine, "LESSONS_ENABLED", False)
    monkeypatch.setattr(engine, "BEST_OF_EDIT_ENABLED", False)
    monkeypatch.setattr(engine, "BEST_OF_ANSWER_ENABLED", False)
    monkeypatch.setattr(engine, "derive_acceptance", lambda *a, **k: [])
    executed = []
    original_execute = engine.execute_tool

    def capture_execution(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        executed.append(result)
        return result

    monkeypatch.setattr(engine, "execute_tool", capture_execution)

    def approve(calls, content):
        if calls[0]["function"]["name"] == "write_file":
            assert api.apply_changeset(_change("VALUE = 2\n" if passes else "VALUE = 3\n"))["applied"]
            return [], "Approved changes were applied. Run the tests next."
        return calls, None

    token = paths.bind_workspace(str(tmp_path))
    try:
        answer = engine.run_graph(context, state, interactive_fn=approve,
                                  output_fn=lambda *a, **k: None)
    finally:
        paths.reset_workspace(token)
    assert len(llm.captured) == 3
    assert ("1 passed" if passes else "1 failed") in str(executed), executed
    assert (llm.captured_tools[-1] is None) is passes
    assert state.pending_applied_changesets == []
    assert ("1 passed" if passes else "1 failed") in str(llm.captured[-1])
    assert ("完了しました" in answer) is passes
