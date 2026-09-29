"""Nested completions must not promote transport failures into text or tool calls."""

import pytest

import engine
import subagent
from pixie_core import llm_client
from pixie_core.turn_control import TurnStopped


def _text(content):
    return {"choices": [{"delta": {"content": content}}]}


def _error(marker=True):
    chunk = {"choices": [{"delta": {"content": "body timeout"}, "finish_reason": "error"}]}
    if marker:
        chunk["__llm_error__"] = "TimeoutError: body timeout"
    return chunk


class _ScriptedLLM:
    n_ctx = 32768

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.scripts.pop(0))


@pytest.mark.parametrize("collector", [subagent._collect_response, subagent._collect_subquery_response])
@pytest.mark.parametrize("marker", [True, False])
def test_collectors_reject_partial_output_and_close_stream(collector, marker):
    closed = []

    def completion():
        try:
            yield _text("Incomplete generated text")
            yield _error(marker)
            pytest.fail("terminal errors must stop consumption")
        finally:
            closed.append(True)

    with pytest.raises(RuntimeError, match="body timeout"):
        collector(completion())
    assert closed == [True]


@pytest.mark.parametrize("collector", [subagent._collect_response, subagent._collect_subquery_response])
@pytest.mark.parametrize("marker", [True, False])
def test_collectors_reject_nonstream_error_before_returning_message(collector, marker):
    payload = {"choices": [{"message": {"content": "Incomplete text"}, "finish_reason": "error"}]}
    if marker:
        payload["__llm_error__"] = "body timeout"

    with pytest.raises(RuntimeError, match="LLM completion failed"):
        collector(payload)


@pytest.mark.parametrize("collector", [subagent._collect_response, subagent._collect_subquery_response])
def test_collectors_preserve_successful_text(collector):
    assert collector({"choices": [{"message": {"content": "Answer"}}]}) == "Answer"
    assert collector(iter([_text(None), _text("An"), _text("swer")])) == "Answer"


@pytest.mark.parametrize("collector", [subagent._collect_response, subagent._collect_subquery_response])
def test_collectors_propagate_cancellation_unchanged(collector):
    cancellation = TurnStopped("cancelled")
    closed = []

    def completion():
        try:
            yield _text("Partial")
            raise cancellation
        finally:
            closed.append(True)

    with pytest.raises(TurnStopped) as caught:
        collector(completion())
    assert caught.value is cancellation
    assert closed == [True]


def test_body_timeout_from_backend_is_rejected_by_subquery_collector(monkeypatch):
    class BrokenResponse:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def read(self):
            raise TimeoutError("body timeout")

    response = BrokenResponse()
    monkeypatch.setattr(llm_client.LMStudioBackend, "_fetch_n_ctx", lambda _: 32768)
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    backend = llm_client.LMStudioBackend("http://localhost:1/v1")

    with pytest.raises(RuntimeError, match="TimeoutError: body timeout"):
        subagent._collect_subquery_response(backend.create_chat_completion([], stream=False))
    assert response.closed


def test_failed_whiteboard_completion_preserves_existing_contents(tmp_path, monkeypatch):
    path = tmp_path / "CONTEXT_SUMMARY.md"
    existing = "# Working memory\n\nPreserve the user's unfinished task.\n"
    path.write_text(existing, encoding="utf-8")
    monkeypatch.setattr(engine, "get_whiteboard_path", lambda: str(path))
    llm = _ScriptedLLM([_text("Incomplete replacement"), _error()])
    output = []

    engine._update_whiteboard(
        llm, [{"role": "user", "content": "Remember my task"}],
        output_fn=lambda text, **kwargs: output.append(text),
    )

    assert path.read_text(encoding="utf-8") == existing
    assert "ホワイトボードの更新に失敗しました" in "".join(output)
    assert "更新が完了しました" not in "".join(output)


def test_failed_delegate_stream_never_executes_partial_tool_calls(monkeypatch):
    partial_call = {"choices": [{"delta": {"tool_calls": [{
        "index": 0, "id": "call_1", "type": "function",
        "function": {"name": "get_cwd", "arguments": "{}"},
    }]}}]}
    llm = _ScriptedLLM([partial_call, _error()])
    executed = []
    monkeypatch.setattr(subagent.tools, "execute_builtin_tool", lambda *args: executed.append(args))

    answer = subagent.run_agent_subquery(llm, question="現在地を確認してください")

    assert "LLM 呼び出し失敗" in answer
    assert "body timeout" in answer
    assert executed == []
    assert len(llm.calls) == 1


def test_failed_delegate_summary_does_not_return_partial_conclusion():
    llm = _ScriptedLLM([_text("")], [_text("Unverified conclusion"), _error()])

    answer = subagent.run_agent_subquery(llm, question="調べてください", max_steps=1)

    assert "結論生成失敗" in answer
    assert "Unverified conclusion" not in answer
    assert len(llm.calls) == 2


def test_failed_generated_input_is_not_sent_to_program():
    llm = _ScriptedLLM([_error()])

    assert subagent._generate_runpython_input(llm, "input()", "Name:", []) is None


def test_delegate_cancellation_is_not_converted_to_failure_text():
    cancellation = TurnStopped("cancelled")

    class CancelledLLM:
        n_ctx = 32768

        def create_chat_completion(self, **kwargs):
            yield _text("Partial")
            raise cancellation

    with pytest.raises(TurnStopped) as caught:
        subagent.run_agent_subquery(CancelledLLM(), question="調べてください")
    assert caught.value is cancellation
