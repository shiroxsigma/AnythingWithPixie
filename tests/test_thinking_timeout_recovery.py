"""Reasoning limits must not discard tool output or consume empty retries."""
import types

import engine
from state import AgentState
from test_acceptance_gate_engine import _MockLLM, _context, _tc


class StreamingLLM(_MockLLM):
    def __init__(self, scripts, clock):
        super().__init__(scripts)
        self.clock = clock
        self.closed = 0

    def create_chat_completion(self, *args, **kwargs):
        script = self.scripts.pop(0)
        def generate():
            try:
                for at, delta, finish in script:
                    self.clock[0] = at
                    yield {"choices": [{"delta": delta, "finish_reason": finish}]}
            finally:
                self.closed += 1
        return generate()


def setup(monkeypatch, scripts):
    clock = [1.0]
    monkeypatch.setattr(engine, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(engine, "DEEP_THINK_BUDGET_SEC", 1)
    monkeypatch.setattr(engine, "BEST_OF_ANSWER_ENABLED", False)
    monkeypatch.setattr(engine, "LESSONS_ENABLED", False)
    monkeypatch.setattr(engine, "derive_acceptance", lambda *args: [])
    llm = StreamingLLM(scripts, clock)
    context = _context(llm)
    context.code_mode = True
    state = AgentState()
    state.chat_history.add("user", "Create index.html with write_file.")
    return llm, context, state


def test_reasoning_limit_preserves_tool_chunk_and_remaining_arguments(monkeypatch):
    call = _tc("write_file", {"path": "index.html", "content": "ok"})
    arguments = call[0]["function"]["arguments"]
    call[0]["function"]["arguments"] = arguments[:15]
    llm, context, state = setup(monkeypatch, [[
        (1, {"reasoning_content": "plan"}, None),
        (3, {"reasoning_content": "done", "tool_calls": call}, None),
        (4, {"tool_calls": [{"index": 0, "function": {
            "arguments": arguments[15:],
        }}]}, "tool_calls"),
    ]])
    _, calls = engine.node_plan(context, state, output_fn=lambda *a, **k: None)
    assert calls[0]["function"]["name"] == "write_file"
    assert calls[0]["function"]["arguments"] == arguments
    assert state.phase != "THINKING_INTERRUPTED"


def test_reasoning_limit_does_not_discard_answer_in_same_chunk(monkeypatch):
    llm, context, state = setup(monkeypatch, [[
        (1, {"reasoning_content": "plan"}, None),
        (3, {"reasoning_content": "done", "content": "Completed."}, "stop"),
    ]])
    content, _ = engine.node_plan(context, state, output_fn=lambda *a, **k: None)
    assert content == "Completed."
    assert state.phase != "THINKING_INTERRUPTED"


def test_repeated_local_timeouts_have_distinct_bounded_exit(monkeypatch):
    llm, context, state = setup(monkeypatch, [
        [(1 + n * 10, {"reasoning_content": "plan"}, None),
         (3 + n * 10, {"reasoning_content": "still planning"}, None)]
        for n in range(3)
    ])
    output = []
    engine.run_graph(context, state, output_fn=lambda text="", **kwargs: output.append(text))
    assert state.exit_reason.startswith("thinking_timeout")
    assert not any("空の応答" in text for text in output)
    assert llm.closed == 3
    history = str(state.chat_history.messages)
    assert "元のユーザー依頼を継続" in history
    assert "Create index.html" in history
