"""Mixed main/delegate models must use the selected backend's tool protocol."""

from types import SimpleNamespace

import pytest

import subagent
from state import AgentState


@pytest.mark.parametrize("entrypoint", ["delegate", "edit_review", "design_review", "review_loop"])
@pytest.mark.parametrize("main_model, delegate_model, main_native, delegate_native", [
    ("bonsai2-27b", "generic-model", True, False),
    ("generic-model", "bonsai2-27b", False, True),
])
def test_selected_delegate_capability_reaches_every_subquery_entrypoint(
    entrypoint, main_model, delegate_model, main_native, delegate_native, monkeypatch,
):
    context = SimpleNamespace(
        llm=SimpleNamespace(model=main_model),
        delegate_llm=SimpleNamespace(model=delegate_model),
        supports_tool_role=main_native,
    )
    calls = []

    def capture(llm, **kwargs):
        calls.append((llm, kwargs["supports_tool_role"]))
        return "問題なし"

    monkeypatch.setattr(subagent, "run_agent_subquery", capture)
    monkeypatch.setattr(subagent, "_delegate_server_counter", 1)
    output = lambda *args, **kwargs: None
    if entrypoint == "delegate":
        subagent._execute_delegate_research(context, {"question": "調べてください"}, output)
    elif entrypoint == "edit_review":
        subagent._run_edit_review(context, "write_file", {"path": "a.py", "content": "VALUE = 1"}, output)
    elif entrypoint == "design_review":
        subagent._run_design_review(context, "設計案", "設計してください", output)
    else:
        state = AgentState()
        state.chat_history.add("user", "設計してください")
        state.chat_history.add("assistant", "設計案")
        subagent.run_review_loop(context, state, rounds=1, output_fn=output)

    assert calls == [(context.delegate_llm, delegate_native)]


@pytest.mark.parametrize("model", ["LiquidAI/LFM2.5-2.6B", "Ternary-Bonsai-2-27B", "bosai2-27b"])
def test_known_delegate_aliases_use_native_tools(model):
    context = SimpleNamespace(llm=object(), supports_tool_role=False)
    assert subagent._subquery_supports_tool_role(context, SimpleNamespace(model=model))


@pytest.mark.parametrize("override", [False, True])
def test_main_backend_preserves_explicit_context_override(override):
    context = SimpleNamespace(llm=SimpleNamespace(model="bonsai2-27b"), supports_tool_role=override)
    assert subagent._subquery_supports_tool_role(context, context.llm) is override


@pytest.mark.parametrize("fallback", [False, True])
def test_unidentified_delegate_preserves_existing_fallback(fallback):
    context = SimpleNamespace(llm=object(), supports_tool_role=fallback)
    assert subagent._subquery_supports_tool_role(context, object()) is fallback
