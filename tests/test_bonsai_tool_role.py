"""Bonsai tool results retain their protocol role through all context builders."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import engine
import main
import paths
import pixie_core
import registry
from llm_client import LMStudioBackend
from state import AgentState

_RUNNER_PATH = Path(__file__).resolve().parent.parent / "evals" / "runner.py"
_SPEC = importlib.util.spec_from_file_location("bonsai_eval_runner", _RUNNER_PATH)
eval_runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_runner)

MODELS = [
    ("prism-ml/Ternary-Bonsai-2-27B-gguf", True, False),
    ("bonsai2-27b", True, False),
    ("bosai2-27b", True, False),
    ("LiquidAI/LFM2.5-2.6B", True, True),
    ("generic-model", False, False),
    ("Bonsai-27B", False, False),
]


@pytest.fixture(autouse=True)
def isolated_context(tmp_path, monkeypatch):
    monkeypatch.setattr(LMStudioBackend, "_fetch_n_ctx", lambda _: 32768)
    workspace_token = paths.bind_workspace(str(tmp_path))
    board_token = registry._state_board_var.set(None)
    try:
        yield
    finally:
        registry._state_board_var.reset(board_token)
        paths.reset_workspace(workspace_token)


def server(model):
    return {"name": model, "model": model, "base_url": "http://localhost:1/v1"}


def build_context(factory, model, tmp_path):
    if factory == "embedded":
        return pixie_core.create_engine(server(model), str(tmp_path)).context
    return eval_runner._build_context(LMStudioBackend.from_config(server(model)))


@pytest.mark.parametrize("factory", ["embedded", "eval"])
@pytest.mark.parametrize("model, native_tools, is_lfm", MODELS)
def test_contexts_detect_native_tools_without_lfm_overrides(factory, model, native_tools, is_lfm, tmp_path):
    context = build_context(factory, model, tmp_path)
    assert context.supports_tool_role is native_tools
    assert context.is_lfm25 is is_lfm


@pytest.mark.parametrize("model, native_tools, is_lfm", MODELS)
def test_cli_startup_sets_model_capabilities(model, native_tools, is_lfm, monkeypatch):
    backend = LMStudioBackend.from_config(server(model))
    monkeypatch.setattr(main, "TrajectoryLogger", lambda: None)
    monkeypatch.setattr(main, "_load_startup_toolpacks", lambda *_: None)
    monkeypatch.setattr(main, "_load_delegate_server", lambda *_: None)
    monkeypatch.setattr(main, "select_model", lambda *_: ("LMSTUDIO", "", server(model)))
    monkeypatch.setattr(main, "initialize_backend", lambda **_: (backend, False, False, is_lfm, False))
    monkeypatch.setattr(main.platform, "system", lambda: "Linux")
    stream = SimpleNamespace(reconfigure=lambda **_: None)
    monkeypatch.setattr(main, "sys", SimpleNamespace(stdout=stream, stdin=stream))
    context = main.setup_application(SimpleNamespace(no_capture=True))
    assert context.supports_tool_role is native_tools
    assert context.is_lfm25 is is_lfm


@pytest.mark.parametrize("model, native_tools, is_lfm", MODELS)
def test_cli_api_switch_recomputes_capabilities(model, native_tools, is_lfm, monkeypatch):
    commands = iter(["/api", "exit"])
    chat_input = SimpleNamespace(get_chat_input=lambda *a, **k: next(commands))
    monkeypatch.setattr(main, "create_chat_input_session", lambda **_: chat_input)
    monkeypatch.setattr(main, "_load_lmstudio_servers", lambda _: [server(model)])
    monkeypatch.setattr("builtins.input", lambda *_: "1")
    context = main.AppContext()
    # An opposite prior setting catches stale flags after switching models.
    context.supports_tool_role = not native_tools
    context.is_lfm25 = not is_lfm
    main.run_cli_chat(context)
    assert context.llm_model_name == model
    assert context.supports_tool_role is native_tools
    assert context.is_lfm25 is is_lfm


@pytest.mark.parametrize("factory", ["embedded", "eval"])
@pytest.mark.parametrize("model, native_tools, is_lfm", [MODELS[1], MODELS[3], MODELS[4]])
def test_node_plan_preserves_native_tool_result_ids(factory, model, native_tools, is_lfm, tmp_path, monkeypatch):
    context = build_context(factory, model, tmp_path)
    sent = []

    def completion(messages, **kwargs):
        sent.append((copy.deepcopy(messages), kwargs))
        yield {"choices": [{"delta": {"content": "The file contains 42."}, "finish_reason": "stop"}]}

    monkeypatch.setattr(context.llm, "create_chat_completion", completion)
    state = AgentState()
    state.chat_history.add("user", "Read config.txt and report the value.")
    state.chat_history.add("assistant", tool_calls=[{
        "id": "call_read_config", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"config.txt"}'},
    }])
    state.chat_history.add("tool", "42", tool_call_id="call_read_config")

    engine.node_plan(
        context, state, tool_choice="required", output_fn=lambda *a, **k: None,
        system_msg_builder=lambda *a, **k: "Use the provided tools.",
    )

    messages, kwargs = sent[0]
    results = [message for message in messages if message["role"] == "tool"]
    if native_tools:
        assert results == [{"role": "tool", "content": "42", "tool_call_id": "call_read_config"}]
        call = next(message for message in messages if message.get("tool_calls"))["tool_calls"][0]
        assert results[0]["tool_call_id"] == call["id"]
    else:
        assert results == []
        assert any(message["role"] == "user" and "[ツール結果]\n42" in message.get("content", "")
                   for message in messages)
    assert kwargs["tool_choice"] == ("auto" if is_lfm else "required")
    assert any(message["role"] == "tool" and message["tool_call_id"] == "call_read_config"
               for message in state.chat_history.messages)
