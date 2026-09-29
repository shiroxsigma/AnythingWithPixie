"""Connection policy must survive each public backend construction path."""

import contextvars
import io
import json

import pytest

import main
import pixie_core
import pixie_core._api as api
from llm_client import LMStudioBackend, initialize_backend
from pixie_core.turn_control import active_control


@pytest.fixture(autouse=True)
def no_discovery_network(monkeypatch):
    monkeypatch.setattr(LMStudioBackend, "_fetch_n_ctx", lambda _: 32768)


@pytest.mark.parametrize("model, expected", [
    ("prism-ml/Ternary-Bonsai-2-27B-gguf", 120),
    ("Bonsai-2-27B-PQ2_0.gguf", 120),
    ("bonsai2-27b", 120),
    ("bosai2-27b", 120),
    ("gemma-4", 30),
    ("Bonsai-27B", 30),
])
def test_model_defaults_and_explicit_override(model, expected):
    backend = LMStudioBackend("http://localhost:1/v1", model=model)
    assert backend.read_idle_timeout == expected
    assert backend.overall_timeout == 180
    assert backend.reasoning_effort == ("medium" if expected == 120 else None)
    explicit = LMStudioBackend("http://localhost:1/v1", model=model, read_idle_timeout=7)
    assert explicit.read_idle_timeout == 7


@pytest.mark.parametrize("key", ["overall_timeout", "read_idle_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_connection_limits_rejected_before_discovery(monkeypatch, key, value):
    def unexpected_discovery(_):
        pytest.fail("invalid configuration must fail before network access")
    monkeypatch.setattr(LMStudioBackend, "_fetch_n_ctx", unexpected_discovery)
    with pytest.raises(ValueError, match="finite and positive"):
        LMStudioBackend("http://localhost:1/v1", **{key: value})


def test_config_round_trip_through_cli_delegate_and_embedding(tmp_path, monkeypatch):
    server = {
        "base_url": "http://localhost:1/v1", "model": "bonsai2-27b",
        "overall_timeout": 240, "read_idle_timeout": 95,
        "reasoning_effort": "xhigh",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"servers": [server], "delegate_server": server}), encoding="utf-8")
    loaded = main._load_lmstudio_servers(path)[0]
    delegate = main._load_delegate_server(path)
    cli, *_ = initialize_backend("LMSTUDIO", lmstudio_config=loaded, use_vision_flag="n")
    assert cli.overall_timeout == 240
    assert cli.read_idle_timeout == 95
    assert cli.reasoning_effort == "xhigh"
    assert LMStudioBackend.from_config(delegate).read_idle_timeout == 95
    assert LMStudioBackend.from_config(delegate).reasoning_effort == "xhigh"
    engine = pixie_core.create_engine(loaded, str(tmp_path))
    assert engine.context.llm.reasoning_effort == "xhigh"

    def observe_control(**_):
        limits = active_control.get().limits
        assert limits.read_idle_timeout == 95
        assert limits.stream_timeout == 240
        return "done"

    monkeypatch.setattr(api, "run_graph", observe_control)
    assert contextvars.copy_context().run(
        engine.run_turn, "hello", output_fn=lambda *a, **k: None,
    ) == "done"


def test_reasoning_effort_reaches_wire_and_request_override_wins(monkeypatch):
    from pixie_core import llm_client

    sent = []

    def respond(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n')

    monkeypatch.setattr(llm_client, "_open_completion", respond)
    bonsai = LMStudioBackend("http://localhost:1/v1", model="bonsai2-27b")
    list(bonsai.create_chat_completion([]))
    list(bonsai.create_chat_completion([], reasoning_effort="xhigh"))
    other = LMStudioBackend("http://localhost:1/v1", model="gemma-4")
    list(other.create_chat_completion([]))
    assert sent[0]["reasoning_effort"] == "medium"
    assert sent[1]["reasoning_effort"] == "xhigh"
    assert "reasoning_effort" not in sent[2]


def test_invalid_runtime_limits_leave_current_policy_unchanged(tmp_path):
    engine = pixie_core.create_engine({"base_url": "http://localhost:1/v1"}, str(tmp_path))
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            engine.set_stream_timeout(250, value)
        assert engine.context.llm.overall_timeout == 180
        assert engine.context.llm.read_idle_timeout == 30


def test_bonsai_default_is_inherited_by_turn_control(tmp_path, monkeypatch):
    engine = pixie_core.create_engine({
        "base_url": "http://localhost:1/v1", "model": "bonsai2-27b",
    }, str(tmp_path))

    def observe_control(**_):
        assert active_control.get().limits.read_idle_timeout == 120
        return "done"

    monkeypatch.setattr(api, "run_graph", observe_control)
    assert contextvars.copy_context().run(
        engine.run_turn, "hello", output_fn=lambda *a, **k: None,
    ) == "done"
