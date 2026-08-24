"""Public API 1.11 contracts for profiles, events, policies, and metrics."""

import pytest

import pixie_core

_SERVER = {"base_url": "http://localhost:1/v1", "model": "test-model"}


def _engine(tmp_path, **kwargs):
    return pixie_core.create_engine(_SERVER, str(tmp_path), **kwargs)


def test_api11_types_are_public():
    assert pixie_core.API_VERSION == "1.11"
    assert pixie_core.AgentProfile.__module__ == "pixie_core._api"
    assert pixie_core.ContextPolicy.__module__ == "pixie_core._api"
    assert pixie_core.EngineEvent.__module__ == "pixie_core._api"


def test_profile_normalizes_collections_and_applies_to_engine(tmp_path):
    profile = pixie_core.AgentProfile(
        name="note",
        tool_set={"read_file", "grep_search"},
        system_suffix="NOTE MODE",
        active_packs={"copilot"},
        context_policy={
            "context_length": 8192,
            "overall_timeout": 91,
            "read_idle_timeout": 17,
        },
    )

    engine = _engine(tmp_path, profile=profile)

    assert profile.tool_set == frozenset({"read_file", "grep_search"})
    assert profile.active_packs == frozenset({"copilot"})
    assert engine.profile is profile
    assert engine.context.fixed_tool_set == profile.tool_set
    assert engine.context.active_packs == {"copilot"}
    assert engine.context.llm._n_ctx == 8192
    assert engine.context.llm.overall_timeout == 91.0
    assert engine.context.llm.read_idle_timeout == 17.0


@pytest.mark.parametrize(
    "values",
    [
        {"context_length": 0},
        {"overall_timeout": -1},
        {"read_idle_timeout": 0},
    ],
)
def test_context_policy_rejects_non_positive_values(values):
    with pytest.raises(ValueError):
        pixie_core.ContextPolicy(**values)


def test_set_context_policy_accepts_mapping(tmp_path):
    engine = _engine(tmp_path)
    applied = engine.set_context_policy(
        {"context_length": 4096, "overall_timeout": 30}
    )
    assert applied == {"context_length": 4096, "overall_timeout": 30}
    assert engine.context.llm._n_ctx == 4096
    assert engine.context.llm.overall_timeout == 30.0


def test_set_profile_replaces_runtime_tools_packs_and_suffix(tmp_path):
    engine = _engine(tmp_path)
    profile = engine.set_profile({
        "name": "plan",
        "tool_set": {"read_file"},
        "active_packs": {"copilot"},
        "system_suffix": "PLAN MODE",
    })

    assert profile is engine.profile
    assert engine.context.fixed_tool_set == frozenset({"read_file"})
    assert engine.context.active_packs == {"copilot"}
    assert engine._system_suffix == "PLAN MODE"


def test_turn_metrics_are_defensive_copies(tmp_path):
    engine = _engine(tmp_path)
    engine.state.llm_call_metrics = [{"decode_tokens": 12}]
    engine.state.tool_call_count = 3
    engine.state.exit_reason = "completed"
    engine.state.acceptance_conditions = [{"text": "tests pass"}]
    engine.state.acceptance_retry_count = 1

    metrics = engine.get_turn_metrics()
    metrics["llm_calls"][0]["decode_tokens"] = 999
    metrics["acceptance_conditions"][0]["text"] = "changed"

    assert engine.state.llm_call_metrics[0]["decode_tokens"] == 12
    assert engine.state.acceptance_conditions[0]["text"] == "tests pass"
    assert metrics["tool_calls"] == 3
    assert metrics["exit_reason"] == "completed"
    assert metrics["acceptance_retries"] == 1


def test_run_turn_events_wraps_legacy_output_and_completion(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    def fake_run_turn(user_text, *, output_fn, interactive_fn, show_thinking):
        assert user_text == "hello"
        assert interactive_fn is None
        assert show_thinking is True
        output_fn("working", end="\n", flush=True)
        engine.state.tool_call_count = 2
        engine.state.exit_reason = "completed"
        return "done"

    monkeypatch.setattr(engine, "run_turn", fake_run_turn)
    events = []

    result = engine.run_turn_events(
        "hello", event_fn=events.append, show_thinking=True
    )

    assert result == "done"
    assert [event.type for event in events] == [
        "turn_started",
        "output",
        "turn_completed",
    ]
    assert events[1].as_dict() == {
        "type": "output",
        "text": "working",
        "end": "\n",
        "flush": True,
    }
    assert events[2].as_dict()["metrics"]["tool_calls"] == 2
    assert events[2].as_dict()["metrics"]["exit_reason"] == "completed"


def test_run_turn_events_emits_error_before_reraising(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "run_turn", fail)
    events = []

    with pytest.raises(RuntimeError, match="boom"):
        engine.run_turn_events("hello", event_fn=events.append)

    assert [event.type for event in events] == ["turn_started", "turn_error"]
    assert events[-1].text == "RuntimeError: boom"
