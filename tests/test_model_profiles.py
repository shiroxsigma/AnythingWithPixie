"""Model-specific inference profile tests."""

from engine import _reasoning_budget_kwargs, _select_sampling_profile


def test_lfm25_26b_uses_official_generation_settings():
    expected = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1}

    assert _select_sampling_profile("LiquidAI/LFM2.5-2.6B") == expected
    assert _select_sampling_profile("lfm2.5-2.6b-q4_k_m.gguf") == expected


def test_other_lfm_models_keep_generic_profile():
    assert _select_sampling_profile("LFM2.5-8B-A1B") == {
        "temperature": 0.2,
        "top_k": 80,
        "repeat_penalty": 1.05,
    }


def test_gemma_shallow_turns_have_a_request_level_reasoning_budget():
    assert _select_sampling_profile("gemma-4") == {
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "shallow_reasoning_budget_tokens": 1536,
    }

    profile = _select_sampling_profile("gemma-4")
    assert _reasoning_budget_kwargs(profile, "shallow") == {
        "thinking_budget_tokens": 1536,
    }
    assert _reasoning_budget_kwargs(profile, "deep") == {}
