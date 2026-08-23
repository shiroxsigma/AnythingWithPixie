"""Model-specific inference profile tests."""

from engine import _select_sampling_profile


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
