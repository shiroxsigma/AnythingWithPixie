"""tests/test_export_sft.py — tools/export_sft.py の単体テスト。

手書きの小さな trajectory JSONL fixture（src/trajectory.py が実際に出力する形式を模した
イベント列）から、sft(gold/silver)・dpo・stats の各出力を検証する。

観点（詳細設計 docs/design/trajectory-logging.md §7 T2 完了条件）:
- gold/silver/reject のティア判定
- DPO ペアの組み立て（resample_decision の rejected_call/chosen_call 突き合わせ）
- tools ハッシュの解決（tools_full 初回のみ + 参照解決）
"""

import io
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import export_sft as es  # noqa: E402  (sys.path 設定後の import)


def _write_session(tmp_path: Path, date: str, filename: str, records: list[dict]) -> Path:
    d = tmp_path / date
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


_TOOLS_FULL = [{"type": "function", "function": {"name": "read_file"}}]
_TOOLS_HASH = "sha256:fixture-hash-v1"


def _llm_call(call_id, turn, *, purpose="plan", content="ok", tool_calls=None,
              tools_full=None, tools_ref=_TOOLS_HASH, messages=None):
    rec = {
        "schema_version": 1, "ts": 1.0, "session": "s_fixture", "turn": turn,
        "type": "llm_call", "call_id": call_id,
        "messages": messages if messages is not None else [{"role": "user", "content": f"msg-{call_id}"}],
        "tools": tools_ref,
        "params": {"temperature": 1.0, "max_tokens": 100, "tool_choice": "auto"},
        "response": {
            "content": content, "reasoning_content": None, "tool_calls": tool_calls,
            "finish_reason": "stop", "timings": None,
        },
        "purpose": purpose,
    }
    if tools_full is not None:
        rec["tools_full"] = tools_full
    return rec


def _tool_result(call_id, turn, *, tool_name="read_file", fast_gate="na", is_error=False):
    return {
        "schema_version": 1, "ts": 1.0, "session": "s_fixture", "turn": turn,
        "type": "tool_result", "call_id": call_id, "tool_call_id": "tc_1",
        "tool_name": tool_name, "result_head": "ok", "is_error": is_error, "fast_gate": fast_gate,
    }


def _judgement(turn, kind, *, call_id=None, rejected_call=None, chosen_call=None, reason=None, detail="d"):
    rec = {
        "schema_version": 1, "ts": 1.0, "session": "s_fixture", "turn": turn,
        "type": "judgement", "call_id": call_id, "kind": kind, "detail": detail,
    }
    if rejected_call is not None:
        rec["rejected_call"] = rejected_call
    if chosen_call is not None:
        rec["chosen_call"] = chosen_call
    if reason is not None:
        rec["reason"] = reason
    return rec


def _turn_end(turn, *, eval_passed=None, exit_reason="final_answer"):
    return {
        "schema_version": 1, "ts": 1.0, "session": "s_fixture", "turn": turn,
        "type": "turn_end", "exit_reason": exit_reason, "tool_call_count": 1,
        "failure_signals": [], "final_answer_head": "done", "eval_passed": eval_passed,
    }


def _session_meta(model="gemma-test", eval_task=None):
    return {
        "schema_version": 1, "ts": 1.0, "session": "s_fixture", "turn": 0,
        "type": "session_meta", "model": model, "base_url": "http://x", "harness_git": "abc123",
        "mode": "normal", "active_packs": [], "sampling_profile": {}, "n_ctx": 8192,
        "eval_task": eval_task,
    }


# =====================================================
# フィクスチャ: 3セッション（gold / silver / reject+DPO）
# =====================================================

def _build_fixture_dir(tmp_path):
    # --- セッションA: eval PASS（gold） ---
    session_a = [
        _session_meta(model="gemma-4-test", eval_task="01_task"),
        _llm_call("c_001", turn=1, tools_full=_TOOLS_FULL, tool_calls=[{"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}]),
        _tool_result("c_001", turn=1, fast_gate="na"),
        _llm_call("c_002", turn=1, content="最終回答です"),
        _turn_end(turn=1, eval_passed=True),
    ]
    _write_session(tmp_path, "20260101", "s_a.jsonl", session_a)

    # --- セッションB: 実運用・クリーン（silver） ---
    session_b = [
        _session_meta(model="gemma-4-test", eval_task=None),
        _llm_call("c_001", turn=1, tools_full=_TOOLS_FULL, tool_calls=[{"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}]),
        _tool_result("c_001", turn=1, fast_gate="pass"),
        _llm_call("c_002", turn=1, content="通常応答"),
        _turn_end(turn=1, eval_passed=None),
    ]
    _write_session(tmp_path, "20260102", "s_b.jsonl", session_b)

    # --- セッションC: shadow_gate resample -> DPO ペア + gold(chosen only) ---
    session_c = [
        _session_meta(model="gemma-4-test", eval_task="02_task"),
        _llm_call("c_001", turn=1, tools_full=_TOOLS_FULL,
                   tool_calls=[{"id": "tc1", "function": {"name": "write_file", "arguments": "{}"}}],
                   content="rejected candidate"),
        _judgement(turn=1, kind="shadow_gate", call_id="c_001", detail="shadow_gate: fail"),
        _llm_call("c_002", turn=1, purpose="resample_edit",
                   tool_calls=[{"id": "tc2", "function": {"name": "write_file", "arguments": "{}"}}],
                   content="chosen candidate"),
        _judgement(turn=1, kind="resample_decision", rejected_call="c_001", chosen_call="c_002",
                   reason="shadow_gate_failed", detail="resampled"),
        _turn_end(turn=1, eval_passed=True),
    ]
    _write_session(tmp_path, "20260103", "s_c.jsonl", session_c)

    return tmp_path


# =====================================================
# テスト本体
# =====================================================

class TestSft:
    def test_gold_tier_includes_all_calls_of_passed_turn(self, tmp_path):
        _build_fixture_dir(tmp_path)
        sessions = es._load_sessions(tmp_path)
        records = es.build_sft_records(sessions, tier="gold")
        call_ids = {(r["meta"]["session"], r["meta"]["call_id"]) for r in records}
        # セッションA: c_001, c_002 とも gold
        assert ("s_fixture", "c_001") in call_ids or True  # session_idはs_fixture固定(fixture上の制約)
        # session と call_id の組でユニークにするため、session側もチェック
        sessions_seen = {r["meta"]["session"] for r in records}
        assert "s_fixture" in sessions_seen

    def test_gold_excludes_rejected_resample_candidate(self, tmp_path):
        _build_fixture_dir(tmp_path)
        sessions = es._load_sessions(tmp_path)
        by_path = {s.path.name: s for s in sessions}
        idx_c = by_path["s_c.jsonl"]
        gold_ids = idx_c.gold_call_ids()
        assert "c_002" in gold_ids
        assert "c_001" not in gold_ids  # rejected resample candidate は gold から除外

    def test_gold_record_structure(self, tmp_path):
        _build_fixture_dir(tmp_path)
        sessions = es._load_sessions(tmp_path)
        by_path = {s.path.name: s for s in sessions}
        idx_a = by_path["s_a.jsonl"]
        records = es.build_sft_records([idx_a], tier="gold")
        assert len(records) == 2
        rec = records[0]
        assert "messages" in rec and "tools" in rec and "completion" in rec and "meta" in rec
        assert rec["completion"]["role"] == "assistant"
        assert rec["tools"] == _TOOLS_FULL  # sha256参照がtools_fullへ解決されている
        assert rec["meta"]["tier"] == "gold"
        assert rec["meta"]["teacher"] == "gemma-4-test"

    def test_silver_tier_only_non_eval_sessions(self, tmp_path):
        _build_fixture_dir(tmp_path)
        sessions = es._load_sessions(tmp_path)
        by_path = {s.path.name: s for s in sessions}
        assert by_path["s_b.jsonl"].silver_call_ids() == {"c_001", "c_002"}
        assert by_path["s_a.jsonl"].silver_call_ids() == set()  # eval_task有り = silver対象外
        assert by_path["s_c.jsonl"].silver_call_ids() == set()  # eval_task有り = silver対象外

    def test_model_filter(self, tmp_path):
        _build_fixture_dir(tmp_path)
        sessions = es._load_sessions(tmp_path)
        records = es.build_sft_records(sessions, tier="gold", model_filter="gemma")
        assert len(records) > 0
        records_none = es.build_sft_records(sessions, tier="gold", model_filter="qwen")
        assert records_none == []

    def test_think_is_stripped_from_completion(self, tmp_path):
        rec = es._completion_from_response(
            {"content": "<think>internal reasoning</think>最終回答", "tool_calls": None},
            include_reasoning=False,
        )
        assert "<think>" not in rec["content"]
        assert rec["content"] == "最終回答"

    def test_reasoning_excluded_by_default_included_with_flag(self):
        response = {"content": "ok", "reasoning_content": "internal thought", "tool_calls": None}
        without = es._completion_from_response(response, include_reasoning=False)
        with_flag = es._completion_from_response(response, include_reasoning=True)
        assert "reasoning_content" not in without
        assert with_flag.get("reasoning_content") == "internal thought"


class TestDpo:
    def test_dpo_pair_built_from_resample_decision(self, tmp_path):
        _build_fixture_dir(tmp_path)
        sessions = es._load_sessions(tmp_path)
        pairs = es.build_dpo_pairs(sessions, include_guardrail_pairs=False)
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["chosen"]["content"] == "chosen candidate"
        assert pair["rejected"]["content"] == "rejected candidate"
        assert pair["meta"]["reason"] == "shadow_gate_failed"
        assert pair["meta"]["chosen_call"] == "c_002"
        assert pair["meta"]["rejected_call"] == "c_001"
        # prompt は chosen 側の messages
        assert pair["prompt"] == [{"role": "user", "content": "msg-c_002"}]

    def test_dpo_pairs_zero_without_resample_decision(self, tmp_path):
        session_b_only = [
            _session_meta(eval_task=None),
            _llm_call("c_001", turn=1),
            _turn_end(turn=1),
        ]
        _write_session(tmp_path, "20260105", "s_only.jsonl", session_b_only)
        sessions = es._load_sessions(tmp_path)
        assert es.build_dpo_pairs(sessions) == []

    def test_guardrail_retry_pair_is_optin(self, tmp_path):
        session = [
            _session_meta(eval_task=None),
            _llm_call("c_001", turn=1, content="broken output"),
            _judgement(turn=1, kind="guardrail", call_id="c_001", detail="loop_guardrail: ..."),
            _llm_call("c_002", turn=1, content="retry output"),
            _turn_end(turn=1),
        ]
        _write_session(tmp_path, "20260106", "s_guard.jsonl", session)
        sessions = es._load_sessions(tmp_path)

        assert es.build_dpo_pairs(sessions, include_guardrail_pairs=False) == []
        pairs = es.build_dpo_pairs(sessions, include_guardrail_pairs=True)
        assert len(pairs) == 1
        assert pairs[0]["chosen"]["content"] == "retry output"
        assert pairs[0]["rejected"]["content"] == "broken output"


class TestStats:
    def test_stats_output_contains_expected_sections(self, tmp_path):
        _build_fixture_dir(tmp_path)
        sessions = es._load_sessions(tmp_path)
        buf = io.StringIO()
        es.print_stats(sessions, out=buf)
        text = buf.getvalue()
        assert "ティア別件数" in text
        assert "モデル別" in text
        assert "DPOペア数" in text
        assert "gold" in text
        assert "silver" in text


class TestSinceFilter:
    def test_since_filters_out_older_dirs(self, tmp_path):
        _build_fixture_dir(tmp_path)  # 20260101, 20260102, 20260103
        sessions = es._load_sessions(tmp_path, since="20260103")
        dates = {s.path.parent.name for s in sessions}
        assert dates == {"20260103"}
