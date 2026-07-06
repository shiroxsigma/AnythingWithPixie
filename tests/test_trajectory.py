"""tests/test_trajectory.py — src/trajectory.py (TrajectoryLogger) の単体テスト。

観点（詳細設計 docs/design/trajectory-logging.md §7 T1 完了条件）:
- イベント形式（schema_version・session・turn・type の共通ヘッダ + 各イベント固有フィールド）
- call_id 採番（c_001, c_002, ... のセッション内通番）
- tools のハッシュ参照（初回のみ tools_full を全量記録、以降は sha256 参照のみ）
- GC（TRAJECTORY_MAX_MB 超過分を古い日付ディレクトリから削除）
- 例外安全（内部で例外が起きても呼び出し元に伝播しない）
- TRAJECTORY_LOG_ENABLED=False（enabled=False）で一切書き込まれない
"""

import json
from pathlib import Path

from trajectory import TrajectoryLogger


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _session_file(logger: TrajectoryLogger) -> Path:
    assert logger._session_path is not None
    return logger._session_path


# =====================================================
# 基本: セッションファイルの作成・共通ヘッダ
# =====================================================

class TestBasics:
    def test_creates_session_file_under_date_dir(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        assert logger.enabled is True
        assert logger.session_id is not None
        session_path = _session_file(logger)
        assert session_path.parent.parent == tmp_path
        # 日付ディレクトリ名は YYYYMMDD の8桁
        assert len(session_path.parent.name) == 8
        assert session_path.parent.name.isdigit()

    def test_session_meta_has_common_header_and_fields(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.log_session_meta(
            model="gemma-4-test", base_url="http://localhost:8080/v1",
            mode="normal", active_packs={"manga"}, sampling_profile={"temperature": 1.0},
            n_ctx=8192, eval_task=None,
        )
        records = _read_jsonl(_session_file(logger))
        assert len(records) == 1
        rec = records[0]
        assert rec["schema_version"] == 1
        assert rec["type"] == "session_meta"
        assert rec["session"] == logger.session_id
        assert rec["turn"] == 0
        assert isinstance(rec["ts"], float)
        assert rec["model"] == "gemma-4-test"
        assert rec["base_url"] == "http://localhost:8080/v1"
        assert rec["mode"] == "normal"
        assert rec["active_packs"] == ["manga"]
        assert rec["sampling_profile"] == {"temperature": 1.0}
        assert rec["n_ctx"] == 8192
        assert rec["eval_task"] is None
        # harness_git は git 環境が無くても None を返すだけで例外にならない
        assert "harness_git" in rec

    def test_start_turn_increments_turn_counter(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.start_turn()
        logger.log_llm_call(
            messages=[{"role": "user", "content": "hi"}], tools=None,
            params={"temperature": 0.7}, response={"content": "hello", "tool_calls": None,
                                                      "finish_reason": "stop", "timings": None,
                                                      "reasoning_content": None},
            purpose="plan",
        )
        logger.start_turn()
        logger.log_llm_call(
            messages=[{"role": "user", "content": "hi again"}], tools=None,
            params={"temperature": 0.7}, response={"content": "hello2", "tool_calls": None,
                                                      "finish_reason": "stop", "timings": None,
                                                      "reasoning_content": None},
            purpose="plan",
        )
        records = _read_jsonl(_session_file(logger))
        assert [r["turn"] for r in records] == [1, 2]


# =====================================================
# call_id 採番
# =====================================================

class TestCallIdNumbering:
    def test_sequential_call_ids(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        ids = []
        for i in range(3):
            call_id = logger.log_llm_call(
                messages=[{"role": "user", "content": f"turn{i}"}], tools=None,
                params={}, response={"content": "ok", "tool_calls": None, "finish_reason": "stop",
                                       "timings": None, "reasoning_content": None},
                purpose="plan",
            )
            ids.append(call_id)
        assert ids == ["c_001", "c_002", "c_003"]
        assert logger.last_call_id == "c_003"

    def test_log_llm_call_returns_none_when_disabled(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=False)
        call_id = logger.log_llm_call(
            messages=[], tools=None, params={}, response={}, purpose="plan",
        )
        assert call_id is None
        assert logger.last_call_id is None


# =====================================================
# tools のハッシュ参照
# =====================================================

class TestToolsHashing:
    def test_first_call_includes_tools_full_subsequent_only_ref(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        tools = [{"type": "function", "function": {"name": "read_file"}}]

        logger.log_llm_call(
            messages=[], tools=tools, params={}, response={"content": "a", "tool_calls": None,
                                                              "finish_reason": "stop", "timings": None,
                                                              "reasoning_content": None},
            purpose="plan",
        )
        logger.log_llm_call(
            messages=[], tools=tools, params={}, response={"content": "b", "tool_calls": None,
                                                              "finish_reason": "stop", "timings": None,
                                                              "reasoning_content": None},
            purpose="plan",
        )
        records = _read_jsonl(_session_file(logger))
        assert len(records) == 2
        first, second = records

        assert first["tools"].startswith("sha256:")
        assert "tools_full" in first
        assert first["tools_full"] == tools

        assert second["tools"] == first["tools"]  # 同一ハッシュ
        assert "tools_full" not in second  # 2回目は参照のみ

    def test_no_tools_yields_no_tools_ref(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.log_llm_call(
            messages=[], tools=None, params={}, response={"content": "a", "tool_calls": None,
                                                             "finish_reason": "stop", "timings": None,
                                                             "reasoning_content": None},
            purpose="plan",
        )
        records = _read_jsonl(_session_file(logger))
        assert records[0]["tools"] is None
        assert "tools_full" not in records[0]

    def test_changed_tools_re_emits_tools_full(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        tools_a = [{"type": "function", "function": {"name": "read_file"}}]
        tools_b = [{"type": "function", "function": {"name": "write_file"}}]
        logger.log_llm_call(messages=[], tools=tools_a, params={},
                              response={"content": "a", "tool_calls": None, "finish_reason": "stop",
                                        "timings": None, "reasoning_content": None}, purpose="plan")
        logger.log_llm_call(messages=[], tools=tools_b, params={},
                              response={"content": "b", "tool_calls": None, "finish_reason": "stop",
                                        "timings": None, "reasoning_content": None}, purpose="plan")
        records = _read_jsonl(_session_file(logger))
        assert records[0]["tools"] != records[1]["tools"]
        assert "tools_full" in records[0]
        assert "tools_full" in records[1]


# =====================================================
# tool_result / judgement / turn_end イベント形式
# =====================================================

class TestOtherEvents:
    def test_log_tool_result(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.log_tool_result(
            call_id="c_001", tool_call_id="tc_1", tool_name="read_file",
            result="Success: file content" + ("x" * 1000), is_error=False,
            fast_gate="na",
        )
        rec = _read_jsonl(_session_file(logger))[0]
        assert rec["type"] == "tool_result"
        assert rec["call_id"] == "c_001"
        assert rec["tool_call_id"] == "tc_1"
        assert rec["tool_name"] == "read_file"
        assert len(rec["result_head"]) <= 500
        assert rec["is_error"] is False
        assert rec["fast_gate"] == "na"
        assert "fast_gate_detail" not in rec

    def test_log_tool_result_fail_includes_detail(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.log_tool_result(
            call_id="c_001", tool_call_id="tc_1", tool_name="write_file",
            result="[ruff check] E999 SyntaxError", is_error=False,
            fast_gate="fail", fast_gate_detail="[ruff check] E999 SyntaxError",
        )
        rec = _read_jsonl(_session_file(logger))[0]
        assert rec["fast_gate"] == "fail"
        assert rec["fast_gate_detail"] == "[ruff check] E999 SyntaxError"

    def test_log_judgement_resample_decision(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.log_judgement(
            kind="resample_decision", detail="shadow_gate failed, resampled",
            rejected_call="c_001", chosen_call="c_002", reason="shadow_gate_failed",
        )
        rec = _read_jsonl(_session_file(logger))[0]
        assert rec["type"] == "judgement"
        assert rec["kind"] == "resample_decision"
        assert rec["rejected_call"] == "c_001"
        assert rec["chosen_call"] == "c_002"
        assert rec["reason"] == "shadow_gate_failed"

    def test_log_judgement_guardrail_has_no_pair_fields(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.log_judgement(kind="guardrail", detail="loop_guardrail: ...", call_id="c_003")
        rec = _read_jsonl(_session_file(logger))[0]
        assert rec["kind"] == "guardrail"
        assert rec["call_id"] == "c_003"
        assert "rejected_call" not in rec
        assert "chosen_call" not in rec
        assert "reason" not in rec

    def test_log_turn_end(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.log_turn_end(
            exit_reason="final_answer (ツール実行 3回後)", tool_call_count=3,
            failure_signals=["fast_gate: ..."], final_answer="結論: OK" * 200,
            eval_passed=None,
        )
        rec = _read_jsonl(_session_file(logger))[0]
        assert rec["type"] == "turn_end"
        assert rec["tool_call_count"] == 3
        assert rec["failure_signals"] == ["fast_gate: ..."]
        assert len(rec["final_answer_head"]) <= 500
        assert rec["eval_passed"] is None

    def test_mark_eval_result_rewrites_last_turn_end_only(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        logger.log_turn_end(exit_reason="e1", tool_call_count=1, failure_signals=[],
                              final_answer="a1", eval_passed=None)
        logger.log_llm_call(messages=[], tools=None, params={},
                              response={"content": "x", "tool_calls": None, "finish_reason": "stop",
                                        "timings": None, "reasoning_content": None}, purpose="plan")
        logger.log_turn_end(exit_reason="e2", tool_call_count=2, failure_signals=[],
                              final_answer="a2", eval_passed=None)

        logger.mark_eval_result(True)

        records = _read_jsonl(_session_file(logger))
        turn_ends = [r for r in records if r["type"] == "turn_end"]
        assert turn_ends[0]["eval_passed"] is None  # 1件目は変更されない
        assert turn_ends[1]["eval_passed"] is True  # 最後の turn_end だけ書き換わる


# =====================================================
# GC
# =====================================================

class TestGC:
    def test_gc_removes_oldest_date_dirs_over_limit(self, tmp_path):
        # 1MB を超える古い日付ディレクトリを事前に作っておく
        old_dir = tmp_path / "20200101"
        old_dir.mkdir(parents=True)
        (old_dir / "s_old.jsonl").write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

        new_dir = tmp_path / "20991231"
        new_dir.mkdir(parents=True)
        (new_dir / "s_new.jsonl").write_bytes(b"y" * 1024)

        # max_mb=1 で新規ロガーを作ると、コンストラクタが GC を1回走らせる
        TrajectoryLogger(base_dir=str(tmp_path), enabled=True, max_mb=1)
        assert not old_dir.exists()
        assert new_dir.exists()

    def test_gc_keeps_current_session_dir(self, tmp_path):
        # 現在のセッションが書き込む日付ディレクトリ自体は自己GCで消えないことを確認する。
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True, max_mb=0)
        logger.log_session_meta(model="m")
        assert _session_file(logger).exists()

    def test_gc_noop_when_disabled(self, tmp_path):
        old_dir = tmp_path / "20200101"
        old_dir.mkdir(parents=True)
        (old_dir / "s_old.jsonl").write_bytes(b"x" * (2 * 1024 * 1024))
        TrajectoryLogger(base_dir=str(tmp_path), enabled=False, max_mb=1)
        assert old_dir.exists()  # disabled 時は GC も走らない


# =====================================================
# 例外安全性
# =====================================================

class TestExceptionSafety:
    def test_log_llm_call_survives_unserializable_messages(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)

        class Unserializable:
            def __repr__(self):
                raise RuntimeError("boom")

        # json.dumps に失敗するオブジェクトを messages に紛れ込ませても例外を送出しない。
        call_id = logger.log_llm_call(
            messages=[{"role": "user", "content": Unserializable()}],
            tools=None, params={}, response={"content": "x"}, purpose="plan",
        )
        # 記録には失敗するが、呼び出し元には例外が伝播しない（call_id は None でもよい）
        assert call_id is None or isinstance(call_id, str)

    def test_all_public_methods_never_raise_when_enabled(self, tmp_path):
        logger = TrajectoryLogger(base_dir=str(tmp_path), enabled=True)
        # session_path を意図的に壊れたパスにして書き込み失敗を誘発する
        logger._session_path = Path("Z:\\definitely\\not\\a\\real\\path\\session.jsonl")

        logger.start_turn()
        logger.log_session_meta(model="m")
        logger.log_llm_call(messages=[], tools=None, params={}, response={}, purpose="plan")
        logger.log_tool_result(call_id="c_001", tool_call_id="t1", tool_name="read_file", result="ok")
        logger.log_judgement(kind="guardrail", detail="d")
        logger.log_turn_end(exit_reason="e", tool_call_count=0, failure_signals=[], final_answer="")
        logger.mark_eval_result(True)
        # ここまで例外が飛ばずに到達すれば成功

    def test_constructor_survives_unwritable_base_dir(self, tmp_path):
        # 既存ファイルをディレクトリのつもりで渡す（mkdir が必ず失敗する状況を模す）
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        logger = TrajectoryLogger(base_dir=str(blocker / "trajectories"), enabled=True)
        assert logger.enabled is False  # 初期化失敗時は自動的に無効化される
        # 無効化後もメソッド呼び出しは例外を出さない
        logger.log_session_meta(model="m")
        assert logger.log_llm_call(messages=[], tools=None, params={}, response={}) is None


# =====================================================
# ENABLED=False で完全無効化
# =====================================================

class TestDisabled:
    def test_disabled_creates_no_directory(self, tmp_path):
        base = tmp_path / "trajectories"
        logger = TrajectoryLogger(base_dir=str(base), enabled=False)
        logger.start_turn()
        logger.log_session_meta(model="m")
        logger.log_llm_call(messages=[{"role": "user", "content": "hi"}], tools=None,
                              params={}, response={"content": "x"})
        logger.log_tool_result(call_id="c_001", tool_call_id="t1", tool_name="read_file", result="ok")
        logger.log_judgement(kind="guardrail", detail="d")
        logger.log_turn_end(exit_reason="e", tool_call_count=0, failure_signals=[], final_answer="")

        assert not base.exists()
        assert logger.session_id is None
        assert logger._session_path is None

    def test_disabled_by_default_config_is_respected(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "TRAJECTORY_LOG_ENABLED", False)
        logger = TrajectoryLogger(base_dir=str(tmp_path))
        assert logger.enabled is False
