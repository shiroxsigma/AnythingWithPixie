"""
AnythingPixie — 軌跡ロギング（SFT/DPO 教師データ産出基盤）

詳細設計: docs/design/trajectory-logging.md

TrajectoryLogger はセッション単位の JSONL に、LLM呼び出し・ツール結果・品質判定・
ターン終了イベントを同期 append する。**記録は本体の動作に一切影響してはならない**
（全メソッドを try/except で保護し、例外は握り潰す。呼び出し元には常に正常復帰する）。
enabled=False（既定は config.TRAJECTORY_LOG_ENABLED）の場合、各メソッドは冒頭で
即 return し、オーバーヘッドはほぼゼロになる。

インスタンスは AppContext.trajectory に保持する想定（lessons.py の
get_lesson_store() のようなプロセス内シングルトンにはしない）。理由:
eval の harvest モードはタスクごとに独立セッション（独立ファイル）が必要なため、
「セッション = インスタンス」の対応をそのまま保てるコンテキスト所有方式の方が単純。

依存: paths のみ（標準ライブラリ + プロジェクト最下層モジュール）。config は遅延 import
（config.py が本モジュールに依存しないようにするため。lessons.py と同じ方針）。
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

from paths import get_app_root, get_project_data_path

#: レコード共通ヘッダの schema_version。イベント型を増やす/形式を変える際にここを上げる。
SCHEMA_VERSION = 1


def _get_config_defaults() -> tuple[bool, int, int]:
    """config.py から既定値を読む。config.py が読めない/壊れている場合の安全弁付き。"""
    try:
        from config import (
            TRAJECTORY_LOG_ENABLED,
            TRAJECTORY_MAX_MB,
            TRAJECTORY_RESULT_HEAD_CHARS,
        )
        return bool(TRAJECTORY_LOG_ENABLED), int(TRAJECTORY_MAX_MB), int(TRAJECTORY_RESULT_HEAD_CHARS)
    except Exception:
        return True, 2048, 500


def _git_short_hash() -> str | None:
    """git rev-parse --short HEAD を取得する。失敗時（git未インストール・非gitリポジトリ等）は None。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=get_app_root(), capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            h = proc.stdout.strip()
            return h or None
    except Exception:
        pass
    return None


class TrajectoryLogger:
    """1セッション分の軌跡ログを JSONL へ同期 append するロガー。

    ファイルレイアウト: <base_dir>/<YYYYMMDD>/<session_id>.jsonl
    （詳細設計 §2.1）。1セッション = 1インスタンス = 1ファイル。
    """

    def __init__(
        self,
        base_dir: str | None = None,
        enabled: bool | None = None,
        max_mb: int | None = None,
        result_head_chars: int | None = None,
        session_suffix: str = "",
    ):
        _def_enabled, _def_max_mb, _def_head = _get_config_defaults()
        self.enabled = _def_enabled if enabled is None else bool(enabled)
        self.max_mb = max_mb if max_mb is not None else _def_max_mb
        self.result_head_chars = result_head_chars if result_head_chars is not None else _def_head

        # 以下は enabled=False でも参照されうる（last_call_id 等）ため、常に初期化しておく。
        self._call_counter = 0
        self.last_call_id: str | None = None
        self._tools_hash: str | None = None
        self._turn = 0
        self.base_dir: Path | None = None
        self.session_id: str | None = None
        self._session_path: Path | None = None

        if not self.enabled:
            return

        try:
            self.base_dir = Path(base_dir) if base_dir else Path(get_project_data_path(".pixie_notes/trajectories"))
            ts = time.strftime("%Y%m%d_%H%M%S")
            rand_hex = f"{random.randrange(16 ** 4):04x}"
            self.session_id = f"s_{ts}_{rand_hex}{session_suffix}"
            date_dir_name = time.strftime("%Y%m%d")
            date_dir = self.base_dir / date_dir_name
            date_dir.mkdir(parents=True, exist_ok=True)
            self._session_path = date_dir / f"{self.session_id}.jsonl"
        except Exception:
            # ディレクトリ作成に失敗した場合は記録機能全体を無効化する（本体には影響させない）。
            self.enabled = False
            self.base_dir = None
            self.session_id = None
            self._session_path = None
            return

        try:
            self._gc()
        except Exception:
            pass

    # =====================================================
    # 内部ヘルパー
    # =====================================================

    def _write(self, record: dict) -> None:
        if not self.enabled or not self._session_path:
            return
        record.setdefault("schema_version", SCHEMA_VERSION)
        record.setdefault("ts", time.time())
        record.setdefault("session", self.session_id)
        record.setdefault("turn", self._turn)
        with open(self._session_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _next_call_id(self) -> str:
        self._call_counter += 1
        return f"c_{self._call_counter:03d}"

    # =====================================================
    # ターン管理
    # =====================================================

    def start_turn(self) -> None:
        """run_graph 呼び出し（= 1ユーザーターン）の開始時に呼ぶ。turn 番号をインクリメントする。"""
        if not self.enabled:
            return
        try:
            self._turn += 1
        except Exception:
            pass

    # =====================================================
    # イベント記録
    # =====================================================

    def log_session_meta(
        self,
        *,
        model: str = "",
        base_url: str = "",
        mode: str = "normal",
        active_packs=None,
        sampling_profile: dict | None = None,
        n_ctx=None,
        eval_task: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            self._write({
                "type": "session_meta",
                "turn": 0,
                "model": model,
                "base_url": base_url,
                "harness_git": _git_short_hash(),
                "mode": mode,
                "active_packs": sorted(active_packs) if active_packs else [],
                "sampling_profile": sampling_profile or {},
                "n_ctx": n_ctx,
                "eval_task": eval_task,
            })
        except Exception:
            pass

    def log_llm_call(self, *, messages, tools, params: dict, response: dict, purpose: str = "plan") -> str | None:
        """LLM呼び出し1回（node_plan の応答確定 or reflection の1回）を記録する。

        ツール定義はセッション内不変が前提のため、初回のみ tools_full（全量）を追加で書き、
        以降は sha256 参照のみにする（詳細設計 §2.2 の tools/tools_full 方式）。

        Returns:
            採番した call_id。記録に失敗した場合は None（last_call_id は更新されない）。
        """
        if not self.enabled:
            return None
        try:
            call_id = self._next_call_id()
            tools_ref = None
            tools_full = None
            if tools:
                try:
                    tools_json = json.dumps(tools, sort_keys=True, ensure_ascii=False)
                except Exception:
                    tools_json = repr(tools)
                digest = hashlib.sha256(tools_json.encode("utf-8", errors="ignore")).hexdigest()
                tools_ref = f"sha256:{digest}"
                if digest != self._tools_hash:
                    tools_full = tools
                    self._tools_hash = digest

            record = {
                "type": "llm_call",
                "call_id": call_id,
                "messages": messages,
                "tools": tools_ref,
                "params": params or {},
                "response": response or {},
                "purpose": purpose,
            }
            if tools_full is not None:
                record["tools_full"] = tools_full
            self._write(record)
            self.last_call_id = call_id
            return call_id
        except Exception:
            return None

    def log_tool_result(
        self,
        *,
        call_id: str | None,
        tool_call_id: str,
        tool_name: str,
        result: str,
        is_error: bool = False,
        fast_gate: str = "na",
        fast_gate_detail: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            head = (result or "")[: self.result_head_chars]
            record = {
                "type": "tool_result",
                "call_id": call_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result_head": head,
                "is_error": bool(is_error),
                "fast_gate": fast_gate,
            }
            if fast_gate == "fail" and fast_gate_detail:
                record["fast_gate_detail"] = fast_gate_detail[:300]
            self._write(record)
        except Exception:
            pass

    def log_judgement(
        self,
        *,
        kind: str,
        detail: str,
        call_id: str | None = None,
        rejected_call: str | None = None,
        chosen_call: str | None = None,
        reason: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            record = {
                "type": "judgement",
                "call_id": call_id,
                "kind": kind,
                "detail": detail,
            }
            if rejected_call is not None:
                record["rejected_call"] = rejected_call
            if chosen_call is not None:
                record["chosen_call"] = chosen_call
            if reason is not None:
                record["reason"] = reason
            self._write(record)
        except Exception:
            pass

    def log_turn_end(
        self,
        *,
        exit_reason: str,
        tool_call_count: int,
        failure_signals,
        final_answer: str,
        eval_passed: bool | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            head = (final_answer or "")[:500]
            self._write({
                "type": "turn_end",
                "exit_reason": exit_reason,
                "tool_call_count": tool_call_count,
                "failure_signals": list(failure_signals or []),
                "final_answer_head": head,
                "eval_passed": eval_passed,
            })
        except Exception:
            pass

    def mark_eval_result(self, passed: bool) -> None:
        """harvest モード専用: セッションファイル中、直近の turn_end レコードの eval_passed を
        事後的に上書きする。

        run_graph 実行完了時点では eval のチェッカー判定がまだ出ていないため、turn_end 記録時は
        eval_passed=null のまま書く。checker 実行後（run_single_task 側）にこのメソッドを呼び、
        該当セッションファイル内の最後の turn_end 行だけを書き換える。
        """
        if not self.enabled or not self._session_path:
            return
        try:
            if not self._session_path.exists():
                return
            lines = self._session_path.read_text(encoding="utf-8").splitlines()
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") == "turn_end":
                    rec["eval_passed"] = bool(passed)
                    lines[i] = json.dumps(rec, ensure_ascii=False)
                    break
            self._session_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    # =====================================================
    # GC（起動時に1回・TRAJECTORY_MAX_MB 超過分を古い日付ディレクトリから削除）
    # =====================================================

    def _gc(self) -> None:
        if not self.enabled or not self.base_dir or not self.base_dir.exists():
            return
        limit_bytes = self.max_mb * 1024 * 1024

        date_dirs = sorted((d for d in self.base_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
        sizes: dict[Path, int] = {}
        total = 0
        for d in date_dirs:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            sizes[d] = size
            total += size

        if total <= limit_bytes:
            return

        for d in date_dirs:
            if total <= limit_bytes:
                break
            # 現在書き込み中のセッションファイルを含むディレクトリは削除しない。
            if self._session_path and d == self._session_path.parent:
                continue
            try:
                shutil.rmtree(d)
                total -= sizes.get(d, 0)
            except Exception:
                continue
