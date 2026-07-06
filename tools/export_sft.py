"""AnythingPixie — SFT/DPO 教師データ export ツール。

詳細設計: docs/design/trajectory-logging.md §4

.pixie_notes/trajectories/**/*.jsonl の生ログ（軌跡・src/trajectory.py が記録）を読み、
SFT向け(sft)・DPO向け(dpo)の学習データJSONL、または統計(stats)を出力する。
LLM不使用・標準ライブラリのみ（evals/ と同様にリポジトリ直下に独立配置。src/ には依存しない）。

使い方:
    python tools/export_sft.py sft --tier gold --out dataset/sft_gold.jsonl
    python tools/export_sft.py sft --tier silver --model-filter gemma --out dataset/sft_silver.jsonl
    python tools/export_sft.py dpo --out dataset/dpo_pairs.jsonl
    python tools/export_sft.py dpo --include-guardrail-pairs --out dataset/dpo_pairs_all.jsonl
    python tools/export_sft.py stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJ_DIR = REPO_ROOT / ".pixie_notes" / "trajectories"

#: reject/silver 判定に使う「悪い判定」の judgement.kind 集合（fast_gate はここに含めない。
#: fast_gate は tool_result イベント側の専用フィールドで判定するため）。
_BAD_JUDGEMENT_KINDS = ("guardrail", "shadow_gate")

#: reject 専用の judgement.kind（DPO の rejected 素材化・SFT からの除外対象）。
_REJECT_JUDGEMENT_KINDS = ("guardrail", "shadow_gate", "parse_rescue")

#: SFT export の対象から常に除外する llm_call.purpose（reflection は教訓抽出専用の
#: 別タスクであり、「ユーザー要求に応答する」という主タスクの教師データとして不適切）。
_EXCLUDED_PURPOSES = ("reflection",)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str | None) -> str:
    """content に inline 混入した <think>...</think> を除去する（詳細設計 §4.3）。"""
    if not text:
        return text or ""
    return _THINK_RE.sub("", text).strip()


# =====================================================
# セッションファイルの読込・索引化
# =====================================================

def _iter_session_files(traj_dir: Path, since: str | None = None):
    if not traj_dir.exists():
        return
    for date_dir in sorted(p for p in traj_dir.iterdir() if p.is_dir()):
        if since and date_dir.name < since:
            continue
        yield from sorted(date_dir.glob("*.jsonl"))


def _load_session_records(path: Path) -> list[dict]:
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return records


class SessionIndex:
    """1セッション分のレコードを解析し、tools解決・ティア判定に必要な索引を構築する。"""

    def __init__(self, path: Path, records: list[dict]):
        self.path = path
        self.records = records
        self.session_meta = next((r for r in records if r.get("type") == "session_meta"), None)
        self.model = (self.session_meta or {}).get("model") or ""
        self.eval_task = (self.session_meta or {}).get("eval_task")
        self.session_id = (self.session_meta or {}).get("session") or path.stem

        self._tools_by_hash: dict[str, list] = {}
        # call_id -> llm_call レコード（_resolved_tools / _record_index を追加した複製）
        self.llm_calls: dict[str, dict] = {}
        self.llm_call_order: list[str] = []  # 出現順の call_id 一覧
        self.tool_results_by_turn: dict[int, list[dict]] = {}
        self.judgements_by_turn: dict[int, list[dict]] = {}
        self.judgements_order: list[tuple[int, dict]] = []  # (record_index, judgement)
        self.turn_end_by_turn: dict[int, dict] = {}

        for idx, rec in enumerate(records):
            rtype = rec.get("type")
            turn = rec.get("turn")
            if rtype == "llm_call":
                tools_ref = rec.get("tools")
                tools_full = rec.get("tools_full")
                resolved_tools = None
                if tools_full is not None:
                    if tools_ref:
                        self._tools_by_hash[tools_ref] = tools_full
                    resolved_tools = tools_full
                elif tools_ref:
                    resolved_tools = self._tools_by_hash.get(tools_ref)
                rec = dict(rec)
                rec["_resolved_tools"] = resolved_tools
                rec["_record_index"] = idx
                call_id = rec.get("call_id")
                if call_id:
                    self.llm_calls[call_id] = rec
                    self.llm_call_order.append(call_id)
            elif rtype == "tool_result":
                self.tool_results_by_turn.setdefault(turn, []).append(rec)
            elif rtype == "judgement":
                self.judgements_by_turn.setdefault(turn, []).append(rec)
                self.judgements_order.append((idx, rec))
            elif rtype == "turn_end":
                self.turn_end_by_turn[turn] = rec

    # ---- ティア判定（詳細設計 §4.1） ----

    def reject_call_ids(self) -> set[str]:
        """DPOのrejected素材 / SFTからの除外対象となる call_id 集合。

        guardrail・shadow_gate・parse_rescue が紐付く call、および resample_decision で
        「採用されなかった側(rejected_call)」を含む（best-of-2 の敗者はSFT正例に使わない）。
        """
        ids: set[str] = set()
        for turn_j in self.judgements_by_turn.values():
            for j in turn_j:
                if j.get("kind") in _REJECT_JUDGEMENT_KINDS and j.get("call_id"):
                    ids.add(j["call_id"])
                if j.get("kind") == "resample_decision" and j.get("rejected_call"):
                    ids.add(j["rejected_call"])
        return ids

    def _turn_of(self, call_id: str):
        return self.llm_calls[call_id].get("turn")

    def gold_call_ids(self) -> set[str]:
        """eval PASS ターンの全 llm_call（reject 側・reflection purposeを除く）。"""
        ids: set[str] = set()
        reject_ids = self.reject_call_ids()
        for turn, te in self.turn_end_by_turn.items():
            if te.get("eval_passed") is not True:
                continue
            for call_id, rec in self.llm_calls.items():
                if rec.get("turn") != turn:
                    continue
                if call_id in reject_ids or rec.get("purpose") in _EXCLUDED_PURPOSES:
                    continue
                ids.add(call_id)
        return ids

    def silver_call_ids(self) -> set[str]:
        """実運用セッション（eval_task が null）で、そのターンに悪い判定が一切無い llm_call。"""
        ids: set[str] = set()
        if self.eval_task:  # silver は実運用セッションのみ対象（詳細設計 §4.1）
            return ids
        reject_ids = self.reject_call_ids()
        for turn in self.turn_end_by_turn:
            judgements = self.judgements_by_turn.get(turn, [])
            if any(j.get("kind") in _BAD_JUDGEMENT_KINDS for j in judgements):
                continue
            tool_results = self.tool_results_by_turn.get(turn, [])
            if any(tr.get("fast_gate") == "fail" for tr in tool_results):
                continue
            for call_id, rec in self.llm_calls.items():
                if rec.get("turn") != turn:
                    continue
                if call_id in reject_ids or rec.get("purpose") in _EXCLUDED_PURPOSES:
                    continue
                ids.add(call_id)
        return ids


def _load_sessions(traj_dir: Path, since: str | None = None) -> list[SessionIndex]:
    sessions = []
    for path in _iter_session_files(traj_dir, since=since):
        records = _load_session_records(path)
        if not records:
            continue
        sessions.append(SessionIndex(path, records))
    return sessions


# =====================================================
# SFT export（詳細設計 §4.1）
# =====================================================

def _completion_from_response(response: dict, include_reasoning: bool) -> dict:
    response = response or {}
    completion = {
        "role": "assistant",
        "content": _strip_think(response.get("content")),
        "tool_calls": response.get("tool_calls"),
    }
    if include_reasoning and response.get("reasoning_content"):
        completion["reasoning_content"] = response["reasoning_content"]
    return completion


def build_sft_records(sessions: list[SessionIndex], tier: str, model_filter: str | None = None,
                       include_reasoning: bool = False) -> list[dict]:
    if tier not in ("gold", "silver"):
        raise ValueError(f"unknown tier: {tier!r} (expected 'gold' or 'silver')")

    out = []
    for idx in sessions:
        if model_filter and model_filter.lower() not in (idx.model or "").lower():
            continue
        call_ids = idx.gold_call_ids() if tier == "gold" else idx.silver_call_ids()
        ordered = sorted(call_ids, key=lambda cid: idx.llm_calls[cid]["_record_index"])
        for call_id in ordered:
            rec = idx.llm_calls[call_id]
            out.append({
                "messages": rec.get("messages"),
                "tools": rec.get("_resolved_tools"),
                "completion": _completion_from_response(rec.get("response") or {}, include_reasoning),
                "meta": {
                    "session": idx.session_id,
                    "call_id": call_id,
                    "tier": tier,
                    "teacher": idx.model,
                    "purpose": rec.get("purpose"),
                },
            })
    return out


# =====================================================
# DPO export（詳細設計 §4.2）
# =====================================================

def _build_pair(idx: SessionIndex, chosen_rec: dict, rejected_rec: dict, reason: str | None) -> dict:
    chosen_resp = chosen_rec.get("response") or {}
    rejected_resp = rejected_rec.get("response") or {}
    return {
        # prompt は「採用された側(chosen)」の messages を正とする。shadow_gate 経由の
        # 再サンプル(resample_edit)は、chosen 側にのみ一時フィードバックメッセージが
        # 追加された状態で生成されているため rejected とは厳密には1メッセージ分ずれるが、
        # 「実際に採用された経路のコンテキスト」を正とするのが一貫している
        # （詳細設計 §4.2: ①②は同一コンテキスト保証 = 採用可否の判断はこのプロンプト基準）。
        "prompt": chosen_rec.get("messages"),
        "chosen": {
            "content": _strip_think(chosen_resp.get("content")),
            "tool_calls": chosen_resp.get("tool_calls"),
        },
        "rejected": {
            "content": _strip_think(rejected_resp.get("content")),
            "tool_calls": rejected_resp.get("tool_calls"),
        },
        "meta": {
            "reason": reason,
            "session": idx.session_id,
            "chosen_call": chosen_rec.get("call_id"),
            "rejected_call": rejected_rec.get("call_id"),
        },
    }


def _guardrail_retry_pairs(idx: SessionIndex) -> list[dict]:
    """③ガードレール発火→再試行成功 の DPO ペア（--include-guardrail-pairs オプトイン専用）。

    judgement(kind=guardrail) 発生直後、同一ターン内で次に出現する llm_call を
    「再試行(chosen)」とみなす。プロンプトが1メッセージ分ずれるため既定では含めない。
    """
    pairs = []
    for rec_idx, j in sorted(idx.judgements_order, key=lambda t: t[0]):
        if j.get("kind") != "guardrail":
            continue
        offending_call_id = j.get("call_id")
        if not offending_call_id or offending_call_id not in idx.llm_calls:
            continue
        offending_rec = idx.llm_calls[offending_call_id]
        turn = offending_rec.get("turn")
        candidates = [
            cid for cid in idx.llm_call_order
            if idx.llm_calls[cid].get("turn") == turn and idx.llm_calls[cid]["_record_index"] > rec_idx
        ]
        if not candidates:
            continue
        retry_rec = idx.llm_calls[candidates[0]]
        detail = (j.get("detail") or "")[:60]
        pairs.append(_build_pair(idx, retry_rec, offending_rec, reason=f"guardrail_retry:{detail}"))
    return pairs


def build_dpo_pairs(sessions: list[SessionIndex], include_guardrail_pairs: bool = False) -> list[dict]:
    pairs = []
    for idx in sessions:
        for turn_j in idx.judgements_by_turn.values():
            for j in sorted(turn_j, key=lambda r: r.get("ts", 0)):
                if j.get("kind") != "resample_decision":
                    continue
                rejected_id = j.get("rejected_call")
                chosen_id = j.get("chosen_call")
                if not rejected_id or not chosen_id:
                    continue
                rejected_rec = idx.llm_calls.get(rejected_id)
                chosen_rec = idx.llm_calls.get(chosen_id)
                if not rejected_rec or not chosen_rec:
                    continue
                pairs.append(_build_pair(idx, chosen_rec, rejected_rec, reason=j.get("reason")))
        if include_guardrail_pairs:
            pairs.extend(_guardrail_retry_pairs(idx))
    return pairs


# =====================================================
# stats（詳細設計 §4.4）
# =====================================================

def _estimate_tokens(obj) -> int:
    """厳密なトークナイザは使わず、文字数/4 の粗い概算（学習に足りるかの目安用）。"""
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except Exception:
        text = str(obj)
    return len(text) // 4


def print_stats(sessions: list[SessionIndex], out=sys.stdout) -> None:
    tier_counts = {"gold": 0, "silver": 0, "reject": 0}
    model_call_counts: dict[str, int] = {}
    total_llm_calls = 0
    est_tokens = 0

    for idx in sessions:
        gold_ids = idx.gold_call_ids()
        silver_ids = idx.silver_call_ids()
        reject_ids = idx.reject_call_ids()
        tier_counts["gold"] += len(gold_ids)
        tier_counts["silver"] += len(silver_ids)
        tier_counts["reject"] += len(reject_ids)
        model_call_counts[idx.model] = model_call_counts.get(idx.model, 0) + len(idx.llm_calls)
        total_llm_calls += len(idx.llm_calls)
        for rec in idx.llm_calls.values():
            est_tokens += _estimate_tokens(rec.get("messages"))

    dpo_pairs = build_dpo_pairs(sessions, include_guardrail_pairs=False)
    dpo_pairs_with_guardrail = build_dpo_pairs(sessions, include_guardrail_pairs=True)

    print(f"セッション数: {len(sessions)}", file=out)
    print(f"llm_call 総数: {total_llm_calls}", file=out)
    print(f"推定トークン数(messages合計, 粗い概算): {est_tokens:,}", file=out)
    print("", file=out)
    print("[ティア別件数]", file=out)
    print(f"  gold   : {tier_counts['gold']}", file=out)
    print(f"  silver : {tier_counts['silver']}", file=out)
    print(f"  reject : {tier_counts['reject']}", file=out)
    print("", file=out)
    print("[モデル別 llm_call 件数]", file=out)
    for model, cnt in sorted(model_call_counts.items(), key=lambda t: -t[1]):
        print(f"  {model or '(不明)'}: {cnt}", file=out)
    print("", file=out)
    print("[DPOペア数]", file=out)
    print(f"  既定（①②のみ）        : {len(dpo_pairs)}", file=out)
    print(f"  --include-guardrail-pairs込み: {len(dpo_pairs_with_guardrail)}", file=out)
    print("", file=out)
    print("[目安] SFT: 500〜5000サンプル / DPO: 200〜1000ペア", file=out)


# =====================================================
# CLI
# =====================================================

def _write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="AnythingPixie SFT/DPO 教師データ export ツール")
    parser.add_argument("--traj-dir", default=str(DEFAULT_TRAJ_DIR),
                         help=f"軌跡ログのルートディレクトリ（既定: {DEFAULT_TRAJ_DIR}）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sft = sub.add_parser("sft", help="SFT向け正例データを export する")
    p_sft.add_argument("--model-filter", help="session_meta.model の部分一致フィルタ（例: gemma）")
    p_sft.add_argument("--tier", choices=["gold", "silver"], default="gold")
    p_sft.add_argument("--since", help="この日付(YYYYMMDD)以降のセッションのみ対象")
    p_sft.add_argument("--out", required=True, help="出力JSONLパス")
    p_sft.add_argument("--include-reasoning", action="store_true",
                        help="reasoning_content を completion に含める（既定は除外）")

    p_dpo = sub.add_parser("dpo", help="DPO向け preference pair を export する")
    p_dpo.add_argument("--since", help="この日付(YYYYMMDD)以降のセッションのみ対象")
    p_dpo.add_argument("--out", required=True, help="出力JSONLパス")
    p_dpo.add_argument("--include-guardrail-pairs", action="store_true",
                        help="ガードレール発火→再試行成功のペアも含める（プロンプトが1メッセージ分"
                             "ずれるためオプトイン。既定はshadow_verify再サンプル/final answer best-of-2のみ）")

    sub.add_parser("stats", help="ティア別件数・DPOペア数等の統計を表示する")

    args = parser.parse_args()
    traj_dir = Path(args.traj_dir)

    if args.command == "sft":
        sessions = _load_sessions(traj_dir, since=args.since)
        records = build_sft_records(
            sessions, tier=args.tier, model_filter=args.model_filter,
            include_reasoning=args.include_reasoning,
        )
        _write_jsonl(records, Path(args.out))
        print(f"[export_sft] {len(records)} 件を {args.out} に書き出しました（tier={args.tier}）。")
        return 0

    if args.command == "dpo":
        sessions = _load_sessions(traj_dir, since=args.since)
        pairs = build_dpo_pairs(sessions, include_guardrail_pairs=args.include_guardrail_pairs)
        _write_jsonl(pairs, Path(args.out))
        print(f"[export_sft] {len(pairs)} 件の DPO ペアを {args.out} に書き出しました。")
        return 0

    if args.command == "stats":
        sessions = _load_sessions(traj_dir)
        print_stats(sessions)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
