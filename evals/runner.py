"""AnythingPixie ローカル eval スイート — メインランナー。

pytest/CI とは独立して動作する自己ベンチマーク。実 LLM（既定: localhost:8080 の
llama-server, OpenAI互換API）に接続し、tasks/ 配下のタスクごとに run_graph を1ターン
実行させ、checkers.py の決定的チェッカーで採点する。

ハーネス改善（プロンプト変更・config 変更・新機能）の効果を客観的に測る適応度関数
として使う。CI では実行しない（実 LLM 前提のため）。

使い方:
    python evals/runner.py                      # 全タスク実行
    python evals/runner.py --task 01_read_config_value   # 単体実行
    python evals/runner.py --dry-run             # LLMを呼ばず設定検証のみ
    python evals/runner.py --compare results/eval_20260101_000000.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
TASKS_DIR = EVALS_DIR / "tasks"
WORKSPACES_DIR = EVALS_DIR / "workspaces"
RESULTS_DIR = EVALS_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import checkers as ck  # noqa: E402  (sys.path 設定後の import)

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "local-model"
DEFAULT_TASK_TIMEOUT_SEC = 300

# run_graph 実行ログ中に出現すると「ガードレール発火」とみなすマーカー文字列。
# src/engine.py のガードレール注入メッセージ（[System] 始まり）に対応する。
GUARDRAIL_MARKERS = (
    "反復出力を検知",
    "フォーマットエラーを検知",
    "次の行動を宣言していますが",
    "回答が短すぎます",
)


# =====================================================
# タスクロード
# =====================================================

def load_tasks(task_id: str | None = None) -> list[dict]:
    tasks = []
    for f in sorted(TASKS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        data.setdefault("id", f.stem)
        tasks.append(data)
    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
        if not tasks:
            raise SystemExit(f"task not found: {task_id}")
    return tasks


def _copy_workspace(task: dict, dest_dir: Path) -> None:
    ws_name = task.get("workspace") or ""
    if not ws_name:
        return
    src = WORKSPACES_DIR / ws_name
    if not src.exists():
        raise FileNotFoundError(f"workspace template not found: {src}")
    for item in src.iterdir():
        target = dest_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _snapshot_files(root: Path) -> set:
    result = set()
    for p in root.rglob("*"):
        if ".pixie_notes" in p.parts:
            continue
        if p.is_file():
            result.add(str(p.relative_to(root)))
    return result


# =====================================================
# .pixie_notes 等のグローバル状態のタスク別隔離
# =====================================================

@contextlib.contextmanager
def _isolated_pixie_env(task_root: Path):
    """paths.get_data_path をタスク専用ディレクトリへ一時的にリダイレクトする。

    state_board.json / CORE_MEMORY.md / CONTEXT_SUMMARY.md / .pixie_notes 配下の
    バックアップ・ログ・code_index キャッシュ等、AnythingPixie 本体が「アプリルート
    基準の絶対パス」で読み書きするあらゆる永続データを、タスクごとの一時ディレクトリに
    差し替える。各モジュールは `from paths import get_data_path` で関数オブジェクトを
    自分の名前空間に束縛済みのため、モジュールごとに参照を差し替える必要がある
    （src/ 本体は一切変更しない・呼び出し側からの monkeypatch のみ）。
    """
    import code_tool
    import engine
    import state
    import subagent
    import tools

    task_root = Path(task_root)

    def _patched(rel_path: str) -> str:
        return str(task_root / rel_path)

    targets = [state, engine, subagent, tools, code_tool]

    # toolpacks.manga は .pixie_notes/manga_* を独自に get_data_path() 経由で読み書きする
    # （tools.py 等とは別モジュール名前空間）ため、ここでも同様に隔離対象に含める。
    # 未実装/未使用環境でも import 失敗で全体を壊さないよう任意扱い。
    try:
        import toolpacks.manga as _manga_pack
        targets.append(_manga_pack)
    except Exception:
        pass

    # 本体は project 単位のパスを get_project_data_path 経由で解決する（models/config 等の
    # アプリ共通リソースのみ従来通り get_data_path）。各モジュール名前空間に束縛済みの両関数を
    # タスク別ディレクトリへ差し替える。モジュールにより import 状況が異なるため、実在する
    # ものだけを対象にする。
    patch_names = ("get_data_path", "get_project_data_path")
    saved_funcs = [
        (m, name, getattr(m, name))
        for m in targets for name in patch_names if hasattr(m, name)
    ]
    # ホワイトボードパスは config.get_whiteboard_path()（engine 名前空間に束縛済み）経由。
    saved_whiteboard = getattr(engine, "get_whiteboard_path", None)

    for m, name, _orig in saved_funcs:
        setattr(m, name, _patched)
    if saved_whiteboard is not None:
        engine.get_whiteboard_path = lambda: _patched("CONTEXT_SUMMARY.md")

    try:
        yield
    finally:
        for m, name, orig in saved_funcs:
            setattr(m, name, orig)
        if saved_whiteboard is not None:
            engine.get_whiteboard_path = saved_whiteboard


# =====================================================
# コンテキスト構築
# =====================================================

def _make_llm(base_url: str, api_key: str, model: str):
    from llm_client import LMStudioBackend
    return LMStudioBackend(base_url=base_url, api_key=api_key, model=model)


def _build_context(llm, code_mode: bool = False, active_packs: set | None = None, task_mode: str | None = None,
                    harvest: bool = False, eval_task_id: str | None = None):
    from main import AppContext
    ctx = AppContext()
    ctx.llm = llm
    ctx.llm_model_name = getattr(llm, "model", "")
    ctx.code_mode = code_mode
    ctx.force_deep = False
    # [LFM専用] main.py の起動時判定・/api 切替時判定と同じロジックをここでも再現する。
    # eval ランナーは --model で渡された文字列だけを頼りに接続先モデルを判別するため、
    # "lfm" を含む場合のみ is_lfm25 / supports_tool_role を有効化する（既定は従来通り False）。
    ctx.is_lfm25 = "lfm" in ctx.llm_model_name.lower()
    ctx.supports_tool_role = ctx.is_lfm25  # main.py の既定と同じ（LM Studio + 非FC変換パス）
    ctx.phase = "EXECUTING"
    ctx.debug_mode = False
    ctx.review_mode = False
    ctx.verify_mode = False
    # ツールパック機構（タスク定義の "toolpacks": [...] / "task_mode": "manga" 等で指定）。
    # 既定は空集合・None = 従来通り（コアツールのみ・通常モード）。
    ctx.active_packs = set(active_packs) if active_packs else set()
    ctx.task_mode = task_mode

    # 軌跡ロギング（--harvest 時のみ強制ON）。詳細設計 §3 の注意: _isolated_pixie_env は
    # get_data_path をタスク別一時ディレクトリへリダイレクトするが、trajectory モジュールは
    # その隔離対象（targets）に含めていないため、TrajectoryLogger が使う get_data_path は
    # 常に実プロジェクトの .pixie_notes/trajectories/ を指す（教師データ収穫が目的のため、
    # 意図的に隔離しない）。タスクごとに独立セッション（s_..._eval_<task_id>）にする。
    if harvest:
        from trajectory import TrajectoryLogger
        suffix = f"_eval_{eval_task_id}" if eval_task_id else "_eval"
        ctx.trajectory = TrajectoryLogger(enabled=True, session_suffix=suffix)
    else:
        ctx.trajectory = None
    return ctx


# =====================================================
# 1タスク実行（別スレッドでタイムアウト監視される本体）
# =====================================================

def _run_task_body(task: dict, task_dir: Path, base_url: str, api_key: str, model: str,
                    harvest: bool = False) -> dict:
    """LLM呼び出しを含む run_graph 1ターンの実処理。"""
    from engine import build_system_text, run_graph
    from registry import set_state_board
    from state import AgentState, AgentStateBoard

    llm = _make_llm(base_url, api_key, model)
    task_packs = set(task.get("toolpacks", []))
    task_mode = task.get("task_mode")
    ctx = _build_context(
        llm, code_mode=task.get("code_mode", False), active_packs=task_packs, task_mode=task_mode,
        harvest=harvest, eval_task_id=task["id"],
    )
    if ctx.trajectory is not None:
        ctx.trajectory.log_session_meta(
            model=model,
            base_url=base_url,
            mode=("manga" if task_mode == "manga" else ("code" if task.get("code_mode") else "normal")),
            active_packs=task_packs,
            sampling_profile={},
            n_ctx=None,
            eval_task=task["id"],
        )

    output_chunks: list[str] = []

    def _out_fn(text="", end="", flush=True, **_kw):
        output_chunks.append(text)

    with _isolated_pixie_env(task_dir):
        # task_mode="manga" 指定時、明示的に toolpacks に "manga" がなくても pack を有効化する
        # （/manga コマンドが task_mode 設定と同時に active_packs へ追加するのと同じ挙動）。
        if task_mode == "manga":
            task_packs.add("manga")
            ctx.active_packs.add("manga")
        for pname in task_packs:
            from toolpacks import load_pack
            load_pack(pname)

        board = AgentStateBoard(file_path=str(task_dir / ".pixie_notes" / "state_board.json"))
        agent_state = AgentState(state_board=board)
        set_state_board(agent_state.state_board)
        agent_state.chat_history.add("user", task["description"])
        agent_state.max_tool_calls = task.get("max_tool_calls", 15)

        prev_cwd = os.getcwd()
        os.chdir(task_dir)
        try:
            final_answer = run_graph(
                context=ctx,
                state=agent_state,
                show_thinking=False,
                system_msg_builder=build_system_text,
                interactive_fn=None,  # フル自動（半自動モードの承認プロンプトを一切出さない）
                output_fn=_out_fn,
            )
        finally:
            os.chdir(prev_cwd)

    full_output = "".join(output_chunks)
    guardrail_fire_count = sum(full_output.count(marker) for marker in GUARDRAIL_MARKERS)

    return {
        "final_answer": final_answer or "",
        "tool_call_count": agent_state.tool_call_count,
        "exit_reason": agent_state.exit_reason,
        "guardrail_fire_count": guardrail_fire_count,
        "output_log": full_output,
        # harvest モード専用: checker 判定後に run_single_task が mark_eval_result() を
        # 呼ぶための TrajectoryLogger 参照（同一プロセス内のスレッド実行のためオブジェクト
        # 参照をそのまま渡せる。非harvest時は None）。
        "trajectory": ctx.trajectory,
    }


def _run_checker(run_result: ck.RunResult, spec: dict) -> dict:
    spec = dict(spec)
    ctype = spec.pop("type")
    fn = ck.CHECKER_REGISTRY.get(ctype)
    if fn is None:
        return {"type": ctype, "passed": False, "detail": f"未知のチェッカー種別: {ctype}", "args": spec}
    try:
        passed, detail = fn(run_result, **spec)
    except Exception as e:
        return {"type": ctype, "passed": False, "detail": f"チェッカー実行時の例外: {e}", "args": spec}
    return {"type": ctype, "passed": bool(passed), "detail": detail, "args": spec}


def run_single_task(
    task: dict, base_url: str, api_key: str, model: str,
    timeout_sec: int = DEFAULT_TASK_TIMEOUT_SEC, keep_workspace: bool = False,
    harvest: bool = False,
) -> dict:
    """1タスクをテンプレートコピー→隔離実行→採点まで一気通貫で行う。

    タイムアウトは ThreadPoolExecutor(max_workers=1) + future.result(timeout=) で監視する。
    タイムアウト時、バックグラウンドスレッド自体は強制終了できない（Python の制約）ため
    生き残る可能性があるが、ランナー全体は次のタスクに進める（ハングでスイート全体が
    止まらないことを優先）。
    """
    task_id = task["id"]
    tmp_root = Path(tempfile.mkdtemp(prefix=f"pixie_eval_{task_id}_"))
    try:
        _copy_workspace(task, tmp_root)
        pre_files = _snapshot_files(tmp_root)

        start = time.monotonic()
        result_data: dict = {}
        crashed = False
        crash_message = ""
        timed_out = False

        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run_task_body, task, tmp_root, base_url, api_key, model, harvest)
            try:
                result_data = future.result(timeout=timeout_sec)
            except FutureTimeoutError:
                timed_out = True
            except Exception:
                crashed = True
                crash_message = traceback.format_exc()

        duration = time.monotonic() - start
        post_files = _snapshot_files(tmp_root)
        new_files = sorted(post_files - pre_files)

        run_result = ck.RunResult(
            task_id=task_id,
            final_answer=result_data.get("final_answer", ""),
            tool_call_count=result_data.get("tool_call_count", 0),
            exit_reason=result_data.get("exit_reason", ""),
            duration_sec=duration,
            workspace_dir=tmp_root,
            guardrail_fire_count=result_data.get("guardrail_fire_count", 0),
            new_files=new_files,
            crashed=crashed,
            crash_message=crash_message,
            timed_out=timed_out,
            output_log=result_data.get("output_log", ""),
        )

        checker_results = []
        if not crashed and not timed_out:
            for spec in task.get("checkers", []):
                checker_results.append(_run_checker(run_result, spec))
        overall_pass = (not crashed) and (not timed_out) and all(r["passed"] for r in checker_results)

        # harvest モード: checker 判定が確定した時点で、run_graph 実行時点では null だった
        # turn_end.eval_passed を事後的に書き戻す（詳細設計 §5 / gold ティア判定の根拠）。
        _traj = result_data.get("trajectory")
        if _traj is not None:
            try:
                _traj.mark_eval_result(overall_pass)
            except Exception:
                pass

        return {
            "task_id": task_id,
            "difficulty": task.get("difficulty", ""),
            "description": task["description"],
            "passed": overall_pass,
            "crashed": crashed,
            "crash_message": crash_message,
            "timed_out": timed_out,
            "duration_sec": round(duration, 2),
            "tool_call_count": run_result.tool_call_count,
            "exit_reason": run_result.exit_reason,
            "guardrail_fire_count": run_result.guardrail_fire_count,
            "final_answer_len": len(run_result.final_answer),
            "final_answer_preview": run_result.final_answer[:300],
            "checker_results": checker_results,
        }
    finally:
        if not keep_workspace:
            shutil.rmtree(tmp_root, ignore_errors=True)


# =====================================================
# dry-run（LLM不使用の設定検証）
# =====================================================

def dry_run(tasks: list[dict]) -> list[str]:
    """タスク定義のロードとチェッカー定義の妥当性のみ検証する（LLM呼び出しなし）。"""
    problems = []
    for task in tasks:
        for required in ("id", "description", "checkers"):
            if required not in task:
                problems.append(f"{task.get('id', '?')}: 必須フィールド欠落: {required}")
        ws = task.get("workspace")
        if ws:
            if not (WORKSPACES_DIR / ws).exists():
                problems.append(f"{task['id']}: workspace が存在しない: {ws}")
        for c in task.get("checkers", []):
            if "type" not in c:
                problems.append(f"{task['id']}: checker に type がない: {c}")
            elif c["type"] not in ck.CHECKER_REGISTRY:
                problems.append(f"{task['id']}: 未知のチェッカー種別: {c['type']}")
    return problems


# =====================================================
# レポート出力
# =====================================================

def write_reports(results: list[dict], meta: dict) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"eval_{ts}.json"
    md_path = RESULTS_DIR / f"eval_{ts}.md"

    payload = {"meta": meta, "results": results}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    total_duration = sum(r["duration_sec"] for r in results)

    lines = [
        f"# Eval Report — {ts}",
        "",
        f"- Base URL: {meta.get('base_url')}",
        f"- Model: {meta.get('model')}",
        f"- Tasks: {n_pass}/{n} passed ({(n_pass / n * 100 if n else 0):.1f}%)",
        f"- Total duration: {total_duration:.1f}s",
        "",
        "| Task | Difficulty | Result | Tool Calls | Duration(s) | Exit Reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["passed"]:
            mark = "PASS"
        elif r["timed_out"]:
            mark = "TIMEOUT"
        elif r["crashed"]:
            mark = "CRASH"
        else:
            mark = "FAIL"
        lines.append(
            f"| {r['task_id']} | {r.get('difficulty', '')} | {mark} | "
            f"{r['tool_call_count']} | {r['duration_sec']} | {r['exit_reason']} |"
        )

    lines.append("")
    lines.append("## Details")
    for r in results:
        lines.append(f"\n### {r['task_id']} — {'PASS' if r['passed'] else 'FAIL'}")
        lines.append(f"- description: {r['description']}")
        lines.append(f"- exit_reason: {r['exit_reason']}")
        lines.append(f"- tool_call_count: {r['tool_call_count']}")
        lines.append(f"- guardrail_fire_count: {r['guardrail_fire_count']}")
        lines.append(f"- final_answer_len: {r['final_answer_len']}")
        if r["final_answer_preview"]:
            lines.append(f"- final_answer_preview: {r['final_answer_preview']!r}")
        if r["crashed"]:
            lines.append(f"- CRASH:\n```\n{r['crash_message']}\n```")
        if r["timed_out"]:
            lines.append("- TIMEOUT")
        for cr in r["checker_results"]:
            cmark = "PASS" if cr["passed"] else "FAIL"
            lines.append(f"  - [{cmark}] {cr['type']}: {cr['detail']}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def compare_results(old_path: str, new_results: list[dict]) -> None:
    old = json.loads(Path(old_path).read_text(encoding="utf-8"))
    old_results = old["results"]

    def _rate(results):
        n = len(results)
        return (sum(1 for r in results if r["passed"]) / n * 100) if n else 0.0

    def _avg_tools(results):
        n = len(results)
        return (sum(r["tool_call_count"] for r in results) / n) if n else 0.0

    print(f"\n=== Compare vs {old_path} ===")
    print(f"Success rate: {_rate(old_results):.1f}% -> {_rate(new_results):.1f}%")
    print(f"Avg tool calls: {_avg_tools(old_results):.2f} -> {_avg_tools(new_results):.2f}")

    old_by_id = {r["task_id"]: r for r in old_results}
    for r in new_results:
        prev = old_by_id.get(r["task_id"])
        if prev and prev["passed"] != r["passed"]:
            direction = "REGRESSED" if prev["passed"] and not r["passed"] else "FIXED"
            print(f"  [{direction}] {r['task_id']}")


# =====================================================
# CLI エントリポイント
# =====================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="AnythingPixie ローカル eval スイート")
    parser.add_argument("--task", help="特定タスクIDのみ実行")
    parser.add_argument("--dry-run", action="store_true", help="LLMを呼ばず設定検証のみ行う")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"既定: {DEFAULT_BASE_URL}")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TASK_TIMEOUT_SEC, help="タスクあたりの秒数上限")
    parser.add_argument("--compare", help="過去の結果JSONと比較する")
    parser.add_argument("--keep-workspace", action="store_true", help="デバッグ用に実行後の一時ディレクトリを残す")
    parser.add_argument("--harvest", action="store_true",
                        help="軌跡ロギングを強制ONにし、実プロジェクトの .pixie_notes/trajectories/ に"
                             "SFT/DPO教師データ用の軌跡を記録する（詳細設計 docs/design/trajectory-logging.md §5）")
    parser.add_argument("--repeat", type=int, default=1,
                        help="各タスクを複数回実行する（温度によるばらつきで複数の正解軌跡を収穫する。既定1回）")
    args = parser.parse_args()

    tasks = load_tasks(args.task)
    if not tasks:
        print("実行対象のタスクがありません。")
        return 1

    if args.dry_run:
        problems = dry_run(tasks)
        if problems:
            print(f"[dry-run] {len(problems)} 件の問題が見つかりました:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"[dry-run] OK: {len(tasks)} タスクのロード・チェッカー定義を検証しました。")
        return 0

    repeat = max(1, args.repeat)
    results = []
    for task in tasks:
        for rep in range(repeat):
            label = f"--- running {task['id']} ---" if repeat == 1 else f"--- running {task['id']} (rep {rep + 1}/{repeat}) ---"
            print(label)
            # タスク定義側の timeout_sec を優先し、なければ CLI の --timeout を使う
            r = run_single_task(
                task, args.base_url, args.api_key, args.model,
                timeout_sec=task.get("timeout_sec", args.timeout),
                keep_workspace=args.keep_workspace,
                harvest=args.harvest,
            )
            status = "PASS" if r["passed"] else ("TIMEOUT" if r["timed_out"] else ("CRASH" if r["crashed"] else "FAIL"))
            print(f"  {status} ({r['duration_sec']}s, tools={r['tool_call_count']}, exit={r['exit_reason']})")
            results.append(r)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "task_count": len(results),
        "pass_count": sum(1 for r in results if r["passed"]),
    }
    json_path, md_path = write_reports(results, meta)
    print(f"\nReport: {json_path}")
    print(f"Report: {md_path}")

    if args.compare:
        compare_results(args.compare, results)

    return 0 if meta["pass_count"] == meta["task_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
