"""同一workspaceで複数のコーディングエージェントを比較する小型ハーネス。"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALS_DIR = HERE.parent
REPO_ROOT = EVALS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = HERE / "results"

for path in (SRC_DIR, EVALS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tasks file must contain a JSON array")
    return data


def create_workspace(task: dict, root: Path) -> None:
    for rel, content in task.get("files", {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def initialize_git_repo(root: Path) -> None:
    """探索トラックで全agentが同じrepository構造を参照できるようにする。"""
    env = os.environ.copy()
    env.update({"GIT_AUTHOR_NAME": "Eval", "GIT_AUTHOR_EMAIL": "eval@local",
                "GIT_COMMITTER_NAME": "Eval", "GIT_COMMITTER_EMAIL": "eval@local"})
    for command in (["git", "init", "-q"], ["git", "add", "."],
                    ["git", "commit", "-q", "-m", "fixture"]):
        subprocess.run(command, cwd=root, env=env, check=True, capture_output=True)


def provided_context(task: dict) -> str:
    blocks = ["\n\n【提供済みファイル（内容は正本）】"]
    for path, body in task.get("files", {}).items():
        blocks.append(f"\n--- {path} ---\n```\n{body}```")
    return "".join(blocks)


def run_check(root: Path, check: dict) -> tuple[bool, str]:
    kind = check["type"]
    if kind == "pytest":
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=root,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr)[-500:]

    if kind == "not_contains_any":
        hits = []
        for file in root.rglob("*"):
            relative_parts = file.relative_to(root).parts
            if (file.is_file()
                    and "__pycache__" not in relative_parts
                    and not any(part.startswith(".") for part in relative_parts)):
                body = file.read_text(encoding="utf-8", errors="replace")
                hits.extend(token for token in check["texts"] if token in body)
        return not hits, f"unexpected tokens={hits}"

    path = root / check.get("path", "")
    if not path.exists():
        return False, f"missing: {check.get('path')}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if kind == "contains":
        ok = check["text"] in text
        return ok, f"contains {check['text']!r}: {ok}"
    if kind == "equals":
        ok = text == check["text"]
        return ok, f"exact content preserved: {ok}"
    if kind == "py_compile":
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return proc.returncode == 0, (proc.stderr or "py_compile passed")[-500:]
    if kind == "json_value":
        value = json.loads(text)
        ok = value.get(check["key"]) == check["value"]
        return ok, f"{check['key']}={value.get(check['key'])!r}"
    if kind == "ast_function_count_at_least":
        count = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(ast.parse(text)))
        ok = count >= int(check["count"])
        return ok, f"function_count={count}"
    raise ValueError(f"unknown check type: {kind}")


def run_awp(task: dict, root: Path, args, prompt: str) -> tuple[int, str, dict]:
    from runner import _run_task_body

    result = _run_task_body(
        {
            "id": task["id"],
            "description": prompt,
            "max_tool_calls": args.max_turns,
            "timeout_sec": args.timeout,
            "workset_files": sorted(task.get("files", {})) if args.track == "provided" else [],
        },
        root,
        args.base_url,
        args.api_key,
        args.model,
    )
    return (0 if not result.get("crashed") else 1), result.get("final_answer", ""), result


def run_external(task: dict, root: Path, adapter: dict, timeout: int, *, prompt: str,
                 track: str) -> tuple[int, str, dict]:
    command = []
    for token in adapter["command"]:
        if token == "{files}":
            if track == "provided":
                command.extend(sorted(task.get("files", {})))
            continue
        command.append(str(token).replace("{prompt}", prompt).replace("{workspace}", str(root)))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in adapter.get("env", {}).items()})
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command, cwd=root, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output, {"duration_sec": round(time.perf_counter() - started, 2)}
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + (exc.stderr or "")) if isinstance(exc.stdout, str) else ""
        return 124, output, {"duration_sec": round(time.perf_counter() - started, 2), "timed_out": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="awp または adapters JSON 内の名前")
    parser.add_argument("--track", choices=("provided", "discovery"), default="provided",
                        help="provided=全ファイル提供、discovery=リポジトリから探索")
    parser.add_argument("--tasks", type=Path, default=HERE / "tasks.json")
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--base-url", default="http://192.168.0.200:8080/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", default="gemma-4-26B-A4B-it-ultra-uncensored-heretic.Q5_K_S.gguf")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    if args.task:
        tasks = [task for task in tasks if task["id"] == args.task]
    if not tasks:
        raise SystemExit("no matching tasks")

    adapters = {}
    if args.agent != "awp":
        if not args.adapters:
            raise SystemExit("external agents require --adapters")
        adapters = json.loads(args.adapters.read_text(encoding="utf-8"))
        if args.agent not in adapters:
            raise SystemExit(f"adapter not found: {args.agent}")
        executable = adapters[args.agent]["command"][0]
        if not args.dry_run and shutil.which(executable) is None:
            raise SystemExit(f"executable not found: {executable}")

    if args.dry_run:
        print(f"OK: agent={args.agent}, track={args.track}, tasks={len(tasks)}, repeat={args.repeat}")
        return 0

    results = []
    for task in tasks:
        for repetition in range(1, max(args.repeat, 1) + 1):
            with tempfile.TemporaryDirectory(prefix=f"compare_{task['id']}_") as tmp:
                root = Path(tmp)
                create_workspace(task, root)
                initialize_git_repo(root)
                prompt = task["prompt"]
                if args.track == "provided" and args.agent == "goose":
                    prompt += provided_context(task)
                started = time.perf_counter()
                if args.agent == "awp":
                    code, output, metrics = run_awp(task, root, args, prompt)
                else:
                    code, output, metrics = run_external(
                        task, root, adapters[args.agent], args.timeout,
                        prompt=prompt, track=args.track,
                    )
                checks = []
                for check in task.get("checks", []):
                    try:
                        passed, detail = run_check(root, check)
                    except Exception as exc:
                        passed, detail = False, f"checker error: {exc}"
                    checks.append({"type": check["type"], "passed": passed, "detail": detail})
                passed = code == 0 and all(check["passed"] for check in checks)
                record = {
                    "task_id": task["id"], "repetition": repetition, "passed": passed,
                    "exit_code": code, "duration_sec": metrics.get(
                        "duration_sec", round(time.perf_counter() - started, 2)),
                    "tool_call_count": metrics.get("tool_call_count"),
                    "checks": checks, "output_preview": output[-1000:],
                    "agent_log": metrics.get("output_log", output)[-20000:],
                }
                results.append(record)
                print(f"{task['id']} rep={repetition}: {'PASS' if passed else 'FAIL'} ({record['duration_sec']}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = RESULTS_DIR / f"{args.agent}_{args.track}_{stamp}.json"
    payload = {
        "meta": {"agent": args.agent, "track": args.track,
                 "model": args.model, "base_url": args.base_url},
        "summary": {"passed": sum(r["passed"] for r in results), "total": len(results)},
        "results": results,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Result: {target}")
    return 0 if payload["summary"]["passed"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
