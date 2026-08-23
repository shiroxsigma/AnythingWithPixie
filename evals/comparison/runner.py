"""同一workspaceで複数のコーディングエージェントを比較する小型ハーネス。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sanitize_base_url(value: str) -> str:
    """Return an endpoint identity without credentials, query parameters, or fragments."""
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<redacted-url>"
    if not parsed.scheme or not hostname:
        return "<redacted-url>"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = hostname if port is None else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def sanitize_command(argv: list[str]) -> list[str]:
    """Remove CLI secrets while retaining enough detail to reproduce a run."""
    sanitized = []
    redact_next = False
    sanitize_url_next = False
    for token in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if sanitize_url_next:
            sanitized.append(sanitize_base_url(token))
            sanitize_url_next = False
            continue
        if token == "--api-key":
            sanitized.append(token)
            redact_next = True
        elif token.startswith("--api-key="):
            sanitized.append("--api-key=<redacted>")
        elif token == "--base-url":
            sanitized.append(token)
            sanitize_url_next = True
        elif token.startswith("--base-url="):
            sanitized.append(f"--base-url={sanitize_base_url(token.split('=', 1)[1])}")
        else:
            sanitized.append(token)
    return sanitized


def _git_bytes(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None


def _untracked_state(repo_root: Path) -> dict:
    proc = _git_bytes(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    if proc is None or proc.returncode != 0:
        return {
            "count": None,
            "paths": None,
            "paths_sha256": None,
            "content_sha256": None,
            "hash_complete": False,
        }

    raw_paths = sorted(path for path in proc.stdout.split(b"\0") if path)
    paths_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    hash_complete = True
    display_paths = []
    for raw_path in raw_paths:
        paths_digest.update(len(raw_path).to_bytes(8, "big"))
        paths_digest.update(raw_path)
        content_digest.update(len(raw_path).to_bytes(8, "big"))
        content_digest.update(raw_path)
        display_paths.append(raw_path.decode("utf-8", errors="replace"))
        target = repo_root / os.fsdecode(raw_path)
        file_digest = hashlib.sha256()
        try:
            if target.is_symlink():
                file_digest.update(os.fsencode(os.readlink(target)))
            else:
                with target.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        file_digest.update(chunk)
        except OSError:
            hash_complete = False
            continue
        content_digest.update(file_digest.digest())

    return {
        "count": len(raw_paths),
        "paths": display_paths,
        "paths_sha256": paths_digest.hexdigest(),
        "content_sha256": content_digest.hexdigest() if hash_complete else None,
        "hash_complete": hash_complete,
    }


def capture_source_state(repo_root: Path = REPO_ROOT) -> dict:
    """Capture a content-derived Git/worktree identity for benchmark provenance."""
    captured_at = _utc_now()
    head_proc = _git_bytes(repo_root, "rev-parse", "--verify", "HEAD")
    diff_proc = _git_bytes(repo_root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    status_proc = _git_bytes(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if any(proc is None or proc.returncode != 0 for proc in (head_proc, diff_proc, status_proc)):
        return {
            "captured_at": captured_at,
            "git_available": False,
            "git_head": None,
            "dirty": None,
            "tracked_dirty": None,
            "tracked_diff_sha256": None,
            "untracked": _untracked_state(repo_root),
            "state_id": None,
        }

    assert head_proc is not None and diff_proc is not None and status_proc is not None
    git_head = head_proc.stdout.decode("ascii", errors="replace").strip()
    tracked_diff_sha256 = hashlib.sha256(diff_proc.stdout).hexdigest()
    status_entries = [entry for entry in status_proc.stdout.split(b"\0") if entry]
    tracked_dirty = bool(diff_proc.stdout) or any(not entry.startswith(b"?? ") for entry in status_entries)
    untracked = _untracked_state(repo_root)
    dirty = bool(status_entries)
    identity = {
        "git_head": git_head,
        "tracked_diff_sha256": tracked_diff_sha256,
        "git_status_sha256": hashlib.sha256(status_proc.stdout).hexdigest(),
        "untracked_paths_sha256": untracked["paths_sha256"],
        "untracked_content_sha256": untracked["content_sha256"],
        "untracked_hash_complete": untracked["hash_complete"],
    }
    state_id = None
    if untracked["hash_complete"]:
        state_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return {
        "captured_at": captured_at,
        "git_available": True,
        "git_head": git_head,
        "dirty": dirty,
        "tracked_dirty": tracked_dirty,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked": untracked,
        "state_id": state_id,
    }


def build_run_metadata(args, tasks: list[dict], *, started_at: str, source_state: dict, command: list[str]) -> dict:
    state_id = source_state.get("state_id")
    return {
        "agent": args.agent,
        "track": args.track,
        "model": args.model,
        "base_url": sanitize_base_url(args.base_url),
        "base_url_identity": sanitize_base_url(args.base_url),
        "run_id": str(uuid.uuid4()),
        "started_at": started_at,
        "command": sanitize_command(command),
        "task": args.task,
        "task_ids": [task["id"] for task in tasks],
        "repeat": max(args.repeat, 1),
        "source_at_start": source_state,
        "source_consistency": {
            "start_state_id": state_id,
            "observed_state_ids": [state_id] if state_id else [],
            "observation_count": 1,
            "same_source_for_all_repetitions": True if state_id else None,
        },
    }


def observe_source_state(meta: dict, source_state: dict) -> None:
    consistency = meta["source_consistency"]
    consistency["observation_count"] += 1
    state_id = source_state.get("state_id")
    if state_id and state_id not in consistency["observed_state_ids"]:
        consistency["observed_state_ids"].append(state_id)
    start_state_id = consistency["start_state_id"]
    current = consistency["same_source_for_all_repetitions"]
    if current is False or current is None:
        return
    if not start_state_id or not state_id:
        consistency["same_source_for_all_repetitions"] = None
    elif state_id != start_state_id:
        consistency["same_source_for_all_repetitions"] = False


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


def save_results(target: Path, meta: dict, results: list[dict]) -> dict:
    """完了済み試行を原子的に保存し、長時間バッチ中断時の結果消失を防ぐ。"""
    payload = {
        "meta": meta,
        "summary": {"passed": sum(r["passed"] for r in results), "total": len(results)},
        "results": results,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + ".tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(target)
    return payload


def main() -> int:
    run_started_at = _utc_now()
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = RESULTS_DIR / f"{args.agent}_{args.track}_{stamp}.json"
    source_at_start = capture_source_state()
    meta = build_run_metadata(
        args,
        tasks,
        started_at=run_started_at,
        source_state=source_at_start,
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
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
                source_before = capture_source_state()
                observe_source_state(meta, source_before)
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
                source_after = capture_source_state()
                observe_source_state(meta, source_after)
                record = {
                    "task_id": task["id"], "repetition": repetition, "passed": passed,
                    "exit_code": code, "duration_sec": metrics.get(
                        "duration_sec", round(time.perf_counter() - started, 2)),
                    "tool_call_count": metrics.get("tool_call_count"),
                    "llm_call_count": metrics.get("llm_call_count"),
                    "decode_tokens": metrics.get("decode_tokens"),
                    "peak_prompt_tokens": metrics.get("peak_prompt_tokens"),
                    "peak_workset_injection_chars": metrics.get("peak_workset_injection_chars"),
                    "workset_reread_count": metrics.get("workset_reread_count"),
                    "llm_call_metrics": metrics.get("llm_call_metrics"),
                    "checks": checks, "output_preview": output[-1000:],
                    "agent_log": metrics.get("output_log", output)[-20000:],
                    "source_state_id_before": source_before.get("state_id"),
                    "source_state_id_after": source_after.get("state_id"),
                }
                results.append(record)
                save_results(target, meta, results)
                print(f"{task['id']} rep={repetition}: {'PASS' if passed else 'FAIL'} ({record['duration_sec']}s)")

    payload = save_results(target, meta, results)
    print(f"Result: {target}")
    return 0 if payload["summary"]["passed"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
