"""AnythingPixie ローカル eval スイート — 決定的チェッカー群。

すべて LLM を一切使わず、ファイル状態・プロセス実行結果・run_graph のメトリクスだけで
合否を判定する。runner.py がタスクごとに RunResult を組み立て、タスク定義 (JSON) の
`checkers` リストに従ってここの関数を呼び出す。

各チェッカー関数のシグネチャ: (run: RunResult, **kwargs) -> (bool, str)
  - bool: 合否
  - str : 人間可読の判定理由（レポートに載る）
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# checkers.py 単体 import（runner.py 経由でない場合）でも src/ を解決できるようにする。
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# =====================================================
# RunResult — 1タスク実行分のメトリクス
# =====================================================

@dataclass
class RunResult:
    """1タスクの run_graph 実行結果 + メタ情報。checkers はこれだけを見て判定する。"""

    task_id: str
    final_answer: str = ""
    tool_call_count: int = 0
    exit_reason: str = ""
    duration_sec: float = 0.0
    workspace_dir: Path | None = None
    guardrail_fire_count: int = 0
    new_files: list = field(default_factory=list)
    crashed: bool = False
    crash_message: str = ""
    timed_out: bool = False
    output_log: str = ""


def _resolve_path(run: RunResult, path: str) -> Path:
    return Path(run.workspace_dir) / path


def _resolve_python(token: str) -> str:
    """コマンドリスト中の "{python}" プレースホルダを現在の Python 実行系に置換する。"""
    return sys.executable if token == "{python}" else token


# =====================================================
# チェッカー本体
# =====================================================

def answer_contains(run: RunResult, text: str) -> tuple[bool, str]:
    """最終回答テキストに text が含まれるか。"""
    ok = text in (run.final_answer or "")
    return ok, f"final_answer に '{text}' が{'含まれる' if ok else '含まれない'}"


def answer_contains_any(run: RunResult, texts: list[str]) -> tuple[bool, str]:
    """最終回答テキストに texts のいずれかが含まれるか（OR条件）。"""
    hits = [t for t in texts if t in (run.final_answer or "")]
    ok = bool(hits)
    return ok, f"候補 {texts} のうち一致: {hits or '(なし)'}"


def answer_not_contains(run: RunResult, text: str) -> tuple[bool, str]:
    """最終回答テキストに text が含まれないこと。"""
    ok = text not in (run.final_answer or "")
    return ok, f"final_answer に '{text}' が{'含まれない(OK)' if ok else '含まれてしまっている'}"


def file_exists(run: RunResult, path: str) -> tuple[bool, str]:
    """タスク作業ディレクトリ内に path が存在するか。"""
    p = _resolve_path(run, path)
    ok = p.exists()
    return ok, f"{path} が{'存在する' if ok else '存在しない'}"


def file_contains(run: RunResult, path: str, text: str) -> tuple[bool, str]:
    """path の内容に text が含まれるか（存在しなければ即失敗）。"""
    p = _resolve_path(run, path)
    if not p.exists():
        return False, f"{path} が存在しない"
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"{path} の読み込みに失敗: {e}"
    ok = text in content
    return ok, f"{path} に '{text}' が{'含まれる' if ok else '含まれない'}"


def file_passes_fast_gate(run: RunResult, path: str) -> tuple[bool, str]:
    """py_compile + import解決 + ruff の高速ゲート（subagent.run_fast_gate_check）を通過するか。"""
    p = _resolve_path(run, path)
    if not p.exists():
        return False, f"{path} が存在しない"
    from subagent import run_fast_gate_check
    out = run_fast_gate_check(str(p))
    ok = out == ""
    return ok, "fast gate 通過" if ok else f"fast gate 失敗:\n{out[:500]}"


def subprocess_stdout_equals(
    run: RunResult, cmd: list[str], expected: str, cwd: str = None, timeout: int = 30
) -> tuple[bool, str]:
    """cmd を実行し、stdout（前後空白 strip）が expected と一致するか。

    cmd の要素に "{python}" があれば実行中の Python 実行系のフルパスに置換する
    （タスク workspace には .venv がないため）。
    """
    workdir = cwd or str(run.workspace_dir)
    resolved_cmd = [_resolve_python(c) for c in cmd]
    try:
        result = subprocess.run(
            resolved_cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", cwd=workdir, timeout=timeout,
        )
    except Exception as e:
        return False, f"実行失敗: {e}"
    actual = (result.stdout or "").strip()
    ok = actual == expected.strip()
    detail = f"stdout={actual!r} expected={expected.strip()!r}"
    if result.returncode != 0:
        detail += f" (returncode={result.returncode}, stderr={(result.stderr or '')[:300]!r})"
    return ok, detail


def tool_calls_under(run: RunResult, n: int) -> tuple[bool, str]:
    """tool_call_count が n 未満か。"""
    ok = run.tool_call_count < n
    return ok, f"tool_call_count={run.tool_call_count} (< {n} が期待値)"


def no_guardrail_fired(run: RunResult) -> tuple[bool, str]:
    """実行ログ中にガードレール発火メッセージが1件も出ていないか。"""
    ok = run.guardrail_fire_count == 0
    return ok, f"guardrail_fire_count={run.guardrail_fire_count}"


def new_file_created(run: RunResult) -> tuple[bool, str]:
    """タスク実行前後で新規ファイルが1件以上作られたか（曖昧指示タスクの完了判定用）。"""
    ok = len(run.new_files) >= 1
    return ok, f"new_files={run.new_files}"


def not_crashed(run: RunResult) -> tuple[bool, str]:
    """run_graph 実行がクラッシュ/タイムアウトしていないか。"""
    if run.crashed:
        return False, f"クラッシュ: {run.crash_message[:300]}"
    if run.timed_out:
        return False, "タイムアウト"
    return True, "OK"


def exit_reason_contains(run: RunResult, text: str) -> tuple[bool, str]:
    """AgentState.exit_reason に text が含まれるか（終了経路の検証用）。"""
    ok = text in (run.exit_reason or "")
    return ok, f"exit_reason='{run.exit_reason}' に '{text}' が{'含まれる' if ok else '含まれない'}"


# =====================================================
# レジストリ — runner.py がタスク定義の "type" 文字列からここを引く
# =====================================================

CHECKER_REGISTRY = {
    "answer_contains": answer_contains,
    "answer_contains_any": answer_contains_any,
    "answer_not_contains": answer_not_contains,
    "file_exists": file_exists,
    "file_contains": file_contains,
    "file_passes_fast_gate": file_passes_fast_gate,
    "subprocess_stdout_equals": subprocess_stdout_equals,
    "tool_calls_under": tool_calls_under,
    "no_guardrail_fired": no_guardrail_fired,
    "new_file_created": new_file_created,
    "not_crashed": not_crashed,
    "exit_reason_contains": exit_reason_contains,
}
