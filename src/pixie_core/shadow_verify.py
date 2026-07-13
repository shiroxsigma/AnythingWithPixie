"""
AnythingPixie — 分岐点限定 lazy best-of-2: 編集のシャドウ検証モジュール

不可逆・高コストな分岐点（破壊的ファイル編集の実行）でのみ、実ファイルに書き込む前に
「適用後のファイル内容」を計算し（shadow_apply）、py_compile + ruff のみの軽量ゲートで
検証する（shadow_gate）。engine.run_graph はこのゲートが失敗した場合のみ最大1回の
再サンプルを行う（通常パス＝候補が最初からクリーンなら追加コストゼロ）。

依存方向: shadow_verify → tools（_compute_* 純粋関数を再利用・重複実装回避）,
          paths（venv Python 解決）。engine には依存しない。
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from paths import resolve_venv_python
from tools import (
    _compute_replace_lines_content,
    _compute_search_and_replace_content,
)

#: shadow_apply/shadow_gate の対象とする破壊的編集ツール名。
SHADOW_EDIT_TOOLS: frozenset[str] = frozenset({
    "write_file", "search_and_replace", "replace_lines", "append_to_file",
})

#: サブプロセス（py_compile/ruff）のタイムアウト秒。
_SHADOW_SUBPROCESS_TIMEOUT_SEC: int = 10


def shadow_apply(tool_name: str, tool_args: dict) -> tuple[str | None, str]:
    """破壊的編集ツールの引数から、実ファイルに書き込まずに「適用後のファイル内容」を計算する。

    tools.py の実適用ロジック（_compute_replace_lines_content /
    _compute_search_and_replace_content）を再利用し、write_file/append_to_file のみ
    ここで直接計算する（これらは fuzzy マッチ等の複雑ロジックを持たないため）。

    Args:
        tool_name: "write_file" | "search_and_replace" | "replace_lines" | "append_to_file"
        tool_args: ツール呼び出し引数（path 等を含む）

    Returns:
        (new_content, reason): 成功時 reason は ""。
        適用不能時は (None, 理由文字列)（"Error: " 始まりとは限らない簡潔な理由）。
    """
    path = tool_args.get("path", "")

    if tool_name == "write_file":
        content = tool_args.get("content")
        if content is None:
            return None, "content 引数がありません"
        return content, ""

    if tool_name == "append_to_file":
        content = tool_args.get("content")
        if content is None:
            return None, "content 引数がありません"
        target = Path(path) if path else None
        existing = ""
        if target is not None and target.exists():
            try:
                existing = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    existing = target.read_text(encoding="cp932")
                except Exception as e:
                    return None, f"既存ファイルの読み込みに失敗: {e}"
            except Exception as e:
                return None, f"既存ファイルの読み込みに失敗: {e}"
        return existing + content, ""

    if tool_name == "replace_lines":
        new_content, err = _compute_replace_lines_content(
            path,
            tool_args.get("start_line"),
            tool_args.get("end_line"),
            tool_args.get("new_content", ""),
        )
        if new_content is None:
            return None, err
        return new_content, ""

    if tool_name == "search_and_replace":
        outcome = _compute_search_and_replace_content(
            path, tool_args.get("search_block", ""), tool_args.get("replace_block", ""),
        )
        if not outcome.get("ok"):
            return None, outcome.get("error", "適用不能")
        return outcome.get("content"), ""

    return None, f"未対応のツール ({tool_name})"


def _shadow_py_compile(tmp_path: str, python_exe: str) -> str:
    """一時ファイルを py_compile で構文検査する（成功/対象外時は ""）。"""
    try:
        result = subprocess.run(
            [python_exe, "-m", "py_compile", tmp_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_SHADOW_SUBPROCESS_TIMEOUT_SEC,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode == 0:
        return ""
    err = (result.stderr or "").strip() or (result.stdout or "").strip()
    if not err:
        return f"[shadow py_compile] 失敗 (exit {result.returncode})"
    return f"[shadow py_compile]\n{err}"


def _shadow_ruff_check(tmp_path: str, python_exe: str) -> str:
    """一時ファイルを ruff (E,F のみ) で検査する（成功/未導入/設定エラー時は ""）。"""
    cmd = [python_exe, "-m", "ruff", "check", "--select", "E,F",
           "--output-format=concise", "--isolated", tmp_path]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_SHADOW_SUBPROCESS_TIMEOUT_SEC,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode == 0:
        return ""
    if result.returncode >= 2 and not (result.stdout or "").strip():
        return ""  # ruff 自体の設定/起動エラーは黙殺（誤検知でブロックしない）
    out = (result.stdout or "").strip()
    return f"[shadow ruff check]\n{out}\n" if out else ""


def shadow_gate(tool_name: str, tool_args: dict) -> str:
    """shadow_apply の結果を py_compile + ruff のみで検証する（.py ファイルのみ対象）。

    import 解決チェックは、一時ファイルのパスコンテキスト（元のプロジェクト構造から
    切り離された場所）では相対 import 等が誤検知されやすいため、意図的に含めない
    （subagent.run_fast_gate_check の3段ゲートとの違い）。

    Args:
        tool_name: 破壊的編集ツール名
        tool_args: ツール呼び出し引数

    Returns:
        問題なし、または .py 以外/shadow_apply 不能（ゲート対象外）の場合は ""。
        問題があればエラー要約文字列。
    """
    path = str(tool_args.get("path", ""))
    if not path.endswith(".py"):
        return ""

    new_content, _reason = shadow_apply(tool_name, tool_args)
    if new_content is None:
        return ""  # shadow_apply 不能 → ゲート対象外（事後の fast gate / エラーFBに委ねる）

    python_exe = resolve_venv_python(path) or sys.executable

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)

        err = _shadow_py_compile(tmp_path, python_exe)
        if err:
            return err

        err = _shadow_ruff_check(tmp_path, python_exe)
        if err:
            return err

        return ""
    except Exception:
        # ゲート機構自体の例外で通常パスを壊さない（検証スキップ扱い）
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
