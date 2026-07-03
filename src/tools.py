"""
ツール群モジュール

全22個の組込みツール（TOOL_REGISTRY）、Visionユーティリティ、テキスト解析ユーティリティを統合。
"""

import base64
import difflib
import inspect
import io
import locale
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# code_tool の7ツールを TOOL_REGISTRY に登録。registry 抽出により code_tool の
# トップレベル依存は registry のみになったため、末尾遅延 import ではなく先頭で
# 安全にロードできる（E402 解消）。
import code_tool  # noqa: F401

# レジストリ・共有グローバル状態は registry.py に集約（tools ↔ code_tool の循環 import 解消）。
# 後方互換のため tools 名前空間にも再エクスポート。
import registry
from config import ALWAYS_RECOMMEND, TOOL_RESULT_MAX_CHARS
from paths import get_bundled_path, get_data_path
from registry import (
    TOOL_REGISTRY,
    register_tool,
)

# ============================
# コアツール定義
# ============================


@register_tool(
    name="get_cwd",
    description="現在の作業ディレクトリの絶対パスを取得します。引数は要りません。",
    schema={"type": "object", "properties": {}, "required": []},
    prompt_desc="get_cwd: 現在の作業ディレクトリパスを取得 (引数なし)",
)
def get_cwd() -> str:
    """現在の作業ディレクトリを取得します。"""
    return str(Path.cwd())


@register_tool(
    name="get_file_dir",
    description="指定されたファイルが存在するディレクトリの絶対パスを取得します。",
    schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "対象ファイルのパス"}},
        "required": ["path"],
    },
    prompt_desc="get_file_dir(path): ファイルの親ディレクトリパスを取得",
)
def get_file_dir(path: str) -> str:
    """指定されたファイルが存在するディレクトリの絶対パスを取得します。"""
    target = Path(path)
    if not target.exists():
        return f"Error: ファイルまたはディレクトリが存在しません ({path})"
    try:
        return str(target.parent.resolve())
    except Exception as e:
        return f"Error: パスの取得に失敗しました: {e}"


@register_tool(
    name="list_directory",
    description="指定されたディレクトリ内のファイルとフォルダの一覧を取得します。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "一覧を取得するディレクトリのパス(.でカレントディレクトリ)"}
        },
        "required": [],
    },
    prompt_desc="list_directory(path): path内のファイル一覧を取得",
)
def list_directory(path: str = ".") -> str:
    """指定されたディレクトリのファイル一覧を取得します。"""
    target = Path(path)
    if not target.exists():
        return f"Error: フォルダが存在しません ({path})"
    if not target.is_dir():
        return f"Error: 指定されたパスはフォルダではありません ({path})"
    try:
        results = []
        for item in target.iterdir():
            stat = item.stat()
            type_str = "<DIR>" if item.is_dir() else f"{stat.st_size} bytes"
            results.append(f"{item.name} ({type_str})")
        return "\n".join(results) if results else "(空のフォルダ)"
    except Exception as e:
        return f"Error: フォルダの内容を取得できませんでした: {e}"


@register_tool(
    name="read_file",
    description="指定されたテキストファイルの内容を読み込みます。行範囲を指定して部分読み込みも可能です。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "読み込むファイルのパス"},
            "start_line": {"type": "integer", "description": "読み込み開始行番号 (1オリジン、省略時は先頭から)"},
            "end_line": {"type": "integer", "description": "読み込み終了行番号 (1オリジン、省略時は末尾まで)"},
        },
        "required": ["path"],
    },
    prompt_desc="read_file(path, start_line?, end_line?): pathのファイル内容を読み取り。行番号を指定して部分読み込み可能。コンテキストに余裕がある場合は全文読みを推奨（search_and_replace の精度向上に直結）。圧迫時は start_line/end_line で範囲を限定",
)
def read_file(path: str, start_line: str = None, end_line: str = None) -> str:
    """指定されたファイルの内容を読み込みます。行範囲指定で部分読み込み可能。"""
    target = Path(path)
    if not target.exists():
        return f"Error: ファイルが存在しません ({path})"
    if not target.is_file():
        return f"Error: 指定されたパスはファイルではありません ({path})"

    try:
        # UTF-8で読み込みを試みる
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            # SHIFT-JIS(cp932)での読み込みを試みる（Windows用フォールバック）
            text = target.read_text(encoding="cp932")
        except Exception as e:
            return f"Error: ファイルをテキストとして読み込めませんでした (エンコーディングエラー): {e}"
    except Exception as e:
        return f"Error: ファイルの読み込みに失敗しました: {e}"

    lines = text.splitlines()
    total_lines = len(lines)

    # 行範囲の解析（LLMが文字列で渡す場合があるため int 変換）
    sl = None
    el = None
    if start_line is not None:
        try:
            sl = int(start_line)
        except (ValueError, TypeError):
            return f"Error: start_line は整数で指定してください (受け取った値: {start_line})"
    if end_line is not None:
        try:
            el = int(end_line)
        except (ValueError, TypeError):
            return f"Error: end_line は整数で指定してください (受け取った値: {end_line})"

    if sl is not None or el is not None:
        # 行範囲指定モード
        sl = max(1, sl) if sl is not None else 1
        el = min(total_lines, el) if el is not None else total_lines
        if sl > total_lines:
            return f"Error: start_line({sl})がファイルの行数({total_lines})を超えています。"
        if el < sl:
            return f"Error: end_line({el})はstart_line({sl})より後である必要があります。"

        selected = lines[sl - 1 : el]
        # 行番号付きで返す
        numbered = [f"{i}: {line}" for i, line in enumerate(selected, start=sl)]
        header = f"[{os.path.basename(path)}] {sl}行目〜{el}行目 (全{total_lines}行)\n"
        return header + "\n".join(numbered)
    else:
        # 全体読み込み — サイズ/行数ヘッダを付与し、大きなファイルは行番号付きで返す。
        size_kb = len(text) / 1024
        header = f"[{os.path.basename(path)}] 全{total_lines}行 ({size_kb:.1f} KB)\n"

        # .py 大ファイル(500行超)は全文読込を抑制: 構造(get_code_outline) + 先頭50行のみ返す。
        # 事後の警告ではエージェントが read_file を連打してしまうため、全文を返さない構造化。
        if total_lines > 500 and target.suffix == ".py":
            head_n = 50
            head = "\n".join(f"{i}: {l}" for i, l in enumerate(lines[:head_n], 1))
            try:
                from code_tool import get_code_outline as _get_code_outline

                outline = _get_code_outline(path)
            except Exception:
                outline = "(構造抽出に失敗)"
            return (
                header + f"⚠ .pyファイル({total_lines}行)のため全文読込を省略。構造と先頭{head_n}行のみ表示。\n"
                "全体構造は `get_code_outline`、個別シンボルは `read_symbol`、"
                "範囲読込は `read_file(path, start_line, end_line)` で取得してください。\n\n"
                f"## 構造\n{outline}\n\n## 先頭{head_n}行\n{head}\n"
            )

        # 非.py または 小ファイルの大きいもの: 警告付きで全文
        if total_lines > 500:
            header = (
                f"⚠ このファイルは {total_lines} 行あります（目安500行超）。"
                "全体を読む前に `get_code_outline` で構造を把握し、"
                "`read_symbol` で必要なシンボルだけ読むことを推奨します。\n" + header
            )
        if len(text) > TOOL_RESULT_MAX_CHARS:
            # 大きなファイル: 行番号付き（切詰め時の正確な再取得ヒントのため）
            numbered = [f"{i}: {line}" for i, line in enumerate(lines, start=1)]
            return header + "\n".join(numbered)
        else:
            # 小さなファイル: そのまま（行番号のオーバーヘッドなし）
            return header + text


@register_tool(
    name="write_file",
    description="指定されたファイルにテキストを書き込みます。存在する場合は上書きされます。全体の書き換えは避けてください。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "書き込むファイルのパス"},
            "content": {"type": "string", "description": "書き込むテキスト内容"},
        },
        "required": ["path", "content"],
    },
    prompt_desc="write_file(path, content): pathにファイル書き込み（上書き注意）",
)
def write_file(path: str, content: str) -> str:
    """指定されたファイルにテキストを書き込みます。存在する場合は上書きされます。"""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Success: {path} にファイルを書き込みました。"
    except Exception as e:
        return f"Error: ファイルの書き込みに失敗しました: {e}"


@register_tool(
    name="append_to_file",
    description="指定されたファイルの末尾にテキストを追記します。ファイルが存在しない場合は新規作成されます。全体の書き換えは避けてください。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "追記するファイルのパス"},
            "content": {"type": "string", "description": "ファイル末尾に追記するテキスト内容"},
        },
        "required": ["path", "content"],
    },
    prompt_desc="append_to_file(path, content): pathのファイルの末尾にテキストを追記（ファイル新規作成も可能）",
)
def append_to_file(path: str, content: str) -> str:
    """指定されたファイルの末尾にテキストを追記します。ファイルが存在しない場合は新規作成されます。"""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Success: {path} に追記しました。"
    except Exception as e:
        return f"Error: ファイルの追記に失敗しました: {e}"


@register_tool(
    name="write_sections",
    description="セクション構造を指定して長文ドキュメントを生成・書き込みします。各セクションは独立した生成エンジンで高品質な本文を作成するため、メインのコンテキストを汚しません。レポート、仕様書、README、設計書などに最適。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "出力ファイルのパス"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {
                            "type": "string",
                            "description": "セクションの見出し（Markdownレベル含む、例: '## 概要'）",
                        },
                        "instruction": {"type": "string", "description": "このセクションで何を書くべきかの詳細な指示"},
                    },
                    "required": ["heading", "instruction"],
                },
                "description": "セクションの構造リスト。各要素に heading と instruction を含める。",
            },
            "context": {"type": "string", "description": "ドキュメント全体の文脈・目的・ターゲット読者（省略可）"},
        },
        "required": ["path", "sections"],
    },
    category="core",
    prompt_desc="write_sections(path, sections, context?): セクション構造を指定して長文ドキュメントを自動生成・書き込み",
)
def write_sections(path: str, sections: list, context: str = "") -> str:
    """セクション構造に基づいてドキュメントを生成する（実際の生成は engine.py でインターセプト）。"""
    # この実装はインターセプト用のダミー。engine.py の execute_tool() で実際の生成処理を行う。
    return "Error: write_sections はインターセプトされていません。engine.py を確認してください。"


def _compute_replace_lines_content(path: str, start_line, end_line, new_content: str) -> tuple[str | None, str]:
    """行範囲置換後のファイル全体内容を、実ファイルへ書き込まずに計算する（純粋関数）。

    replace_lines() 本体と shadow_verify.shadow_apply が共有するロジック
    （重複実装回避）。実際の適用と完全に同一の計算を行う。

    Returns:
        (new_full_content, error_message): 成功時 error_message は ""。
        失敗時 new_full_content は None（error_message に "Error: " 始まりの理由）。
    """
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None, f"Error: 対象ファイルが存在しません ({path})"
    try:
        # int変換（LLMが文字列で渡す場合の対策）
        start_line = int(start_line)
        end_line = int(end_line)

        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        if start_line < 1 or start_line > len(lines):
            return None, f"Error: 開始行({start_line})が範囲外です。ファイルは全{len(lines)}行です。"
        if end_line < start_line:
            return None, f"Error: 終了行({end_line})は開始行({start_line})より後である必要があります。"

        actual_end_line = min(end_line, len(lines))
        prefix = lines[: start_line - 1]
        suffix = lines[actual_end_line:]

        new_content_lines = new_content.splitlines(keepends=True)
        if new_content and not new_content.endswith("\n") and not new_content.endswith("\r\n"):
            new_content_lines[-1] = new_content_lines[-1] + "\n"

        new_lines = prefix + new_content_lines + suffix
        return "".join(new_lines), ""
    except Exception as e:
        return None, f"Error: 行の置換に失敗しました: {e}"


@register_tool(
    name="replace_lines",
    description="指定されたファイルの特定の行範囲(1オリジン)を新しい内容で置換します。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "置換対象のファイルパス"},
            "start_line": {"type": "integer", "description": "置換を開始する行番号 (1オリジン)"},
            "end_line": {"type": "integer", "description": "置換を終了する行番号 (1オリジン)"},
            "new_content": {"type": "string", "description": "置換後の新しいテキスト内容(改行を含む)"},
        },
        "required": ["path", "start_line", "end_line", "new_content"],
    },
    prompt_desc="replace_lines(path, start_line, end_line, new_content): ファイルの特定の行を置換",
)
def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """指定されたファイルの特定の行範囲(1オリジン)を新しい内容で置換します。"""
    new_full, err = _compute_replace_lines_content(path, start_line, end_line, new_content)
    if new_full is None:
        return err
    try:
        Path(path).write_text(new_full, encoding="utf-8")
    except Exception as e:
        return f"Error: 行の置換に失敗しました: {e}"
    try:
        sl, el = int(start_line), int(end_line)
    except (ValueError, TypeError):
        sl, el = start_line, end_line
    return f"Success: {path} の {sl}行目〜{el}行目を置換しました。"


def _build_search_hint(search_lines: list[str], content_lines: list[str], max_hints: int = 3) -> str:
    """search_block の先頭行に近いファイル内の行をヒントとして生成する。

    AI が search_block のインデントや表記を間違えた場合に、
    「ファイル内の実際の内容」を提示して自己修正を促す。
    """
    if not search_lines:
        return ""

    first_line_stripped = search_lines[0].strip()
    if not first_line_stripped:
        return ""

    hints = []
    seen = set()
    for i, line in enumerate(content_lines):
        stripped = line.strip()
        # 先頭行と類似（部分一致）する行を収集
        if first_line_stripped and first_line_stripped in stripped and stripped not in seen:
            hints.append(f"  行{i + 1}: {line}")
            seen.add(stripped)
            if len(hints) >= max_hints:
                break

    # 部分一致で見つからなかった場合: 先頭数文字が一致する行を探す
    if not hints and len(first_line_stripped) >= 10:
        prefix = first_line_stripped[:15]
        for i, line in enumerate(content_lines):
            stripped = line.strip()
            if prefix in stripped and stripped not in seen:
                hints.append(f"  行{i + 1}: {line}")
                seen.add(stripped)
                if len(hints) >= max_hints:
                    break

    if hints:
        return "\n【ヒント: ファイル内の類似行】\n" + "\n".join(hints)
    return ""


def _fuzzy_apply(content: str, search_block: str, replace_block: str, threshold: float = None):
    """完全一致が失敗した後に search_block を content へファジーマッチする（厳格モード）。

    レイヤード（安全な順）:
      L2 空白正規化ウィンドウ一致: 各行 strip() 後、len(search_lines) 幅の窓が
         正規化状態で完全一致する位置が「一意」なら採用
      L3 difflib ファジー: 先頭行をアンカーに候補を絞り、窓の ratio() が閾値以上
         かつ2位候補と十分離れていれば採用

    厳格: 窓は常に len(search_lines) 幅（行数違い/略記は対象外）。スパン拡張なし。
    曖昧（複数候補が同点、あるいは2位も閾値ギリギリ）なら安全のため失敗する。

    Returns:
        (new_content, method): 成功。method は "normalized"/"fuzzy(0.xx)"。
        (None, None): 適用不可（呼び出し元でヒント表示）。
    """
    import difflib

    if threshold is None:
        from config import FUZZY_MATCH_THRESHOLD as _THR

        threshold = _THR

    if not search_block:
        return None, None

    search_lines = search_block.splitlines()
    file_lines = content.splitlines()
    n = len(search_lines)
    if n == 0 or n > len(file_lines):
        return None, None

    norm_search = [s.strip() for s in search_lines]
    norm_file = [f.strip() for f in file_lines]

    # L2: 空白正規化ウィンドウ一致（一意位置のみ）
    l2 = [i for i in range(len(file_lines) - n + 1) if norm_file[i : i + n] == norm_search]
    if len(l2) == 1:
        actual = "\n".join(file_lines[l2[0] : l2[0] + n])
        new_content = content.replace(actual, replace_block, 1)
        if new_content != content:
            return new_content, "normalized"
    # L2 で 0件 or 複数件 → L3 でより高い基準で絞り込めるか試す

    # L3: difflib ファジー（先頭行アンカーで候補を絞り込み、計算量を抑える）
    first = norm_search[0]
    scored = []
    for i in range(len(file_lines) - n + 1):
        if difflib.SequenceMatcher(None, norm_file[i], first).ratio() < 0.70:
            continue  # 先頭行が全く似ない位置はスキップ
        r = difflib.SequenceMatcher(None, norm_search, norm_file[i : i + n]).ratio()
        scored.append((r, i))
    scored.sort(reverse=True)
    if scored and scored[0][0] >= threshold:
        best_r, best_i = scored[0]
        second_r = scored[1][0] if len(scored) > 1 else 0.0
        # 一意性: 2位候補と5pt以上離れている、または2位が閾値未満なら採用
        if best_r - second_r >= 0.05 or second_r < threshold:
            actual = "\n".join(file_lines[best_i : best_i + n])
            new_content = content.replace(actual, replace_block, 1)
            if new_content != content:
                return new_content, f"fuzzy({best_r:.2f})"

    return None, None


def _compute_search_and_replace_content(path: str, search_block: str, replace_block: str) -> dict:
    """search_and_replace の適用後内容を、実ファイルへ書き込まずに計算する（純粋関数）。

    search_and_replace() 本体と shadow_verify.shadow_apply が共有するロジック
    （L1完全一致→L2正規化ウィンドウ→L3 difflib ファジーの _fuzzy_apply を再利用・重複実装回避）。

    Returns:
        成功時: {"ok": True, "content": <適用後の全文>, "method": "exact"|"normalized"|"fuzzy(0.xx)"}
        失敗時: {"ok": False, "error": "<'Error: ' 始まりの理由文字列>"}
    """
    target = Path(path)
    if not target.exists() or not target.is_file():
        return {"ok": False, "error": f"Error: 対象ファイルが存在しません ({path})"}

    if not search_block:
        return {"ok": False, "error": "Error: search_block が空です。置換対象のコードを指定してください。"}

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = target.read_text(encoding="cp932")
        except Exception as e:
            return {"ok": False, "error": f"Error: ファイルの読み込みに失敗しました: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Error: ファイルの読み込みに失敗しました: {e}"}

    count = content.count(search_block)
    if count == 1:
        new_content = content.replace(search_block, replace_block, 1)
        return {"ok": True, "content": new_content, "method": "exact"}
    if count > 1:
        # 複数マッチ: より長いコンテキストを含めるよう誘導
        return {"ok": False, "error": (
            f"Error: search_block がファイル内で {count} 箇所にマッチしました。"
            f"一意に特定できるよう、前後の行を含めて search_block を長くしてください。"
        )}

    # count == 0: ファジーマッチ（厳格モード）で再挑戦
    new_content, method = _fuzzy_apply(content, search_block, replace_block)
    if new_content is not None:
        return {"ok": True, "content": new_content, "method": method}

    # ファジーマッチも失敗 → 近接行ヒントで自己修正を促す
    search_lines = search_block.splitlines()
    hint_lines = _build_search_hint(search_lines, content.splitlines())
    return {"ok": False, "error": (
        f"Error: search_block がファイル内に見つかりませんでした。"
        f"※ read_file で対象箇所を確認し、対象ブロックの全行を正確にコピーして再実行してください"
        f"（多少のインデント差・表記揺れは自動補正しますが、行の省略は不可）。"
        f"{hint_lines}"
    )}


@register_tool(
    name="search_and_replace",
    description="既存ファイルの一部を安全に置換します。行番号の代わりに、ファイル内の正確な既存コードブロック(search_block)と新しいコードブロック(replace_block)を指定します。既存ファイルの修正には必ずこのツールを使用し、write_file による全体上書きは避けてください。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "対象ファイルのパス"},
            "search_block": {
                "type": "string",
                "description": "置換対象となる既存のコード（read_fileで確認してから指定）。多少のインデント差・表記揺れは自動補正するが、対象ブロックの全行を省略なく含めること",
            },
            "replace_block": {"type": "string", "description": "置換後の新しいコード"},
        },
        "required": ["path", "search_block", "replace_block"],
    },
    prompt_desc="search_and_replace(path, search_block, replace_block): 既存ファイルの一部を安全に置換。行番号不要・ファジーマッチ対応（インデント差/表記揺れを自動補正・全行必須）",
)
def search_and_replace(path: str, search_block: str, replace_block: str) -> str:
    """既存ファイルの一部を正確な文字列マッチで安全に置換する。"""
    outcome = _compute_search_and_replace_content(path, search_block, replace_block)
    if not outcome["ok"]:
        return outcome["error"]
    try:
        Path(path).write_text(outcome["content"], encoding="utf-8")
    except Exception as e:
        return f"Error: ファイルの書き込みに失敗しました: {e}"
    if outcome["method"] == "exact":
        return f"Success: {path} の該当箇所を置換しました。"
    return f"Success: {path} の該当箇所を置換しました。（{outcome['method']} マッチ: インデント差・表記揺れを自動補正）"


@register_tool(
    name="move_file",
    description="ファイルまたはフォルダを移動、あるいは名前を変更します。",
    schema={
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "移動元のパス"},
            "dst": {"type": "string", "description": "移動先のパス"},
        },
        "required": ["src", "dst"],
    },
    prompt_desc="move_file(src, dst): srcからdstへ移動・リネーム",
)
def move_file(src: str, dst: str) -> str:
    """ファイルを移動、またはリネームします。"""
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        return f"Error: 移動元のファイルが存在しません ({src})"
    if dst_path.exists() and dst_path.is_file():
        return f"Error: 移動先に同名のファイルが既に存在します ({dst})"
    try:
        shutil.move(str(src_path), str(dst_path))
        return f"Success: {src} を {dst} に移動しました。"
    except Exception as e:
        return f"Error: ファイルの移動に失敗しました: {e}"


@register_tool(
    name="make_directory",
    description="新しいディレクトリを作成します。途中の階層も必要なら自動的に作成されます。",
    schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "作成するディレクトリのパス"}},
        "required": ["path"],
    },
    prompt_desc="make_directory(path): pathにディレクトリ作成",
)
def make_directory(path: str) -> str:
    """ディレクトリを作成します（親ディレクトリも含む）。"""
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        return f"Success: フォルダを作成しました ({path})"
    except Exception as e:
        return f"Error: フォルダの作成に失敗しました: {e}"


@register_tool(
    name="delete_file",
    description="指定されたファイルを削除します。",
    schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "削除するファイルのパス"}},
        "required": ["path"],
    },
    prompt_desc="delete_file(path): pathのファイルを削除",
)
def delete_file(path: str) -> str:
    """ファイルを削除します（ディレクトリは削除できません）。"""
    target = Path(path)
    if not target.exists():
        return f"Error: 削除対象のファイルが存在しません ({path})"
    if target.is_dir():
        return f"Error: これはフォルダです。ファイルの削除に使用してください ({path})"
    try:
        target.unlink()
        return f"Success: {path} を削除しました。"
    except Exception as e:
        return f"Error: ファイルの削除に失敗しました: {e}"


def _decode_console_bytes(b: bytes, prefer_utf8: bool = False) -> str:
    """コンソール出力のバイト列をエンコーディングを考慮してデコードします。

    prefer_utf8=True の場合は UTF-8 を優先して試行します（子プロセスを PYTHONUTF8=1
    で起動した場合など、出力が UTF-8 と分かっているときの文字化けを防ぐ）。
    """
    if not b:
        return ""
    if b"\x00" in b:
        try:
            return b.decode("utf-16le")
        except UnicodeDecodeError:
            pass
    if prefer_utf8:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            pass
    try:
        return b.decode("cp932")
    except UnicodeDecodeError:
        pass
    return b.decode("utf-8", errors="replace")


@register_tool(
    name="run_command",
    description=(
        "OSのシェル（WindowsならPowerShell、Linuxならbash）で任意のコマンドを実行します。"
        "ディレクトリを移動する場合は command 内で cd を連結せず working_directory を使ってください"
        "（WindowsのPowerShellでは && が使えずエラーになるため）。"
        "入力を待つ対話的プログラム（input() 等）には input で終了指示等を渡してください。"
    ),
    schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "実行するシェルコマンド（cd 連結は避け working_directory を使用）"},
            "working_directory": {"type": "string", "description": "コマンドを実行する作業ディレクトリのパス（省略時は現在のディレクトリ）"},
            "input": {"type": "string", "description": "コマンドの標準入力に渡すテキスト。対話的プログラム（input() を待つ等）に自動応答する際に使用"},
            "timeout": {"type": "integer", "description": "実行のタイムアウト秒（既定30）。超えると強制終了"},
        },
        "required": ["command"],
    },
    prompt_desc="run_command(command, working_directory?, input?, timeout?): 任意のOSコマンドを実行",
)
def run_command(command: str, working_directory: str = None, input: str = None, timeout: int = 30) -> str:  # noqa: A002
    """OSに応じて適切なシェルを用いてコマンドを実行します。

    working_directory で作業ディレクトリを指定（cd 連結不要・Windowsの && エラー回避）、
    input で標準入力に自動応答（対話プログラムの input() 待ちを解消）、
    timeout で実行上限を制御します。長文出力の切り詰めにも対応。
    """
    os_name = platform.system()
    if os_name == "Windows":
        cmd_args = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", command]
    else:
        cmd_args = ["bash", "-c", command]

    try:
        # input を渡す場合、子プロセス(Python等)の stdin を UTF-8 に強制し、Windows cp932
        # での文字化け（"終了" 等が正しく認識されずループから抜けない）を防ぐ。
        # 非Pythonプロセスには PYTHONUTF8 は無視されるため副作用がない。
        env = {**os.environ, "PYTHONUTF8": "1"} if input is not None else None
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            timeout=timeout,
            cwd=working_directory or None,
            input=input.encode("utf-8") if input else None,
            env=env,
        )
        stdout_txt = _decode_console_bytes(result.stdout, prefer_utf8=env is not None).strip()
        stderr_txt = _decode_console_bytes(result.stderr, prefer_utf8=env is not None).strip()

        def truncate_output(text: str, max_len: int = 4000) -> str:
            if len(text) <= max_len:
                return text
            half = (max_len - 100) // 2
            return f"{text[:half]}\n\n... (出力が長いため {len(text)} 文字から {max_len} 文字に省略しました) ...\n\n{text[-half:]}"

        stdout_txt = truncate_output(stdout_txt)
        stderr_txt = truncate_output(stderr_txt)

        if result.returncode != 0:
            return f"Error ({result.returncode}):\n{stderr_txt}\nOutput:\n{stdout_txt}"
        if not stdout_txt and not stderr_txt:
            return "Success: (出力なし)"
        return stdout_txt if stdout_txt else stderr_txt
    except subprocess.TimeoutExpired:
        return f"Execution Timeout: コマンドの実行が{timeout}秒を超えたため強制終了しました。"
    except Exception as e:
        return f"Execution Failed: {e}"


@register_tool(
    name="update_core_memory",
    description="AI自身のタスク進捗や計画を管理する Core Memory (CORE_MEMORY.md) を上書き更新します。タスクが一つ終わるごとに呼び出してください。内容は必ず【計画(完了/未完了のリスト)】【現在の焦点】【判明事項】のフォーマットに従ってください。",
    schema={
        "type": "object",
        "properties": {"content": {"type": "string", "description": "Core Memoryの新しい全体内容(Markdownで記述)"}},
        "required": ["content"],
    },
    prompt_desc="update_core_memory(content): AI自身のタスクリストや状態を記録する Core Memory 全体を更新",
)
def update_core_memory(content: str) -> str:
    """Core Memory を上書き更新します。"""
    try:
        target = Path(get_data_path("CORE_MEMORY.md"))
        target.write_text(content, encoding="utf-8")
        return "Success: Core Memory を更新しました。"
    except Exception as e:
        return f"Error: Core Memory の更新に失敗しました: {e}"


@register_tool(
    name="grep_search",
    description="ripgrep(rg)を用いた超高速なファイル検索。ファイルタイプフィルタや、マッチ箇所の前後行（コンテキスト）の取得が可能です。単なる文字列検索にも正規表現にも対応しています。",
    schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "検索する文字列または正規表現"},
            "path": {"type": "string", "description": "検索対象のディレクトリまたはファイルのパス (デフォルト: .)"},
            "is_regex": {"type": "boolean", "description": "正規表現として扱うか (デフォルト: false)"},
            "file_extensions": {
                "type": "string",
                "description": "検索対象の拡張子をカンマ区切りで指定（例: 'py,js,md'）。省略時は全ファイル",
            },
            "context_lines": {
                "type": "integer",
                "description": "マッチした行の前後何行を取得するか (デフォルト: 2, 0でマッチ行のみ)",
            },
        },
        "required": ["pattern"],
    },
    prompt_desc="grep_search(pattern, path?, is_regex?, file_extensions?, context_lines?): ripgrepによる超高速検索。マッチ箇所の前後行も同時に取得可能",
)
def grep_search(
    pattern: str, path: str = ".", is_regex: bool = False, file_extensions: str = None, context_lines: int = 2
) -> str:
    """ネイティブのripgrep(Linuxはgrep)を呼び出し、高速検索と前後の文脈を同時に取得する。"""

    target = Path(path)
    if not target.exists():
        return f"Error: 対象パスが存在しません ({path})"

    use_regex = str(is_regex).lower() in ("true", "1", "yes")
    try:
        ctx_lines = min(max(int(context_lines), 0), 10)
    except (ValueError, TypeError):
        ctx_lines = 2

    # Windows: プロジェクトルートの rg.exe を優先、Linux/Mac: PATH の rg を優先、なければ grep
    if platform.system() == "Windows":
        rg_path = Path(get_bundled_path("rg.exe"))
        cmd_base = [str(rg_path)] if rg_path.exists() else None
        if cmd_base is None:
            return "Error: 'rg.exe' がプロジェクトフォルダに見つかりません。"
        encoding = "utf-8"  # rg.exe は常に UTF-8 出力
    else:
        cmd_base = None
        try:
            result = subprocess.run(["rg", "--version"], capture_output=True)
            if result.returncode == 0:
                cmd_base = ["rg"]
        except FileNotFoundError:
            pass
        if cmd_base is None:
            cmd_base = ["grep", "-rn", "--color=never"]
        encoding = locale.getpreferredencoding(False)

    cmd = list(cmd_base)

    if "rg" in cmd[0]:
        cmd.extend(["-n", "-H", "--heading", "--color=never", "--max-columns=300"])
        if not use_regex:
            cmd.append("-F")
        if ctx_lines > 0:
            cmd.extend(["-C", str(ctx_lines)])
        if file_extensions:
            for ext in file_extensions.split(","):
                ext = ext.strip().lstrip(".")
                if ext:
                    cmd.extend(["-g", f"*.{ext}"])
        cmd.extend(["-e", pattern, str(target)])
    else:
        # grep フォールバック
        if not use_regex:
            cmd.append("-F")
        if ctx_lines > 0:
            cmd.extend(["-C", str(ctx_lines)])
        if file_extensions:
            for ext in file_extensions.split(","):
                ext = ext.strip().lstrip(".")
                if ext:
                    cmd.extend(["--include", f"*.{ext}"])
        cmd.extend(["--", pattern, str(target)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding=encoding, errors="replace")

        if cmd[0].endswith("grep"):
            # grep: exit 0 = マッチあり, 1 = なし, 2 = エラー
            if result.returncode == 0:
                pass
            elif result.returncode == 1:
                return f"No matches found for pattern '{pattern}' in {path}."
            else:
                return f"Error: {result.stderr.strip()}"
        else:
            # rg: exit 0 = マッチあり, 1 = なし, 2+ = エラー
            if result.returncode == 1:
                return f"No matches found for pattern '{pattern}' in {path}."
            elif result.returncode >= 2:
                return f"Error running ripgrep: {result.stderr.strip()}"

        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern '{pattern}' in {path}."

        max_chars = 10000
        if len(output) > max_chars:
            return (
                output[:max_chars]
                + f"\n\n... (出力が長すぎるため {max_chars} 文字で切り捨てました。file_extensions や pattern で条件を絞ってください)"
            )
        return output

    except FileNotFoundError:
        return "Error: 検索コマンドが見つかりません。Windowsでは rg.exe をプロジェクトフォルダに配置してください。"
    except Exception as e:
        return f"Error: 検索中に予期せぬエラーが発生しました: {e}"

    # ============================
    # 拡張ツール定義
    # ============================


@register_tool(
    name="view_tree",
    description="指定されたディレクトリの階層構造を省略表記付きのツリーテキストで返します。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "対象ディレクトリパス"},
            "max_depth": {"type": "integer", "description": "ツリーの深さ (デフォルト3)"},
        },
        "required": ["path"],
    },
    category="extended",
    prompt_desc="view_tree(path, max_depth): ディレクトリのツリー構造を出力する",
)
def view_tree(path: str, max_depth: int = 3) -> str:
    """ディレクトリ構造をツリー形式で出力します。"""
    target = Path(path)
    if not target.exists() or not target.is_dir():
        return f"Error: ディレクトリが存在しません ({path})"

    # max_depthのint変換
    try:
        max_depth = int(max_depth)
    except (ValueError, TypeError):
        max_depth = 3

    lines = []

    def walk_tree(current_dir: Path, prefix: str = "", depth: int = 1):
        if depth > max_depth:
            return
        try:
            ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}
            items = sorted(
                [x for x in current_dir.iterdir() if x.name not in ignore_dirs], key=lambda x: (not x.is_dir(), x.name)
            )
        except Exception as e:
            lines.append(f"{prefix}[Error reading dir: {e}]")
            return
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                extension = "    " if is_last else "\u2502   "
                walk_tree(item, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{item.name}")

    lines.append(f"{target.name}/")
    walk_tree(target)
    result = "\n".join(lines)
    if len(result) > 4000:
        return result[:4000] + "\n... (省略)"
    return result


@register_tool(
    name="inspect_tool",
    description="拡張ツールの詳細な使い方や引数スキーマを取得します。",
    schema={
        "type": "object",
        "properties": {"tool_name": {"type": "string", "description": "仕様を確認したい拡張ツールの名前"}},
        "required": ["tool_name"],
    },
    prompt_desc="inspect_tool(tool_name): 拡張ツールの詳細な使い方や引数スキーマを取得する",
)
def inspect_tool(tool_name: str) -> str:
    """指定されたツールの詳細なマニュアルを取得します。"""
    entry = TOOL_REGISTRY.get(tool_name)
    if entry:
        schema_info = entry["schema"]
        props = schema_info.get("properties", {})
        required = schema_info.get("required", [])

        lines = [f"【{tool_name} のマニュアル】"]
        lines.append(f"説明: {entry['description']}")
        lines.append(f"カテゴリ: {entry['category']}")
        lines.append("引数:")
        for arg_name, arg_info in props.items():
            req_mark = " (必須)" if arg_name in required else " (省略可)"
            lines.append(f"  - {arg_name}: {arg_info.get('description', '')}{req_mark}")
        return "\n".join(lines)
    else:
        return f"Error: ツール '{tool_name}' のマニュアルは見つかりません。"

    # ============================
    # インターセプト対象のスタブツール（CLI/GUI側でハンドリングされる）
    # ============================


@register_tool(
    name="view_image",
    description="画像の内容を視覚的に解析・確認する（拡張ツール）。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "画像ファイルのパス"},
            "analysis_prompt": {"type": "string", "description": "画像をどういう観点で見るかのプロンプト"},
        },
        "required": ["path"],
    },
    category="extended",
    prompt_desc="view_image(path, analysis_prompt): 画像の内容を視覚的に解析・確認する",
)
def view_image(path: str = "", analysis_prompt: str = None) -> str:
    """CLI/GUIのインターセプタでハンドリングされるスタブ。直接呼ばれた場合のフォールバック。"""
    return "Error: 現在起動しているモデルは「テキスト専用モード」のため、画像解析ツール 'view_image' はサポートされていません。ユーザーに画像解析が実行できない旨を報告してください。"


@register_tool(
    name="run_async_test",
    description="長時間実行コマンド（テスト、ビルド等）を別コンソールウィンドウで非同期実行し、プロセスIDとログパスを返します。",
    schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "実行するコマンド（例: pytest tests/test_main.py -v）"},
            "log_file": {
                "type": "string",
                "description": "ログ出力先パス（省略時は .pixie_notes/logs/async_<timestamp>.log）",
            },
        },
        "required": ["command"],
    },
    prompt_desc="run_async_test(command, log_file?): 長時間コマンドを別ウィンドウで非同期実行し、PIDとログパスを返す",
)
def run_async_test(command: str, log_file: str = "") -> str:
    """コマンドを別コンソールで非同期実行し、PIDとログパスを返す。"""
    os_name = platform.system()
    import time

    # ログファイルの決定
    if not log_file:
        logs_dir = get_data_path(".pixie_notes/logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"async_{timestamp}.log")

    try:
        log_f = open(log_file, "w", encoding="utf-8")
    except Exception as e:
        return f"Error: ログファイル {log_file} を開けません: {e}"

    try:
        if os_name == "Windows":
            # Windows: 新規コンソールで実行
            cmd_array = ["cmd", "/c", f"{command} 2>&1"]
            process = subprocess.Popen(
                cmd_array, stdout=log_f, stderr=log_f, creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # Unix: バックグラウンド実行
            cmd_array = ["sh", "-c", f"{command} 2>&1"]
            process = subprocess.Popen(cmd_array, stdout=log_f, stderr=log_f, start_new_session=True)

        pid = process.pid
        log_f.close()  # 親プロセスのハンドルを閉じる（子プロセスは fd を継承して書き続ける）
        return (
            f"Success: 非同期実行を開始しました。\nPID: {pid}\nLog: {log_file}\n※ poll_process で進捗確認してください。"
        )

    except Exception as e:
        log_f.close()
        return f"Error: コマンド実行に失敗しました: {e}"


@register_tool(
    name="poll_process",
    description="指定したPIDのプロセスの生存確認と、ログファイルの末尾（最新30行）を返します。",
    schema={
        "type": "object",
        "properties": {
            "pid": {"type": "integer", "description": "監視対象のプロセスID"},
            "log_file": {"type": "string", "description": "ログファイルパス"},
        },
        "required": ["pid", "log_file"],
    },
    prompt_desc="poll_process(pid, log_file): プロセス生存確認とログ末尾30行を取得",
)
def poll_process(pid: int, log_file: str) -> str:
    """プロセスの生存確認とログ末尾を取得する。"""
    os_name = platform.system()

    # === プロセス生存確認 ===
    alive = False
    try:
        if os_name == "Windows":
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
            alive = str(pid) in result.stdout
        else:
            os.kill(pid, 0)  # シグナル送信（終了しない）
            alive = True
    except Exception:
        alive = False

    status = "実行中" if alive else "終了"

    # === ログ末尾取得 ===
    tail_lines = []
    if os.path.exists(log_file):
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                tail_lines = lines[-30:] if len(lines) > 30 else lines
        except Exception as e:
            tail_lines = [f"[Error] ログ読み込み失敗: {e}"]
    else:
        tail_lines = ["[ログファイルがまだ作成されていません]"]

    tail_output = "".join(tail_lines).strip()

    return f"""プロセス状態: {status}
PID: {pid}
ログ末尾（最新30行）:
---
{tail_output}
---"""


@register_tool(
    name="kill_process",
    description="指定したPIDのプロセスを強制終了します。",
    schema={
        "type": "object",
        "properties": {"pid": {"type": "integer", "description": "終了するプロセスID"}},
        "required": ["pid"],
    },
    prompt_desc="kill_process(pid): プロセスを強制終了",
)
def kill_process(pid: int) -> str:
    """プロセスを強制終了する。"""
    os_name = platform.system()

    try:
        if os_name == "Windows":
            result = subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True)
            if result.returncode == 0:
                return f"Success: PID {pid} を終了しました。"
            else:
                return f"Error: 終了に失敗しました: {result.stderr}"
        else:
            import signal

            os.kill(pid, signal.SIGKILL)
            return f"Success: PID {pid} を終了しました。"

    except ProcessLookupError:
        return f"Error: PID {pid} は既に終了しています。"
    except Exception as e:
        return f"Error: 終了に失敗しました: {e}"


@register_tool(
    name="analyze_file",
    description="長文テキストファイルを裏で別のLLMに部分解析・要約させ、指定された分析結果のみを引出します。メインコンテキストを節約します。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "解析対象のファイルパス"},
            "analysis_prompt": {
                "type": "string",
                "description": "要約・抽出したい観点や指示のプロンプト(日本語で思考および出力するように指示してください)",
            },
        },
        "required": ["path"],
    },
    prompt_desc="analyze_file(path, analysis_prompt): 長大ファイルの調査用ツール。裏で別AIに要約させ結果のみ返す",
)
def analyze_file(path: str = "", analysis_prompt: str = None) -> str:
    """CLI/GUIのインターセプタでハンドリングされるスタブ。"""
    return "Error: この環境では 'analyze_file' がインターセプタにハンドリングされませんでした。"

    # ============================
    # エージェント状態ツール群（統合ステートボード）
    # ============================


@register_tool(
    name="set_goal",
    description="ユーザーの目標を設定します。セッションの開始時やタスクが変更された場合に使用してください。一度設定すると原則変更しません。",
    schema={
        "type": "object",
        "properties": {"goal": {"type": "string", "description": "ユーザーの目標や要求（簡潔に）"}},
        "required": ["goal"],
    },
    category="extended",
    prompt_desc="set_goal(goal): ユーザーの目標を設定（セッション開始時）",
)
def set_goal(goal: str) -> str:
    """ユーザーの目標を設定する。"""
    sb = registry._state_board
    if sb is None:
        return "Error: ステートボードが初期化されていません。"
    try:
        sb.set_goal(goal)
        return f"Success: 目標を設定しました: {goal[:100]}"
    except Exception as e:
        return f"Error: 目標の設定に失敗しました: {e}"


@register_tool(
    name="update_state",
    description="エージェントの作業状態を一括更新します。ツール実行結果から得られた結論や、次のアクション計画を記録します。引数はすべて省略可能で、渡されたものだけが更新されます。",
    schema={
        "type": "object",
        "properties": {
            "current_step": {"type": "string", "description": "現在実行中の作業内容（上書き）"},
            "next_to_do": {"type": "string", "description": "次にやることのリスト（改行区切り、上書き）"},
            "found_knowledge": {"type": "string", "description": "判明した結論（key=value形式、改行区切り、追記）"},
            "errors": {
                "type": "string",
                "description": "発生中のエラー（改行区切り、上書き。解決したエラーはリストから外してください）",
            },
        },
        "required": [],
    },
    category="extended",
    prompt_desc="update_state(current_step?, next_to_do?, found_knowledge?, errors?): エージェントの状態を一括更新（全引数省略可・状態記録専用・実行や読取の代わりにならない）。進捗/メモ/記録/次にやること 系に",
)
def update_state(
    current_step: str = None, next_to_do: str = None, found_knowledge: str = None, errors: str = None
) -> str:
    """エージェントの状態を一括更新する。"""
    sb = registry._state_board
    if sb is None:
        return "Error: ステートボードが初期化されていません。"
    try:
        sb.update(
            current_step=current_step,
            next_to_do=next_to_do,
            found_knowledge=found_knowledge,
            errors=errors,
        )
        parts = []
        if current_step:
            parts.append(f"実行中: {current_step[:50]}")
        if next_to_do:
            count = len([l for l in next_to_do.strip().splitlines() if l.strip()])
            parts.append(f"次のステップ: {count}件")
        if found_knowledge:
            count = len([l for l in found_knowledge.strip().splitlines() if l.strip()])
            parts.append(f"知識: {count}件追記")
        return f"Success: 状態を更新しました ({', '.join(parts) if parts else '空の更新'})"
    except Exception as e:
        return f"Error: 状態の更新に失敗しました: {e}"


@register_tool(
    name="query_whiteboard",
    description="エージェントのステートボードから情報を検索・取得します。タスク状態、判明した事実、ファイル解析サマリーなどを検索できます。",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "検索クエリ（キーワードをスペース区切りで指定）"},
        },
        "required": ["query"],
    },
    category="extended",
    prompt_desc="query_whiteboard(query): ステートボードからタスク・事実・ファイル情報を検索",
)
def query_whiteboard(query: str, category: str = "all") -> str:
    """ステートボードから情報を検索する。"""
    sb = registry._state_board
    if sb is None:
        return "Error: ステートボードが初期化されていません。"
    try:
        return sb.query(query)
    except Exception as e:
        return f"Error: ステートボード検索に失敗しました: {e}"


# ============================
# 委譲サブエージェント (delegate_research)
# ============================
@register_tool(
    name="delegate_research",
    description=(
        "メインコンテキストを汚さずに、独立したサブエージェントに調査・解析を委譲する。"
        "サブエージェントは読み取り専用ツール(grep/read/outline/symbol等)だけで調査し、"
        "結論だけを返す。プロジェクト全体の説明・構造把握・原因特定・設計検証など、"
        "複数ファイルをまたぐ理解や解析に最適。"
        " キーワード: 調べ 調査 理由 探す なぜ 説明 詳細 全体 構造 把握 解析 概要 理解 教えて まとめて 全体像"
    ),
    schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "サブエージェントに調査させる質問・指示。具体的に。"
                "例: 'engine.py の execute_tool が analyze_file をインターセプトする仕組みと理由を調査せよ'",
            },
            "file_hints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "調査を始めるファイル/ディレクトリのヒント(省略可)。サブエージェントが最初に参照する。",
            },
            "focus": {
                "type": "string",
                "description": "調査の焦点・制約(省略可)。例: '関数シグネチャ変更が他の呼び出し元に与える影響のみ'",
            },
            "max_steps": {
                "type": "integer",
                "description": "サブエージェントの最大調査ステップ数(省略時6)。大きいほど深いが遅い。",
            },
        },
        "required": ["question"],
    },
    category="core",
    prompt_desc=(
        "delegate_research(question): 独立サブエージェントに調査を委譲し結論だけ取得(コンテキスト節約)。"
        " 調べ/調査/理由/探す/説明/詳細/全体/構造/把握/解析/概要/理解/教えて 系に。複数ファイル横断・プロジェクト全体像・根本原因の調査で活用。"
    ),
)
def delegate_research(question: str, file_hints: list = None, focus: str = None, max_steps: int = None) -> str:
    """registry 上のスタブ。実際の処理は engine._execute_delegate_research が
    execute_tool のインターセプトで実行する(context.llm が必要なため)。
    直接呼ばれた場合は engine 経由を促すメッセージを返す。"""
    return (
        "Error: delegate_research はエンジン経由でのみ実行されます "
        "(execute_tool のインターセプトを通る必要があります)。"
    )

    # ============================
    # JIT ツールスコアリング
    # ============================


@register_tool(
    name="run_python",
    description=(
        "Pythonコードをサンドボックス環境で実行する。input() で入力待ちになった場合、"
        "エージェントが自動で入力値を生成して継続する（インタラクティブな対話プログラムも実行可能）。"
        "外部副作用に注意（一時ディレクトリ・envサニタイズ・タイムアウト付き）。"
        " キーワード: 実行 走らせる 動かす python コード 試す 確認 テスト プログラム スクリプト"
    ),
    schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "実行するPythonコード。input() を含む対話的なコードも可。"
                "プロンプト検出時にエージェントが自動入力を生成して継続する。",
            },
            "stdin_seed": {
                "type": "string",
                "description": "最初の input() に事前入力として送る固定値(省略可)。"
                "プロンプト検出時はLLM生成より優先される安全網。",
            },
            "max_inputs": {
                "type": "integer",
                "description": "LLM自動入力の上限回数(省略時6)。超過時は実行を停止する。",
            },
            "timeout": {
                "type": "integer",
                "description": "総タイムアウト秒(省略時30)。",
            },
        },
        "required": ["code"],
    },
    category="extended",
    prompt_desc=(
        "run_python(code): Pythonをサンドボックス実行。input()検出時にLLMが自動入力を生成して継続。"
        " 実行/走らせる/動かす/python/コード/試す/確認/テスト/プログラム/スクリプト 系に。"
    ),
)
def run_python(code: str, stdin_seed: str = None, max_inputs: int = None, timeout: int = None) -> str:
    """registry 上のスタブ。実際の処理は engine._execute_run_python が
    execute_tool のインターセプトで実行する(context.llm が必要なため)。
    直接呼ばれた場合は engine 経由を促すメッセージを返す。"""
    return (
        "Error: run_python はエンジン経由でのみ実行されます "
        "(execute_tool のインターセプトを通る必要があります)。"
    )


def score_tools(user_input: str, top_n: int = 5) -> list[str]:
    """ユーザー入力に対してキーワードスコアリングを行い、推薦ツールを返す。

    スコアリング基準:
    1. ツール名が入力に直接含まれる (+5.0)
    2. 入力トークンとツール説明文のトークンオーバーラップ (+1.0/個)
    3. 入力トークンがツール説明文の部分文字列 (+0.5/個)

    Args:
        user_input: ユーザーの入力テキスト
        top_n: スコア上位から取得するツール数（ALWAYS_RECOMMEND に追加）

    Returns:
        推薦ツール名リスト（ALWAYS_RECOMMEND + スコア上位 top_n）
    """
    input_lower = user_input.lower()

    # トークン化: 英数字 + 日本語bi-gram
    input_tokens = set(re.findall(r"[\w]+", input_lower))

    # 日本語bi-gram抽出（文字bigramで単語境界を仮定）
    japanese_chars = re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", input_lower)
    japanese_tokens = set()
    for i in range(len(japanese_chars) - 1):
        japanese_tokens.add(japanese_chars[i] + japanese_chars[i + 1])

    all_input_tokens = input_tokens | japanese_tokens

    scores = []
    for name, entry in TOOL_REGISTRY.items():
        score = 0.0

        # 検索対象テキスト（prompt_desc + description）
        searchable_text = (entry.get("prompt_desc", "") + " " + entry.get("description", "")).lower()

        # 1. ツール名が入力に直接含まれる
        if name.lower() in input_lower:
            score += 5.0

        # 2. トークンオーバーラップ（説明文のトークンと入力トークンの一致数）
        desc_tokens = set(re.findall(r"[\w]+", searchable_text))
        desc_chars = re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", searchable_text)
        desc_japanese = set()
        for i in range(len(desc_chars) - 1):
            desc_japanese.add(desc_chars[i] + desc_chars[i + 1])
        all_desc_tokens = desc_tokens | desc_japanese

        overlap = all_input_tokens & all_desc_tokens
        score += len(overlap) * 1.0

        # 3. 部分文字列マッチ（入力トークンが説明文に含まれる）
        for token in all_input_tokens:
            if len(token) >= 2 and token in searchable_text:
                score += 0.5

        scores.append((name, score))

    # スコア降順でソート
    scores.sort(key=lambda x: x[1], reverse=True)

    # ALWAYS_RECOMMEND を常に含め、スコア上位を追加
    recommended = set(ALWAYS_RECOMMEND)
    for name, _score in scores:
        if len(recommended) >= top_n + len(ALWAYS_RECOMMEND):
            break
        recommended.add(name)

    # 常時推奨ツールを先頭に、残りをスコア順に並べる
    result = list(ALWAYS_RECOMMEND & recommended)
    remaining = [name for name, _ in scores if name in recommended and name not in result]
    result.extend(remaining)

    return result


def clean_schema(schema: dict) -> dict:
    """JSON schema の各プロパティから description を削除しトークンを節約する。

    type/enum/default/required は保持し、Function Calling の引数精度を維持したまま
    ツール定義ブロックを軽量化する（実測で約23%削減）。ツール定義は prefill 計算量に
    直結するため、削減はキャッシュの有無に関わらず推論高速化に寄与する。
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    props = schema.get("properties", {})
    clean = {}
    for k, v in props.items():
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "description"}
        else:
            clean[k] = v
    out["properties"] = clean
    return out


def registry_to_openai_tools(tool_names: list[str] = None) -> list[dict]:
    """レジストリからOpenAI tools パラメータ形式のリストを生成します。

    Args:
        tool_names: 含めるツール名のリスト。Noneの場合は全ツール。

    Returns:
        [{"type": "function", "function": {"name", "description", "parameters"}}, ...]
    """
    names = set(tool_names) if tool_names else None
    schema = []
    for name, entry in TOOL_REGISTRY.items():
        if names and name not in names:
            continue
        schema.append(
            {
                "type": "function",
                "function": {"name": name, "description": entry["description"], "parameters": clean_schema(entry["schema"])},
            }
        )
    return schema


def get_tools_schema() -> list:
    """レジストリからOpenAI Function Callingスキーマを動的に生成します（後方互換）。"""
    return registry_to_openai_tools()


# 後方互換のためにモジュールレベルで保持
TOOLS_SCHEMA = get_tools_schema()


# =====================================================
# 動作ルールプロンプト — セクション定数 & ビルダー
# generate_behavior_prompt はこれらを組み立てて出力する（出力テキストは不変）。
# =====================================================

_BASIC_POLICY_DEEP = """\
【行動の基本方針 — 深度思考モード】
- **深く推論してから結論を出せ。** <think> ブロック内で複数の仮説を立て、それぞれの根拠と反証を比較し、最も妥当な結論を導いてください。急いでツールを呼ぶ必要はありません。
- **推論は省略するな。** なぜその結論に至ったかの根拠と、検討して棄却した代替案を述べてください。
- **思考を引き継げ。** 前回の思考メモ（注入済み）があれば、それを踏まえて議論を前進させてください。
- **search_and_replace を使う前は必ず read_file で対象箇所を確認し、ファイル内の実際のテキストを search_block にコピーすること。** 多少のインデント差・表記揺れは自動補正するが、対象ブロックの全行を省略なく含めること（行の省略は失敗する）。
- **update_state は状態の記録専用。実行・読み取りの代わりにならない。** ユーザーが「実行して」「動かして」「走らせて」と言ったら `run_command`(同期待機) か `run_async_test`(別ウィンドウ・非同期) で実際に実行すること。「別ウィンドウで」「バックグラウンドで」は `run_async_test`。update_state に「実行中」と書くだけでは実行したことにならない。"""

_BASIC_POLICY_SHALLOW = """\
【行動の基本方針】
- **考えた後に実行せよ。** ツールを呼び出す前に、引数が正しいか、実行結果が期待通りになるかを簡潔に確認すること。
- **同じ内容を繰り返すな。** 1回考えたら実行に移り、同じ検討を何度も繰り返さないこと。
- **search_and_replace を使う前は必ず read_file で対象箇所を確認し、ファイル内の実際のテキストを search_block にコピーすること。** 多少のインデント差・表記揺れは自動補正するが、対象ブロックの全行を省略なく含めること（行の省略は失敗する）。
- **出力は簡潔に。** 長文の解説は不要。ツールを呼び出して結果を得ること。
- **update_state は状態の記録専用。実行・読み取りの代わりにならない。** ユーザーが「実行して」「動かして」「走らせて」と言ったら `run_command`(同期待機) か `run_async_test`(別ウィンドウ・非同期) で実際に実行すること。「別ウィンドウで」「バックグラウンドで」は `run_async_test`。update_state に「実行中」と書くだけでは実行したことにならない。"""


_CODE_MODE_POLICY = """\
【行動の基本方針 — コード専門モード（/code）】
コードの調査・設計・修正では、コンテキストを汚さず段階的に進めること。生コードの全文読みでコンテキストを埋め尽くすのを避け、構造化ツールで絞り込む。
1. **まずシステムの【プロジェクト構造】（/code-init で記憶済み・state に注入されている）を参照せよ。** 全体構造が既に把握済みの場合は `view_tree`/`gather_project_info`/`map_codebase`（重い）を**再実行せず**、不足する個別ファイルのみ `get_code_outline` で補うこと。【プロジェクト構造】が空のときだけ `view_tree`+`get_code_outline` で初回把握する。
2. **複数ファイル横断の詳細調査・根本原因の特定には `delegate_research` で独立サブエージェントに調査を委譲**する（結論だけが戻りメインコンテキストを節約）。全体把握はステップ1で済むので、delegate_research は深掘り時に使うこと。
3. **モジュールの依存関係を詳細に知る場合は `map_codebase`、役割を深く知る場合は `analyze_file`** を使う（裏で要約・メインコンテキストを汚さない）。
4. **設計案は生コードではなく、これらの要約された知識をベースに組み立てる**こと。
5. **対象ファイルを編集する前に、必ず `get_code_outline` で構造（関数・クラスの一覧）を把握**する。
6. **編集前は `research_code_paths` で影響範囲（コールサイト・定義点・使用点）を確認**し、他を壊さないか検証する。
7. **修正は「1つの関数・1つのクラス」ごとに分割**し、該当箇所は `read_symbol` でシンボル単位で読み込む。`read_file` の全文読みは編集直前の最小範囲確認に限定する。
- **深く推論してから結論を出せ。** `<think>` ブロック内で複数の仮説を立て、根拠と反証を比較し、最も妥当な結論を導くこと。急いでツールを呼ぶ必要はない。
- **search_and_replace を使う前は必ず該当箇所を確認し、実際のコードを search_block にコピーすること。** 多少のインデント差・表記揺れは自動補正するが、対象ブロックの全行を省略なく含めること。
- **スクリプト・テスト・バッチの実行は `run_command`(同期待機) か `run_async_test`(別ウィンドウ・非同期)。** 「別ウィンドウで」「バックグラウンドで」は `run_async_test`。update_state に「実行中」と記録するだけでは実行したことにならない。"""


_UPDATE_STATE_SECTION = """\
【状態の維持（update_state）】
行動を起こす前、または新しい事実が判明した場合、タスクが完了した場合は、必ず `update_state` ツールを呼び出して自身の状態を整理し、記憶を最新化してください。
複数の項目を一度に更新できます（引数はすべて省略可能です）。"""

_FINAL_CHECK_SECTION = """\
---
【実行前の最終確認】
ツールを呼び出す前に、以下を確認してください。
- search_and_replace の search_block は read_file で確認した実際のファイル内容と一致しているか（多少の差は自動補正・全行の包含を確認）
- write_file や replace_lines で書き換える内容は正しいか
- 不要な繰り返しや再考に陥っていないか

【重要：行動宣言と最終回答の違い】
- **「次に〜します」「これから〜を確認します」** と書く場合は、**必ず同じ応答内で対応する tool_call を出してください。** tool_call を出さない自然文は、エンジン側で「ユーザーへの最終回答」として扱われ、調査が強制終了されます。
- **調査が未完了なら、説明だけで終わらず必ずツールを呼び出してください。**
- **調査が完了している場合は、** 「結論」「対応案」「まとめ」など最終回答であることが分かる構造で出力してください。"""


def _section_tool_usage(has, pick):
    """S2: ツール使用の鉄則（トークン節約）。該当ツールがなければ None。"""
    saving_lines = []
    outline_tool = pick("get_code_outline", "research_code_paths")
    read_sym = pick("read_symbol")
    if outline_tool and has("read_file"):
        line = f"- ファイルを読む前に `{outline_tool}` で構造（関数/クラス一覧）を把握してください。"
        if read_sym:
            line += (
                f"大きいファイル（目安500行超）は `{read_sym}` で必要なシンボルだけを読み、"
                "`read_file` の全文読込は小ファイルまたは編集直前の確認に限定してください。"
            )
        else:
            line += (
                "`read_file` は `start_line`/`end_line` で必要範囲だけを読み、"
                "全文読込は小ファイルまたは編集直前の確認に限定してください。"
            )
        saving_lines.append(line)
    elif outline_tool:
        saving_lines.append(f"- まずは `{outline_tool}` で構造を把握してください。")
    elif has("read_file"):
        saving_lines.append(
            "- `read_file` は `start_line`/`end_line` で必要範囲だけを読み、"
            "全文読込は小ファイルまたは編集直前の確認に限定してください。"
        )

    if has("analyze_file"):
        saving_lines.append(
            "- ファイルの全体像やロジックを理解したい場合は、メインの文脈を汚さない `analyze_file` を優先してください。"
        )

    if saving_lines:
        return "【ツール使用の鉄則 - トークン節約】\n" + "\n".join(saving_lines)
    return None


def _section_update_state(has):
    """S3: update_state が利用可能なら状態維持セクションを返す。"""
    return _UPDATE_STATE_SECTION if has("update_state") else None


def _section_action_rules(has, pick):
    """S4: 複数ツールの連続呼び出しルール。該当ルールがなければ None。"""
    action_rules = []

    parallel_tools = [t for t in ("list_directory", "read_file", "grep_search", "get_code_outline") if has(t)]
    if parallel_tools:
        parallel_str = ", ".join(f"`{t}`" for t in parallel_tools)
        action_rules.append(
            f"1. **1回の返答に複数のツール呼び出しを含めることができます**。"
            f"互いに依存しない読み取り専用ツール（{parallel_str} など）は、"
            "1回の返答にまとめて並列実行できます。"
        )
    else:
        action_rules.append(
            "1. **1回の返答に複数のツール呼び出しを含めることができます**。"
            "互いに依存しない読み取り専用ツールは、1回の返答にまとめて並列実行できます。"
        )

    serial_tools = [t for t in ("search_and_replace", "write_file", "replace_lines", "run_command") if has(t)]
    if serial_tools:
        serial_str = ", ".join(f"`{t}`" for t in serial_tools)
        action_rules.append(
            f"2. **読み取りツールの結果を見てから実行する必要がある操作**"
            f"（{serial_str} など）は、必ず別の返答で呼び出してください。"
        )
    else:
        action_rules.append(
            "2. **読み取りツールの結果を見てから実行する必要がある操作**は、必ず別の返答で呼び出してください。"
        )

    edit_tool = pick("search_and_replace", "replace_lines")
    write_tool = pick("write_file")
    if edit_tool or write_tool:
        edit_lines = []
        if edit_tool:
            edit_lines.append(f"   - **既存ファイルの修正**: 必ず `{edit_tool}` を使用すること。")
            if write_tool:
                edit_lines.append(f"   `{write_tool}` で既存ファイルを全体上書きしてはならない。")
        if write_tool:
            edit_lines.append(f"   - **新規ファイルの作成**: `{write_tool}` を使用すること。")
        if edit_tool and has("read_file"):
            edit_lines.append("   - 修正前に `read_file` で該当箇所を確認し、正確なコードをコピーすること。")
        if edit_lines:
            action_rules.append("3. **ファイル編集の鉄則（最も重要）**:\n" + "\n".join(edit_lines))

    if has("run_command") and write_tool:
        action_rules.append(
            f"4. **複雑なコマンドの禁止**: `run_command` で複雑なPythonワンライナー"
            f'（`python -c "..."`）を実行しないこと。Windowsのクォーテーション仕様により'
            f"構文エラーがループする。複雑な処理は `{write_tool}` で一時スクリプト"
            "（例: `temp_script.py`）を作成してから `run_command` で "
            "`python temp_script.py` のように実行すること。"
        )

    if has("run_command"):
        action_rules.append(
            f"{len(action_rules) + 1}. **`run_command` のシェル仕様**: "
            "WindowsのPowerShellでは `&&` が使えず即座にパースエラーになるため絶対に使わない"
            "（複数コマンドは `;` で繋ぐ）。ディレクトリを移動する場合は `cd` 連結ではなく "
            "`working_directory` 引数を使う。入力を待つ対話的プログラム（`input()` 等）には "
            "`input` 引数で終了指示を渡しハングさせないこと。"
            "実行失敗時はエラー文を読んで根本原因を取り除き、セパレータの差し替え等の"
            "表面的な修正で同じコマンドを繰り返し試さないこと。"
        )

    if action_rules:
        return "【重要な動作ルール：タスクの実行と複数ツールの連続呼び出し】\n" + "\n".join(action_rules)
    return None


def _section_doc_strategy(has, pick):
    """S5: 仕様書・ドキュメント作成戦略。該当フェーズがなければ None。"""
    overview_tool = pick("gather_project_info", "view_tree", "list_directory")
    outline_for_doc = pick("get_code_outline", "research_code_paths")
    search_tool = pick("grep_search")
    analysis_tool = pick("analyze_file")
    read_tool = pick("read_file")
    write_for_doc = pick("write_file")

    doc_phases = []
    if overview_tool or outline_for_doc:
        phase1_parts = []
        if overview_tool:
            phase1_parts.append(f"`{overview_tool}` でプロジェクト全体像を把握する")
        if outline_for_doc:
            phase1_parts.append(f"`{outline_for_doc}` で対象ファイルの構造を抽出する")
        if len(phase1_parts) == 2:
            phase1_text = (
                f"`{overview_tool}` でプロジェクト全体像を把握し、`{outline_for_doc}` で対象ファイルの構造を抽出する"
            )
        else:
            phase1_text = phase1_parts[0]
        doc_phases.append(("構造把握", f"**構造把握フェーズ**: {phase1_text}"))

    if search_tool:
        doc_phases.append(
            (
                "検索",
                f"**検索・絞り込みフェーズ**: `{search_tool}` でキーワード検索して対象ファイルをピンポイントに絞る",
            )
        )

    if analysis_tool:
        cache_note = "（結果は自動的に `.pixie_notes/analysis_cache.md` にキャッシュされます）"
        doc_phases.append(
            ("収集", f"**収集フェーズ**: 絞り込んだファイルを `{analysis_tool}` で個別に要約{cache_note}")
        )

    if read_tool and write_for_doc:
        doc_phases.append(
            (
                "生成",
                f"**生成フェーズ**: `{read_tool}` でキャッシュを読み込み、"
                f"蓄積された要約を元に `{write_for_doc}` でタスクを実行",
            )
        )

    if doc_phases:
        numbered = [f"{i + 1}. {text}" for i, (_label, text) in enumerate(doc_phases)]
        doc_section = (
            "【仕様書・ドキュメント作成時の推奨戦略】\n"
            "大量のファイルを調査・修正・作成する場合は、以下のフェーズで進めてください：\n" + "\n".join(numbered)
        )
        if read_tool and (outline_for_doc or analysis_tool):
            note_text = "※ "
            note_text += f"`{read_tool}` でファイル全文をコンテキストに載せると記憶が溢れることがあります。"
            sub_notes = []
            if outline_for_doc:
                sub_notes.append(f"アウトラインだけ見たい場合は `{outline_for_doc}`")
            if analysis_tool:
                sub_notes.append(f"要約目的なら `{analysis_tool}`")
            if sub_notes:
                note_text += "圧迫時は " + "、".join(sub_notes) + " を使ってください。"
            note_text += "ただしコンテキストに余裕がある場合は全文読みで構いません。"
            doc_section += "\n\n" + note_text
        return doc_section
    return None


def _section_doc_gen(has, pick):
    """S6: 長文・ドキュメント生成の鉄則。該当ツールがなければ None。"""
    sections_tool = pick("write_sections")
    append_tool = pick("append_to_file")
    write_tool = pick("write_file")

    if sections_tool or write_tool:
        doc_gen_rules = []
        doc_gen_rules.append(
            "【長文・ドキュメント生成の鉄則】\n"
            "レポート、仕様書、README、設計書などの長いMarkdownドキュメントを生成する場合は以下を厳守せよ："
        )

        if sections_tool:
            doc_gen_rules.append(
                f"1. **`{sections_tool}` ツールを最優先で使用せよ。** "
                "セクション構造（見出しと各セクションの説明指示）を渡すだけで、"
                "セクションごとに独立した生成エンジンが高品質な本文を作成する。"
            )
            fallback_parts = []
            if write_tool:
                fallback_parts.append(f"まず `{write_tool}` で第1セクション（冒頭〜最初の見出し）を書く")
            if append_tool:
                fallback_parts.append(f"続くセクションは `{append_tool}` で追記する")
            elif write_tool:
                fallback_parts.append(f"続くセクションも `{write_tool}` で書く")
            fallback_parts.append("1回のツール呼び出しで全内容を書こうとするな（トークン上限で途切れる）")
            doc_gen_rules.append(
                f"2. **`{sections_tool}` が使えない場合の代替戦略:**\n   - " + "\n   - ".join(fallback_parts)
            )
        elif write_tool:
            doc_gen_rules.append(
                f"1. まず `{write_tool}` で第1セクションを書き、"
                + (f"`{append_tool}` で続くセクションを追記する。" if append_tool else "続くセクションも個別に書く。")
            )
            doc_gen_rules.append("2. 1回のツール呼び出しで全内容を書こうとするな（トークン上限で途切れる）")

        doc_gen_rules.append(
            "3. **省略は厳禁。** 「...」「（省略）」「（以下同様）」「など」で内容を端折るな。"
            "読者が知りたい全ての情報を書け。"
        )
        doc_gen_rules.append("4. **箇条書きだけで済ませるな。** 箇条書きの後に必ず説明文を添えよ。")
        return "\n".join(doc_gen_rules)
    return None


def generate_behavior_prompt(
    available_tools: set[str] | None = None, thinking_mode: str = "shallow", mode: str = "normal"
) -> str:
    """Function Calling用の動作ルールプロンプトを動的に生成します。

    available_tools に指定されたツールのみをプロンプト内で言及し、
    API tools パラメータとの不一致を防ぎます。
    None の場合は全ツールを言及（後方互換）。

    ツールスキーマは tools パラメータで別枠送信されるため、
    システムプロンプトには動作ルールのみを含みます。

    Args:
        available_tools: 利用可能なツール名のセット（None時は全ツール）
        thinking_mode: "shallow"（即断即実・簡潔）または "deep"
                       （<think>で複数仮説を推論）。セクション1の基本方針が切り替わる。
        mode: "code" のときセクション1を _CODE_MODE_POLICY（コード専門ワークフロー）に切替。
    """
    if available_tools is None:
        available_tools = set(TOOL_REGISTRY.keys())

    def _pick(*candidates: str) -> str | None:
        """候補リストから利用可能な最初のツールを返す。"""
        for c in candidates:
            if c in available_tools:
                return c
        return None

    _has = lambda name: name in available_tools

    # 各セクションを組み立て（None のセクションは除外）。セクション間は改行2つ。
    if mode == "code":
        _base_policy = _CODE_MODE_POLICY
    else:
        _base_policy = _BASIC_POLICY_DEEP if thinking_mode == "deep" else _BASIC_POLICY_SHALLOW
    parts = [
        _base_policy,
        _section_tool_usage(_has, _pick),
        _section_update_state(_has),
        _section_action_rules(_has, _pick),
        _section_doc_strategy(_has, _pick),
        _section_doc_gen(_has, _pick),
        _FINAL_CHECK_SECTION,
    ]
    return "\n\n".join(p for p in parts if p is not None)

    # ============================
    # ツール実行エンジン
    # ============================


def _color_diff(old_text: str, new_text: str, old_label: str = "before", new_label: str = "after") -> str:
    """二つのテキストのunified diffをANSIカラー付きで生成する。"""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=old_label, tofile=new_label, lineterm="")

    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"

    colored = []
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            colored.append(GREEN + line + RESET)
        elif line.startswith("-") and not line.startswith("---"):
            colored.append(RED + line + RESET)
        else:
            colored.append(line)
    return "\n".join(colored) if colored else "(差分はありません)"


@register_tool(
    name="diff_files",
    description="2つのテキストファイル、またはテキスト同士の差分をカラーで表示します。追加行は緑、削除行は赤で表示されます。",
    schema={
        "type": "object",
        "properties": {
            "old_path": {"type": "string", "description": "変更前のファイルパス（または変更前テキスト）"},
            "new_path": {"type": "string", "description": "変更後のファイルパス（または変更後テキスト）"},
            "old_label": {"type": "string", "description": "変更前の表示名（省略時はファイル名）"},
            "new_label": {"type": "string", "description": "変更後の表示名（省略時はファイル名）"},
        },
        "required": ["old_path", "new_path"],
    },
    category="extended",
    prompt_desc="diff_files(old_path, new_path, old_label?, new_label?): 2ファイル間の差分を緑(追加)/赤(削除)で表示",
)
def diff_files(old_path: str, new_path: str, old_label: str = None, new_label: str = None) -> str:
    """2つのテキスト間の差分をカラーで表示する。ファイルが存在すればその内容を使用し、なければ引数をテキストとして扱う。"""
    try:
        old_p = Path(old_path)
        new_p = Path(new_path)
        old_exists = old_p.exists() and old_p.is_file()
        new_exists = new_p.exists() and new_p.is_file()
    except Exception:
        old_exists = new_exists = False

    try:
        old_text = old_p.read_text(encoding="utf-8") if old_exists else old_path
        new_text = new_p.read_text(encoding="utf-8") if new_exists else new_path
    except UnicodeDecodeError:
        try:
            old_text = old_p.read_text(encoding="cp932") if old_exists else old_path
            new_text = new_p.read_text(encoding="cp932") if new_exists else new_path
        except Exception as e:
            return f"Error: テキストの読み込みに失敗: {e}"
    except Exception as e:
        return f"Error: テキストの読み込みに失敗: {e}"

    _old_label = old_label or (old_p.name if old_exists else "old")
    _new_label = new_label or (new_p.name if new_exists else "new")

    return _color_diff(old_text, new_text, _old_label, _new_label)


def _execute_builtin_tool(name: str, arguments: dict[str, Any]) -> str:
    """レジストリからツールを検索して実行します。"""
    entry = TOOL_REGISTRY.get(name)
    if not entry:
        return f"Error: ツール '{name}' は見つかりません。"

    try:
        func = entry["func"]
        # 関数のシグネチャに基づいて引数を渡す
        sig = inspect.signature(func)
        valid_args = {}
        for param_name, param in sig.parameters.items():
            if param_name in arguments:
                valid_args[param_name] = arguments[param_name]
            elif param.default is inspect.Parameter.empty:
                # 必須引数が不足
                return f"Error: 必要な引数 '{param_name}' が不足しています。"
        return func(**valid_args)
    except Exception as e:
        return f"Error: ツール実行中の予期せぬエラー: {e}"


def _truncation_suffix(kept: str, cap: int = TOOL_RESULT_MAX_CHARS) -> str:
    """切詰め後テキストから、行番号付き read_file 用の正確な再取得ヒントを生成する。

    kept: 切り詰め後に残ったテキスト（先頭〜cap文字まで）。
    行番号付き（"NNN: ..."）であれば最終行を検出して「続きは start_line=N」を案内し、
    そうでなければ汎用メッセージを返す。重複再読込の防止が目的。
    """
    # ヘッダから全行数を抽出（"[file.py] 全487行 (...)" 等）
    total_lines = None
    m_total = re.search(r"全(\d+)行", kept)
    if m_total:
        total_lines = int(m_total.group(1))

    # 残ったテキスト中の最後の "NNN: " 行番号プレフィックスを検出
    line_nums = list(re.finditer(r"(?m)^(\d+):\s", kept))
    if line_nums:
        last_line = int(line_nums[-1].group(1))
        nxt = last_line + 1
        if total_lines and nxt <= total_lines:
            return (
                f"\n...[System: 出力が長いため{cap}文字で切り捨てられました。"
                f"全{total_lines}行中 {last_line}行目まで表示済み。"
                f"続きは read_file の start_line={nxt} で取得してください（重複読込禁止）]..."
            )
        return (
            f"\n...[System: 出力が長いため{cap}文字で切り捨てられました。"
            f"{last_line}行目まで表示済み。続きは read_file の start_line={nxt} で取得してください]..."
        )

    return (
        f"\n...[System: 出力が長いため{cap}文字で切り捨てられました。"
        f"analyze_fileで要約するか、ツールの範囲指定で再取得してください]..."
    )


def execute_builtin_tool(name: str, arguments: dict[str, Any]) -> str:
    """ツール名と引数を受け取り、対応するPython関数を実行します。結果が巨大な場合は切り捨てます。

    上限は engine がコンテキスト使用率から逆算して set_tool_result_max_chars() で設定した
    動的値（_dynamic_max_chars）を優先し、未設定時は TOOL_RESULT_MAX_CHARS にフォールバックする。
    """
    result = _execute_builtin_tool(name, arguments)

    # 巨大な生データによるLLMのコンテキスト溢れを防ぐため上限で切り捨て。
    # read_file の行番号付き結果には正確な続き行を案内し、重複再読込を防ぐ。
    cap = registry._dynamic_max_chars or TOOL_RESULT_MAX_CHARS
    if len(result) > cap:
        result = result[:cap] + _truncation_suffix(result[:cap], cap)

    return result

    # ============================
    # Visionユーティリティ
    # ============================


def resize_and_encode_image(path: str, max_size: int = 1024, quality: int = 85) -> str:
    """画像をリサイズ・圧縮してBase64 data URI文字列を返す。
    Pillowがインストールされている場合はリサイズを行い、ない場合はそのままBase64化する。
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")

    try:
        import io

        from PIL import Image

        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            if w > max_size or h > max_size:
                if w > h:
                    new_w = max_size
                    new_h = int(h * (max_size / w))
                else:
                    new_h = max_size
                    new_w = int(w * (max_size / h))
                if hasattr(Image, "Resampling"):
                    resample_method = Image.Resampling.LANCZOS
                else:
                    resample_method = Image.ANTIALIAS
                img = img.resize((new_w, new_h), resample_method)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            data = buffer.getvalue()
    except ImportError:
        print("[警告] Pillowがインストールされていないため、画像のリサイズ最適化をスキップします。")
        with open(path, "rb") as f:
            data = f.read()

    base64_data = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"


def grab_screen_and_encode(bbox, max_size: int = 1024, quality: int = 85) -> str:
    """指定領域のスクリーンショットを取得し、Base64 JPEG data URI で返す。
    Pillowがインストールされていない場合はエラーメッセージを返す。
    """

    try:
        from PIL import Image, ImageGrab
    except ImportError:
        return "Error: スクリーンショット機能を使用するにはPillowが必要です。'pip install Pillow' を実行してください。"

    try:
        # OS固有機能ではなくPillowでスクショを取得
        img = ImageGrab.grab(bbox=bbox)

        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if w > max_size or h > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * (max_size / w))
            else:
                new_h = max_size
                new_w = int(w * (max_size / h))
            if hasattr(Image, "Resampling"):
                resample_method = Image.Resampling.LANCZOS
            else:
                resample_method = Image.ANTIALIAS
            img = img.resize((new_w, new_h), resample_method)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        data = buffer.getvalue()

        base64_data = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"
    except Exception as e:
        return f"Error: スクリーンショットの取得に失敗しました: {e}"


def check_loop_detected(executed_actions: list, current_action: str, threshold: int = 2) -> bool:
    """直近のツール呼び出し履歴を確認し、同一アクションの無限ループを検知する。

    Args:
        executed_actions: これまでのアクション文字列のリスト（"ツール名:引数" 形式）
        current_action: 今回実行しようとしているアクション文字列
        threshold: 何回連続で同一アクションが続いたらループと判断するか

    Returns:
        ループが検知された場合 True
    """
    if len(executed_actions) < threshold - 1:
        return False

    # 直近 (threshold-1) 回がすべて current_action と同じならループ
    recent = executed_actions[-(threshold - 1) :]
    return all(action == current_action for action in recent)
