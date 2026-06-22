"""コード解析・調査系ツール群（code_index.py のラッパ）。

tools.py の TOOL_REGISTRY に登録するため、本モジュールは tools.py 末尾で
import される（循環回避: register_tool のみトップレベル import し、view_tree 等
tools 内関数は関数内遅延 import する）。

依存: tools.py (register_tool), code_index.py (AST), paths.py, 標準ライブラリ
"""

import os
import re
from pathlib import Path

from paths import get_data_path
from tools import register_tool

# =====================================================
# 正規表現アウトライン（AST フォールバック / JS・TS 用）
# =====================================================


def _regex_outline_py(file_path: Path) -> list:
    """正規表現ベースの Python アウトライン（AST 失敗時のフォールバック）。"""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = text.splitlines()
    py_re = r"^\s*(async\s+)?(def|class)\s+\w+"
    sigs = []
    for i, line in enumerate(lines, 1):
        if re.match(py_re, line):
            sigs.append((i, line.rstrip(":")))
    out = []
    for idx, (start_no, sig) in enumerate(sigs):
        end_no = (sigs[idx + 1][0] - 1) if idx + 1 < len(sigs) else len(lines)
        out.append(f"  {start_no}-{end_no} ({end_no - start_no + 1}行): {sig}")
    return out


def _regex_outline_js(file_path: Path) -> list:
    """正規表現ベースの JS/TS アウトライン（code_index は Python 専用のため）。"""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = text.splitlines()
    js_re1 = r"^\s*(export\s+)?(default\s+)?(async\s+)?(function|class)\s+\w+"
    js_re2 = r"^\s*(const|let|var)\s+\w+\s*=\s*(async\s+)?\(.*\)\s*=>"
    sigs = []
    for i, line in enumerate(lines, 1):
        if re.match(js_re1, line) or re.match(js_re2, line):
            sigs.append((i, line.strip()))
    out = []
    for idx, (start_no, sig) in enumerate(sigs):
        end_no = (sigs[idx + 1][0] - 1) if idx + 1 < len(sigs) else len(lines)
        out.append(f"  {start_no}-{end_no} ({end_no - start_no + 1}行): {sig}")
    return out


# =====================================================
# get_code_outline（Python は AST、JS/TS は正規表現）
# =====================================================


@register_tool(
    name="get_code_outline",
    description="ファイルやディレクトリ内のソースコード（Python, JS/TS等）からクラスと関数のシグネチャを抽出し、ファイルのアウトライン（構造マップ）を取得します。コードの全体像を高速に把握するのに最適です。",
    schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "対象ファイルまたはディレクトリのパス"}},
        "required": ["path"],
    },
    prompt_desc="get_code_outline(path): ファイル・ディレクトリ内のクラスと関数名一覧（アウトライン）を高速抽出",
)
def get_code_outline(path: str) -> str:
    """指定されたソースコードからクラスと関数のシグネチャを抽出します。

    Python は code_index.outline (AST) で正確に抽出（ネスト関数・デコレータ・
    正確な end_lineno に対応）。AST 失敗時は正規表現フォールバック。
    JS/TS は正規表現（code_index は Python 専用のため）。
    """
    target = Path(path)
    if not target.exists():
        return f"Error: パスが存在しません ({path})"

    def extract_from_file(file_path: Path) -> list:
        # --- Python: AST via code_index.outline (フォールバック: 正規表現) ---
        if file_path.suffix == ".py":
            try:
                from code_index import outline as _py_outline

                syms = _py_outline(file_path)
                if syms:
                    out = []
                    for s in syms:
                        start = s.get("lineno")
                        end = s.get("end_lineno") or start
                        nlines = (end - start + 1) if end else 1
                        span = f"{start}-{end}" if end and end != start else f"{start}"
                        deco = s.get("decorators") or []
                        deco_str = f"  @{', '.join(deco)}" if deco else ""
                        kind = s.get("kind", "")
                        name = s.get("qualname") or s.get("name", "")
                        out.append(f"  {span} ({nlines}行): {kind} {name}{deco_str}")
                    return out
            except Exception:
                pass  # AST 失敗 → 正規表現フォールバック
            return _regex_outline_py(file_path)

        # --- JS/TS: 正規表現（code_index は Python 専用） ---
        if file_path.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            return _regex_outline_js(file_path)
        return []

    results = []

    if target.is_file():
        out = extract_from_file(target)
        if out:
            results.append(f"[{target.name}]\n" + "\n".join(out))
    else:
        ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}
        for item in sorted(target.rglob("*")):
            if not item.is_file() or any(part in ignore_dirs for part in item.parts):
                continue
            if item.suffix in [".py", ".js", ".ts", ".jsx", ".tsx"]:
                out = extract_from_file(item)
                if out:
                    try:
                        rel = item.relative_to(target)
                    except Exception:
                        rel = item.name
                    results.append(f"[{rel}]\n" + "\n".join(out))

    if not results:
        return f"関数やクラスの定義が見つかりませんでした ({path})"

    return "\n\n".join(results)


# =====================================================
# research_code_paths
# =====================================================


@register_tool(
    name="research_code_paths",
    description="指定したキーワードの定義箇所（点）と使用箇所（線）を調査し、コード内の影響範囲やデータフローを可視化します。",
    schema={
        "type": "object",
        "properties": {"keyword": {"type": "string", "description": "調査したい変数名、関数名、または定数名"}},
        "required": ["keyword"],
    },
    category="extended",
    prompt_desc="research_code_paths(keyword): キーワードの定義箇所と使用箇所を追跡し、コードの『点と線』を抽出する",
)
def research_code_paths(keyword: str) -> str:
    """キーワードの定義と使用箇所を検索し、構造的に返します。"""
    # 定義箇所のパターン（点）
    def_patterns = [
        re.compile(rf"^\s*{keyword}\s*="),  # 変数代入
        re.compile(rf"^\s*def\s+{keyword}\("),  # 関数定義
        re.compile(rf"^\s*class\s+{keyword}"),  # クラス定義
        re.compile(rf"(?:self|cls)\.{keyword}\s*="),  # インスタンス変数/クラス変数
    ]

    dots = []
    lines = []

    # プロジェクト全体から検索
    ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pixie_notes"}
    cwd = Path.cwd()

    files_searched = 0
    for item in sorted(cwd.rglob("*")):
        if not item.is_file() or any(part in ignore_dirs for part in item.parts):
            continue
        if item.suffix not in [".py", ".js", ".ts", ".md", ".json"]:
            continue

        files_searched += 1
        try:
            content = item.read_text(encoding="utf-8")
        except Exception:
            continue

        rel_path = item.relative_to(cwd)
        for i, line_text in enumerate(content.splitlines(), 1):
            if keyword in line_text:
                is_def = any(p.search(line_text) for p in def_patterns)
                entry = f"{rel_path}:{i}: {line_text.strip()[:120]}"
                if is_def:
                    dots.append(entry)
                else:
                    lines.append(entry)

        if len(dots) + len(lines) > 200:  # 多すぎる場合は打ち切り
            break

    result = [f"### 『点』: {keyword} の定義箇所 (Definitions)"]
    if dots:
        result.extend(dots)
    else:
        result.append("(定義箇所は見つかりませんでした)")

    result.append(f"\n### 『線』: {keyword} の使用箇所・データフロー (Usages/Flow)")
    if lines:
        result.extend(lines[:50])  # 表示は50件まで
        if len(lines) > 50:
            result.append(f"...（他 {len(lines) - 50} 件を省略）")
    else:
        result.append("(使用箇所は見つかりませんでした)")

    return "\n".join(result)


# =====================================================
# gather_project_info
# =====================================================


@register_tool(
    name="gather_project_info",
    description="プロジェクトディレクトリの構造を取得し、主要ファイル(py,js,ts,md等)をanalyze_fileでバッチ要約して .pixie_notes/analysis_cache.md にキャッシュします。仕様書作成の前処理として使用してください。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "対象プロジェクトのディレクトリパス"},
            "max_files": {"type": "integer", "description": "解析する最大ファイル数（デフォルト: 15）"},
            "extensions": {"type": "string", "description": "対象拡張子をカンマ区切り（デフォルト: 'py,js,ts,md'）"},
        },
        "required": ["path"],
    },
    category="extended",
    prompt_desc="gather_project_info(path, max_files?, extensions?): プロジェクト全体のファイル構造と主要ファイルの要約を一括取得し、キャッシュに蓄積する。仕様書作成の前処理に最適",
)
def gather_project_info(path: str, max_files: int = 15, extensions: str = "py,js,ts,md") -> str:
    """プロジェクトのファイル構造を取得し、主要ファイルをバッチ解析してキャッシュに蓄積します。

    注意: 実際のanalyze_file呼び出しはCLI/GUIのインターセプタ側で行われるため、
    このスタブではツリー構造の取得とファイルリストの生成のみ行います。
    """
    target = Path(path)
    if not target.exists() or not target.is_dir():
        return f"Error: ディレクトリが存在しません ({path})"

    # max_files の int 変換
    try:
        max_files = int(max_files)
    except (ValueError, TypeError):
        max_files = 15

    # 拡張子リストのパース
    ext_set = set()
    for ext in extensions.split(","):
        ext = ext.strip().lstrip(".")
        if ext:
            ext_set.add(f".{ext}")

    # 無視するディレクトリ
    ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pixie_notes"}

    # ツリー構造の取得（3階層まで）— view_tree は tools.py の関数（遅延 import で循環回避）
    from tools import view_tree

    tree_result = view_tree(path, max_depth=3)

    # 対象ファイルの列挙
    target_files = []
    for item in sorted(target.rglob("*")):
        if not item.is_file():
            continue
        if any(part in ignore_dirs for part in item.parts):
            continue
        if item.suffix.lower() in ext_set:
            target_files.append(str(item))
            if len(target_files) >= max_files:
                break

    # キャッシュディレクトリの準備（古いキャッシュをクリア）
    cache_dir = get_data_path(".pixie_notes")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "analysis_cache.md")
    with open(cache_path, "w", encoding="utf-8") as cf:
        cf.write("# プロジェクト解析キャッシュ\n")
        cf.write(f"対象: `{path}`\n\n")
        cf.write(f"## ディレクトリ構造\n```\n{tree_result}\n```\n\n---\n")

    # ファイルリストをINTERCEPT用マーカー付きで返す
    # react_loop側でこのマーカーを検知してanalyze_fileをバッチ実行する
    file_list_str = "\n".join([f"- {f}" for f in target_files])

    result = "プロジェクト構造を取得し、キャッシュを初期化しました。\n"
    result += f"キャッシュファイル: {cache_path}\n\n"
    result += f"## ディレクトリ構造\n```\n{tree_result}\n```\n\n"
    result += f"## 解析対象ファイル ({len(target_files)}件)\n{file_list_str}\n\n"
    result += (
        "次のステップ: 上記ファイルを `analyze_file` で個別に解析してください。(思考と出力は日本語のみにしてください)\n"
    )
    result += f"解析結果は自動的に `{cache_path}` にキャッシュされます。\n"
    result += f"全ファイルの解析完了後、`read_file` で `{cache_path}` を読み込んで仕様書を作成してください。"

    return result


# =====================================================
# map_codebase
# =====================================================


@register_tool(
    name="map_codebase",
    description="コードベース全体をASTで解析し、モジュール/シンボル/外部依存/複雑度ホットスポット/デッドコード候補数の全体像をテキストで返します。Python(stdlib ast)のみ使用、第三依存なし。初回はキャッシュ(.pixie_notes/code_index.json)を構築し、2回目以降は変更ファイルのみ再解析します。全体把握の最初の一歩として最適。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "解析ルートディレクトリ(.でカレント)"},
            "force_refresh": {
                "type": "boolean",
                "description": "キャッシュを無視して全ファイル再解析(デフォルト false)",
            },
        },
        "required": [],
    },
    category="extended",
    prompt_desc="map_codebase(path?, force_refresh?): コードベース全体のAST構造・依存・デッドコード候補数を概観。全体把握の最初の一歩",
)
def map_codebase(path: str = ".", force_refresh: bool = False) -> str:
    """コードベース全体のASTインデックスを構築/ロードし、コンパクトな全体サマリを返す。"""
    try:
        from code_index import build_index, find_dead_symbols, summarize
    except Exception as e:
        return f"Error: code_index モジュールの読み込みに失敗しました (AST機能は無効): {e}"
    cache_path = None
    try:
        from paths import get_data_path

        cache_path = get_data_path(".pixie_notes/code_index.json")
    except Exception:
        cache_path = None
    try:
        import os

        root = os.path.abspath(path)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        index = build_index(root, cache_path=cache_path, force=bool(force_refresh))
        out = summarize(index)
        try:
            n_dead = len(find_dead_symbols(index, include_dynamic_string_check=False))
            out += f"\n\n---\nデッドコード候補: {n_dead} 件 → `detect_dead_code` で詳細確認。個別シンボル閲覧は `read_symbol`。"
        except Exception:
            pass
        return out
    except NotADirectoryError:
        return f"Error: ディレクトリではありません ({path})"
    except Exception as e:
        return f"Error: インデックス構築に失敗しました: {e}"


# =====================================================
# detect_dead_code
# =====================================================


@register_tool(
    name="detect_dead_code",
    description="ASTインデックスとコールグラフ到達性解析から、エントリポイント(@register_tool装飾関数/__main__/main)から到達不能なデッドコード'候補'をファイル別に一覧します。動的呼出/文字列dispatchの偽陽性があるため'候補'扱い(確定ではない)。Python(stdlib ast)のみ。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "解析ルートディレクトリ(.でカレント)"},
            "force_refresh": {
                "type": "boolean",
                "description": "キャッシュを無視して全ファイル再解析(デフォルト false)",
            },
        },
        "required": [],
    },
    category="extended",
    prompt_desc="detect_dead_code(path?, force_refresh?): 到達不能なデッドコード候補をファイル別一覧(偽陽性注意・候補扱い)",
)
def detect_dead_code(path: str = ".", force_refresh: bool = False) -> str:
    """デッドコード候補を到達性解析+文字列出現フィルタで抽出し、ファイル別に返す。"""
    try:
        from code_index import build_index, find_dead_symbols
    except Exception as e:
        return f"Error: code_index モジュールの読み込みに失敗しました: {e}"
    cache_path = None
    try:
        from paths import get_data_path

        cache_path = get_data_path(".pixie_notes/code_index.json")
    except Exception:
        cache_path = None
    try:
        import os

        root = os.path.abspath(path)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        index = build_index(root, cache_path=cache_path, force=bool(force_refresh))
        candidates = find_dead_symbols(index, include_dynamic_string_check=True)
    except Exception as e:
        return f"Error: デッドコード解析に失敗しました: {e}"

    if not candidates:
        return "デッドコード候補は検出されませんでした(全シンボルがエントリポイントから到達可能、または文字列出現フィルタで除外済み)。"

    high = [s for s in candidates if not s.get("low_confidence")]
    low = [s for s in candidates if s.get("low_confidence")]

    lines = [
        "# デッドコード候補 (candidates — 確定ではない)",
        "",
        "注意: 動的呼出(getattr/文字列dispatch/コールバック/default_factory)は AST で追跡",
        "できないため'候補'扱い。削除前に `read_symbol` で実体確認 + 文字列検索で参照を再点検。",
        "",
        f"高信頼(文字列出現なし): {len(high)} 件 / 低信頼(出現あり・動的参照の可能性): {len(low)} 件",
        "",
    ]

    def _render(title: str, group: list) -> None:
        if not group:
            return
        lines.append(f"## {title}")
        by_file: dict[str, list] = {}
        for sym in group:
            by_file.setdefault(sym["file"], []).append(sym)
        for f in sorted(by_file):
            for sym in sorted(by_file[f], key=lambda s: s["lineno"]):
                rng = f"L{sym['lineno']}" + (f"-{sym['end_lineno']}" if sym.get("end_lineno") else "")
                lines.append(f"- {f}::{sym['qualname']} ({sym['kind']}, {rng})")
        lines.append("")

    _render("高信頼 — 真のデッドの可能性高い", high)
    _render("低信頼 — 動的参照の可能性（要確認）", low)
    out = "\n".join(lines)
    HARD = 15800
    if len(out) > HARD:
        out = out[:HARD] + f"\n\n...[出力打切り: 全{len(candidates)}件のうち一部のみ表示]"
    return out


# =====================================================
# read_symbol
# =====================================================


@register_tool(
    name="read_symbol",
    description="指定ファイル内のシンボル(関数/クラス/メソッド)のソースを行範囲で読み込んで返します。read_fileのシンボル単位版。ASTで正確な行範囲を特定(AST不可時は正規フォールバック)。context行数分の前後パディングを付与可能。Python(stdlib ast)のみ。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "対象Pythonファイルのパス"},
            "symbol": {"type": "string", "description": "関数/クラス/メソッド名"},
            "context": {"type": "integer", "description": "シンボル前後に含める余分行数(デフォルト 0)"},
        },
        "required": ["path", "symbol"],
    },
    category="extended",
    prompt_desc="read_symbol(path, symbol, context?): 指定シンボルのソースを行範囲で読込。read_fileのシンボル単位版",
)
def read_symbol(path: str, symbol: str, context: int = 0) -> str:
    """シンボルの行範囲をAST(または正規フォールバック)で解決し、ソースを返す。"""
    from pathlib import Path

    target = Path(path)
    if not target.is_file():
        return f"Error: ファイルが存在しません ({path})"
    try:
        context = max(0, min(int(context), 200))
    except (ValueError, TypeError):
        context = 0

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = target.read_text(encoding="cp932")
        except Exception as e:
            return f"Error: 読み込み失敗: {e}"
    except Exception as e:
        return f"Error: 読み込み失敗: {e}"

    start, end = None, None
    try:
        from code_index import _parse_file

        rec, _ = _parse_file(target, target.name)
        cands = [s for s in rec["symbols"] if s["name"] == symbol]
        if cands:
            cands.sort(key=lambda s: (s["qualname"].count("."), s["lineno"]))
            start = cands[0]["lineno"]
            end = cands[0].get("end_lineno")
    except Exception:
        start, end = None, None

    if start is None:
        import re

        pat = re.compile(rf"^(\s*)(async\s+)?(def|class)\s+{re.escape(symbol)}\b")
        for i, ln in enumerate(text.splitlines(), 1):
            if pat.match(ln):
                start = i
                break

    if start is None:
        return f"シンボル '{symbol}' が {path} 内で見つかりませんでした。"

    lines = text.splitlines()
    total = len(lines)
    if end is None:
        end = total
    lo = max(1, start - context)
    hi = min(total, end + context)
    body = "\n".join(f"{i:>5}: {lines[i - 1]}" for i in range(lo, hi + 1))
    rng_label = f"L{start}-EOF" if end >= total else f"L{start}-{end}"
    header = f"# {target.name}::{symbol}  ({rng_label}, context={context})\n"
    out = header + body
    HARD = 15800
    if len(out) > HARD:
        out = out[:HARD] + "\n...[切り詰め: read_file の start_line/end_line で続きを取得]"
    return out


# =====================================================
# get_file_stats
# =====================================================


@register_tool(
    name="get_file_stats",
    description="指定されたディレクトリ内のファイル一覧と、それぞれの行数・サイズを取得します。run_commandでwc等のコマンドを使わずに安全にファイル統計を取得できます。",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "対象ディレクトリのパス（デフォルト: カレントディレクトリ）"},
            "extensions": {
                "type": "string",
                "description": "対象拡張子（カンマ区切り、デフォルト: .py,.md,.json,.js,.ts,.tsx,.css,.html,.yaml,.yml,.toml）",
            },
        },
        "required": [],
    },
    prompt_desc="get_file_stats(path?, extensions?): ディレクトリ内のファイル一覧と行数・サイズを安全に取得",
    category="core",
)
def get_file_stats(path: str = ".", extensions: str = ".py,.md,.json,.js,.ts,.tsx,.css,.html,.yaml,.yml,.toml") -> str:
    target = Path(path)
    if not target.exists():
        return f"Error: パスが存在しません ({path})"
    if not target.is_dir():
        return f"Error: ディレクトリではありません ({path})"

    ext_list = [e.strip().lower() for e in extensions.split(",")]
    results = []
    for root, _, files in os.walk(target):
        for file in sorted(files):
            if Path(file).suffix.lower() in ext_list:
                filepath = Path(root) / file
                try:
                    size = filepath.stat().st_size
                    with open(filepath, encoding="utf-8", errors="replace") as f:
                        lines = sum(1 for _ in f)
                    rel = filepath.relative_to(target) if filepath.is_relative_to(target) else filepath
                    results.append(f"{rel}: {lines}行 ({size:,}B)")
                except Exception:
                    pass

    if not results:
        return f"対象ファイルが見つかりませんでした (path={path}, extensions={extensions})"
    return "\n".join(results)
