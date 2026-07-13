"""AnythingPixie — コードベース全体の AST インデックス (stdlib ast のみ, 第三依存なし)。

tools.py からは各ツール関数内で遅延 import される（本モジュールが欠落/エラーでも
アプリ全体は起動する）。純粋関数 + JSON キャッシュ可能。本モジュールは app に依存せず、
キャッシュパスは呼び出し側（ツール）が明示的に渡す。

対応言語: Python のみ（標準 ast）。JS/TS は将来 tree-sitter 拡張を想定した構造。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "1"

DEFAULT_IGNORE_DIRS = (
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".pixie_notes", "debug", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "env", "build", "dist",
)

# py3.9 以下のフォールバック用 stdlib 名（本プロジェクトは py3.11 想定だが安全のため）
_STDLIB_FALLBACK = frozenset({
    "os", "sys", "json", "ast", "hashlib", "pathlib", "collections", "dataclasses",
    "datetime", "re", "io", "time", "subprocess", "threading", "queue", "typing",
    "contextlib", "ctypes", "urllib", "importlib", "argparse", "difflib", "platform",
    "multiprocessing", "shutil", "warnings", "math", "functools", "itertools",
    "string", "textwrap", "unicodedata", "enum", "abc",
})


# =====================================================
# 補助: AST ノードからの名前抽出
# =====================================================

def _decorator_base_name(node) -> str:
    """デコレータノードから基底名を抽出（register_tool, app.route 等）。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_base_name(node.func)
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _callee_name(node) -> str | None:
    """Call ノードから呼び出し対象の名前を best-effort で抽出。"""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        parts: list[str] = []
        cur = f
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    return None


def _is_external_module(root_mod: str) -> bool:
    """モジュールルートが stdlib でなければ外部（ローカル/.py または第三）とみなす。"""
    if not root_mod:
        return False
    stdlib = getattr(sys, "stdlib_module_names", None)
    pool = stdlib if stdlib is not None else _STDLIB_FALLBACK
    return root_mod not in pool


def _is_main_guard(node) -> bool:
    """`if __name__ == "__main__":` を判定。"""
    test = node.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return False
    left, right = test.left, test.comparators[0] if test.comparators else None
    cond_a = (isinstance(left, ast.Name) and left.id == "__name__"
              and isinstance(right, ast.Constant) and right.value == "__main__")
    cond_b = (isinstance(right, ast.Name) and right.id == "__name__"
              and isinstance(left, ast.Constant) and left.value == "__main__")
    return bool(cond_a or cond_b)


# =====================================================
# AST 抽出ビジター
# =====================================================

class _Extractor(ast.NodeVisitor):
    """1ファイルの AST を走査し、シンボル/import/コール辺/main guard を収集。"""

    def __init__(self, file_rel: str):
        self.file_rel = file_rel
        self.symbols: list[dict] = []
        self.imports: list[str] = []
        self.external_imports: list[str] = []
        self.has_main_guard = False
        self.main_guard_calls: list[str] = []
        self.call_edges: dict[str, list[str]] = {}  # qualname -> [callee names]
        self._stack: list[str] = []
        self._class_depth = 0

    def _qualname(self, name: str) -> str:
        return ".".join(self._stack + [name]) if self._stack else name

    def _decorator_names(self, node) -> list[str]:
        return [_decorator_base_name(d) for d in node.decorator_list]

    def _collect_callees(self, node) -> list[str]:
        callees: list[str] = []
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, ast.Call):
                cn = _callee_name(child)
                if cn:
                    callees.append(cn)
        return callees

    def _handle_func(self, node, kind: str):
        qn = self._qualname(node.name)
        self.symbols.append({
            "file": self.file_rel,
            "name": node.name,
            "qualname": qn,
            "kind": kind,
            "lineno": node.lineno,
            "end_lineno": getattr(node, "end_lineno", None),
            "decorators": self._decorator_names(node),
            "is_entrypoint_seed": False,
        })
        self.call_edges[qn] = self._collect_callees(node)
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node):
        self._handle_func(node, "method" if self._class_depth > 0 else "function")

    def visit_AsyncFunctionDef(self, node):
        self._handle_func(node, "method" if self._class_depth > 0 else "function")

    def visit_ClassDef(self, node):
        qn = self._qualname(node.name)
        self.symbols.append({
            "file": self.file_rel,
            "name": node.name,
            "qualname": qn,
            "kind": "class",
            "lineno": node.lineno,
            "end_lineno": getattr(node, "end_lineno", None),
            "decorators": self._decorator_names(node),
            "is_entrypoint_seed": False,
        })
        self.call_edges[qn] = self._collect_callees(node)
        self._stack.append(node.name)
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1
        self._stack.pop()

    def visit_Import(self, node):
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            self.imports.append(f"import {alias.name}")
            if _is_external_module(root_mod):
                self.external_imports.append(root_mod)

    def visit_ImportFrom(self, node):
        if node.module:
            root_mod = node.module.split(".")[0]
            names = ", ".join(a.name for a in node.names)
            self.imports.append(f"from {node.module} import {names}")
            if _is_external_module(root_mod):
                self.external_imports.append(root_mod)
        self.generic_visit(node)

    def visit_If(self, node):
        if _is_main_guard(node):
            self.has_main_guard = True
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    cn = _callee_name(child)
                    if cn:
                        self.main_guard_calls.append(cn)
        self.generic_visit(node)


# =====================================================
# ファイル解析
# =====================================================

def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _parse_file(path: Path, file_rel: str) -> tuple[dict, dict[str, list[str]]]:
    """1ファイルを解析し、FileRecord(dict) と call_edges を返す。

    Returns:
        (file_record, call_edges) — file_record は JSON 直列化可能。
        構文エラー時は parse_error に格納し、空シンボルで返す（例外を出さない）。
    """
    try:
        raw = path.read_bytes()
    except Exception as e:
        return ({"path": file_rel, "md5": "", "symbols": [], "imports": [],
                 "external_imports": [], "parse_error": f"read error: {e}",
                 "n_lines": 0, "has_main_guard": False, "main_guard_calls": []}, {})
    md5 = hashlib.md5(raw).hexdigest()
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            source = raw.decode("cp932")
        except Exception as e:
            return ({"path": file_rel, "md5": md5, "symbols": [], "imports": [],
                     "external_imports": [], "parse_error": f"decode error: {e}",
                     "n_lines": 0, "has_main_guard": False, "main_guard_calls": []}, {})
    n_lines = source.count("\n") + (0 if source.endswith("\n") else 1) if source else 0

    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as e:
        return ({"path": file_rel, "md5": md5, "symbols": [], "imports": [],
                 "external_imports": [], "parse_error": str(e)[:200], "n_lines": n_lines,
                 "has_main_guard": False, "main_guard_calls": []}, {})

    ext = _Extractor(file_rel)
    ext.visit(tree)
    rec = {
        "path": file_rel,
        "md5": md5,
        "symbols": ext.symbols,
        "imports": ext.imports,
        "external_imports": list(dict.fromkeys(ext.external_imports)),
        "parse_error": None,
        "n_lines": n_lines,
        "has_main_guard": ext.has_main_guard,
        "main_guard_calls": ext.main_guard_calls,
        "_call_edges": ext.call_edges,  # call_graph 復元用（キャッシュ保存時も含める）
    }
    return rec, ext.call_edges


def outline(file_path) -> list[dict]:
    """1ファイルの AST からシンボル一覧を返す（get_code_outline 用公開API）。

    Python 専用。file_path は str/Path 両対応。読込失敗・非Python・構文エラー時は
    空リストを返す（例外を出さない）。get_code_outline が _Extractor の精度
    （ネスト関数・デコレータ・正確な end_lineno）を利用するための公開ラッパ。
    """
    try:
        p = Path(file_path)
    except Exception:
        return []
    if not p.is_file() or p.suffix != ".py":
        return []
    try:
        rec, _ = _parse_file(p, p.name)
    except Exception:
        return []
    return list(rec.get("symbols", []))


# =====================================================
# build_index
# =====================================================

def build_index(
    root_dir,
    cache_path=None,
    force: bool = False,
    ignore_dirs=DEFAULT_IGNORE_DIRS,
) -> dict:
    """コードベースを AST 解析しインデックスを構築（MD5 増分キャッシュ付き）。

    Args:
        root_dir: 解析ルートディレクトリ。
        cache_path: JSON キャッシュパス（None ならキャッシュしない）。
        force: True ならキャッシュを無視して全ファイル再解析。
        ignore_dirs: スキップするディレクトリ名。

    Returns:
        IndexResult 相当の dict（schema_version/root/files/call_graph/built_at/stats）。
        1ファイルの構文エラーは全体を止めない（parse_error に記録）。

    Raises:
        NotADirectoryError: root_dir がディレクトリでない。
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    cached = None
    if cache_path and not force and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("schema_version") != SCHEMA_VERSION or cached.get("root") != str(root):
                cached = None
        except Exception:
            cached = None
    cached_files = cached.get("files", {}) if cached else {}

    ignore_set = set(ignore_dirs)
    files: dict[str, dict] = {}
    call_graph: dict[str, list[str]] = {}
    reused = 0
    reparsed = 0
    parse_errors = 0

    for path in sorted(root.rglob("*.py")):
        if any(part in ignore_set for part in path.parts):
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            md5 = _md5_bytes(path.read_bytes())
        except Exception:
            continue

        crec = cached_files.get(rel)
        if crec and crec.get("md5") == md5:
            files[rel] = crec
            reused += 1
            if crec.get("parse_error"):
                parse_errors += 1
            for qn, callees in crec.get("_call_edges", {}).items():
                call_graph[f"{rel}::{qn}"] = callees
        else:
            rec, edges = _parse_file(path, rel)
            files[rel] = rec
            reparsed += 1
            if rec.get("parse_error"):
                parse_errors += 1
            for qn, callees in edges.items():
                call_graph[f"{rel}::{qn}"] = callees

    n_symbols = sum(len(f.get("symbols", [])) for f in files.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "files": files,
        "call_graph": call_graph,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {
            "n_files": len(files),
            "n_symbols": n_symbols,
            "n_parse_errors": parse_errors,
            "reused_from_cache": reused,
            "reparsed": reparsed,
        },
    }

    if cache_path:
        cache_parent = Path(cache_path).parent
        cache_parent.mkdir(parents=True, exist_ok=True)
        tmp = str(cache_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        os.replace(tmp, cache_path)

    return result


# =====================================================
# デッドコード検出
# =====================================================

def find_dead_symbols(
    index: dict,
    include_dynamic_string_check: bool = True,
    seed_decorator_names: tuple[str, ...] = ("register_tool",),
) -> list[dict]:
    """到達性解析でデッドコード「候補」を返す（確定ではない）。

    シード: seed_decorator_names のデコレータが付いた関数 / `__main__` ガード内の
    呼び出し対象 / `main` 関数。call_graph で BFS 到達し、未到達の function/class を候補とする。

    偽陽性緩和:
      - call_graph の callee 名で、同名シンボルがどこかにあれば到達とみなす（動的呼出の逃げ道）。
      - include_dynamic_string_check 時、コードベース全体で name の単語出現が2箇所以上
        （定義行 + 参照）あれば low_confidence=True に格下げ。
    """
    all_syms: list[dict] = []
    sym_by_name: dict[str, list[tuple[str, str]]] = {}
    for rel, fr in index["files"].items():
        for s in fr.get("symbols", []):
            s2 = dict(s)
            s2.setdefault("file", rel)
            all_syms.append(s2)
            sym_by_name.setdefault(s2["name"], []).append((s2["file"], s2["qualname"]))

    seed_keys: set[str] = set()
    for s in all_syms:
        decos = s.get("decorators", []) or []
        if any(d in seed_decorator_names for d in decos):
            seed_keys.add(f"{s['file']}::{s['qualname']}")
    for _rel, fr in index["files"].items():
        if fr.get("has_main_guard"):
            for callee in fr.get("main_guard_calls", []):
                for cf, cqn in sym_by_name.get(callee, []):
                    seed_keys.add(f"{cf}::{cqn}")
            for cf, cqn in sym_by_name.get("main", []):
                seed_keys.add(f"{cf}::{cqn}")

    cg = index["call_graph"]
    reached: set[str] = set()
    queue = deque(seed_keys)
    while queue:
        key = queue.popleft()
        if key in reached:
            continue
        reached.add(key)
        for callee in cg.get(key, []):
            # callee は name または qualname(self.method, obj.fn 等)。最終セグメントで同名探索。
            targets = sym_by_name.get(callee, [])
            if not targets and "." in callee:
                targets = sym_by_name.get(callee.split(".")[-1], [])
            for cf, cqn in targets:
                ckey = f"{cf}::{cqn}"
                if ckey not in reached:
                    queue.append(ckey)

    # import される公開シンボルを到達に追加（呼び出しグラフだけでは import API を
    # 見逃し偽陽性が増えるため。「いずれかのファイルから import されるシンボル」は到達）
    imported_names: set[str] = set()
    for fr in index["files"].values():
        for imp in fr.get("imports", []):
            if imp.startswith("from ") and " import " in imp:
                names_part = imp.split(" import ", 1)[1]
                for n in names_part.split(","):
                    n = n.strip().split(" as ")[0].strip()
                    if n and n != "*":
                        imported_names.add(n)
    for name in imported_names:
        for cf, cqn in sym_by_name.get(name, []):
            reached.add(f"{cf}::{cqn}")

    unreached = [
        s for s in all_syms
        if f"{s['file']}::{s['qualname']}" not in reached
        and f"{s['file']}::{s['qualname']}" not in seed_keys
        and s["kind"] in ("function", "method", "class")
    ]

    if include_dynamic_string_check and unreached:
        root = Path(index["root"])
        file_texts: dict[str, str] = {}
        for rel in index["files"]:
            try:
                file_texts[rel] = (root / rel).read_text(encoding="utf-8", errors="replace")
            except Exception:
                file_texts[rel] = ""
        for s in unreached:
            name = s["name"]
            pat = re.compile(rf"\b{re.escape(name)}\b")
            occ = 0
            for text in file_texts.values():
                occ += len(pat.findall(text))
            s["low_confidence"] = occ >= 2  # 定義行 + 参照があれば低信頼

    return unreached


# =====================================================
# シンボル行範囲
# =====================================================

def get_symbol_range(index: dict, file_path: str, symbol_name: str):
    """インデックスからシンボルの (lineno, end_lineno|None) を返す。未発見は None。"""
    root = Path(index["root"])
    try:
        rel = str(Path(file_path).resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        rel = str(Path(file_path)).replace("\\", "/")
    fr = index["files"].get(rel)
    if not fr:
        return None
    candidates = [s for s in fr.get("symbols", []) if s["name"] == symbol_name]
    if not candidates:
        return None
    candidates.sort(key=lambda s: (s["qualname"].count("."), s["lineno"]))
    s = candidates[0]
    return (s["lineno"], s.get("end_lineno"))


# =====================================================
# 全体サマリ
# =====================================================

def summarize(index: dict, max_chars: int = 12000) -> str:
    """コードベース全体のコンパクトなテキストサマリ（max_chars 以内）。"""
    stats = index["stats"]
    lines: list[str] = []
    lines.append("# コードベース概観（AST インデックス）")
    lines.append("")
    lines.append(
        f"ファイル {stats['n_files']} / シンボル {stats['n_symbols']} / "
        f"解析エラー {stats['n_parse_errors']} / "
        f"キャッシュ再利用 {stats['reused_from_cache']} / 再解析 {stats['reparsed']}"
    )

    lines.append("")
    lines.append("## モジュール（シンボル数順）")
    mods = sorted(index["files"].items(), key=lambda kv: len(kv[1].get("symbols", [])), reverse=True)
    for rel, fr in mods[:40]:
        exts = ", ".join(fr.get("external_imports", [])[:6])
        suffix = f"  import: {exts}" if exts else ""
        lines.append(f"- `{rel}` — {len(fr.get('symbols', []))}シンボル{suffix}")
    if len(mods) > 40:
        lines.append(f"- …(+{len(mods) - 40} files)")

    lines.append("")
    lines.append("## 外部依存")
    all_ext = sorted({e for fr in index["files"].values() for e in fr.get("external_imports", [])})
    lines.append(", ".join(f"`{e}`" for e in all_ext) if all_ext else "(なし — 標準ライブラリのみ)")

    lines.append("")
    lines.append("## 複雑度ホットスポット（行数順 top 12）")
    hotspots = []
    for rel, fr in index["files"].items():
        for s in fr.get("symbols", []):
            el = s.get("end_lineno")
            if el and s.get("lineno"):
                hotspots.append((rel, s["qualname"], s["kind"], el - s["lineno"] + 1))
    hotspots.sort(key=lambda x: x[3], reverse=True)
    for rel, qn, kind, n in hotspots[:12]:
        lines.append(f"- `{rel}::{qn}` ({kind}, {n}行)")

    n_dead = len(find_dead_symbols(index, include_dynamic_string_check=False))
    lines.append("")
    lines.append(f"## デッドコード候補: {n_dead} 件 → `detect_dead_code` で詳細確認")

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n…(切り詰め: read_symbol で個別参照)"
    return result
