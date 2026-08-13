"""小型モデル向けWorkset候補の決定的な索引・順位付け。

LLMやembeddingは使わない。Pythonは既存AST index、Markdownは軽量行パーサで解析し、
明示選択の周辺に caller / dependency / test / spec を追加する。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from code_index import build_index

IGNORE_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".pixie_notes",
})
TEXT_SUFFIXES = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".rst", ".txt", ".json",
    ".yaml", ".yml", ".toml",
})
DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})
STOPWORDS = frozenset({
    "この", "ファイル", "コード", "実装", "修正", "変更", "追加", "確認", "して",
    "する", "ください", "the", "and", "with", "file", "code", "change", "update",
})
_REQ_RE = re.compile(r"\b(?:REQ|FR|NFR|SPEC|ADR)-[A-Za-z0-9_.-]+\b", re.I)
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")


def task_terms(task: str) -> set[str]:
    terms = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}|[ぁ-んァ-ヶ一-龠]{2,}", task.lower())
    return {term.strip("._-") for term in terms
            if term.strip("._-") and term not in STOPWORDS}


def markdown_index(text: str) -> dict:
    """Markdownの見出し範囲、要件ID、ローカルリンク、Mermaid図を抽出する。"""
    lines = text.splitlines()
    headings = []
    requirements = []
    links = []
    mermaid = []
    in_fence = False
    fence_kind = ""
    fence_start = 0
    mermaid_lines: list[str] = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_kind = stripped[3:].strip().lower().split()[0] if stripped[3:].strip() else ""
                fence_start = number
                mermaid_lines = []
            elif stripped.startswith(marker):
                if fence_kind == "mermaid":
                    src = "\n".join(mermaid_lines)
                    ids = sorted(set(re.findall(r"(?m)^\s*([A-Za-z_][\w.-]*)\s*(?:[\[({]|-->)", src)))
                    mermaid.append({"range": [fence_start, number], "ids": ids[:50]})
                in_fence = False
                fence_kind = ""
            continue
        if in_fence:
            if fence_kind == "mermaid":
                mermaid_lines.append(line)
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append({"heading": match.group(2).strip(), "level": len(match.group(1)),
                             "range": [number, len(lines)]})
        requirements.extend({"id": req.upper(), "line": number} for req in _REQ_RE.findall(line))
        for label, target in _LINK_RE.findall(line):
            if not re.match(r"^[a-z][a-z0-9+.-]*://", target, re.I) and not target.startswith("mailto:"):
                links.append({"label": label, "target": target, "line": number})
    for i, heading in enumerate(headings):
        for nxt in headings[i + 1:]:
            if nxt["level"] <= heading["level"]:
                heading["range"][1] = nxt["range"][0] - 1
                break
    return {
        "sections": headings,
        "requirements": requirements,
        "links": links,
        "mermaid": mermaid,
    }


def _module_candidates(module: str) -> tuple[str, str]:
    base = module.lstrip(".").replace(".", "/")
    return f"{base}.py", f"{base}/__init__.py"


def _import_modules(imports: list[str]) -> set[str]:
    modules = set()
    for value in imports:
        if value.startswith("import "):
            modules.add(value[7:].split()[0])
        elif value.startswith("from ") and " import " in value:
            modules.add(value[5:].split(" import ", 1)[0])
    return modules


def analyze(root: Path, task: str, explicit_paths: list[str],
            buffers: dict[str, dict] | None = None, scan_limit: int = 20_000):
    """関連候補、走査打切り、path別metadata、索引statsを返す。"""
    root = root.resolve()
    buffers = buffers or {}
    terms = task_terms(task)
    explicit_rels = set()
    stems = set()
    for raw in explicit_paths:
        target = Path(raw)
        if not target.is_absolute():
            target = root / target
        try:
            rel = target.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        explicit_rels.add(rel)
        stem = Path(rel).stem.lower().removeprefix("test_").removesuffix("_test")
        if stem:
            stems.add(stem)

    candidates: dict[str, tuple[str, float, str]] = {}
    metadata: dict[str, dict] = {}

    def add(rel: str, role: str, score: float, reason: str, **meta):
        if rel in explicit_rels:
            metadata.setdefault(rel, {}).update(meta)
            return
        old = candidates.get(rel)
        if old is None or score > old[1]:
            candidates[rel] = (role, score, reason)
        if meta:
            metadata.setdefault(rel, {}).update(meta)

    # Python AST: symbol、import依存、逆import、call graph、テスト。
    try:
        code = build_index(str(root), cache_path=str(root / ".pixie_notes" / "code_index.json"))
    except (OSError, ValueError):
        code = {"files": {}, "call_graph": {}, "stats": {"n_files": 0, "n_symbols": 0}}
    files = code.get("files", {})
    explicit_symbols = set()
    target_modules = set()
    for rel in explicit_rels:
        record = files.get(rel)
        if not record:
            continue
        symbols = record.get("symbols", [])
        selected = [s for s in symbols if not terms or any(t in s["name"].lower() for t in terms)]
        selected = (selected or symbols)[:30]
        explicit_symbols.update(s["name"] for s in selected)
        metadata.setdefault(rel, {})["symbols"] = [
            {"name": s["qualname"], "kind": s["kind"],
             "range": [s["lineno"], s.get("end_lineno") or s["lineno"]]}
            for s in selected
        ]
        target_modules.add(rel[:-3].replace("/", ".") if rel.endswith(".py") else "")
        for module in _import_modules(record.get("imports", [])):
            for dep in _module_candidates(module):
                if dep in files:
                    add(dep, "dependency", 0.80, "import_dependency")

    for rel, record in files.items():
        symbols = record.get("symbols", [])
        matching = [s for s in symbols if any(t in s["name"].lower() for t in terms if len(t) >= 3)]
        if matching:
            add(rel, "symbol", 0.78, "task_symbol", symbols=[
                {"name": s["qualname"], "kind": s["kind"],
                 "range": [s["lineno"], s.get("end_lineno") or s["lineno"]]}
                for s in matching[:20]
            ])
        imports = _import_modules(record.get("imports", []))
        if any(module and (module in imports or module.split(".")[-1] in imports)
               for module in target_modules):
            add(rel, "caller", 0.81, "reverse_import")

    for key, callees in code.get("call_graph", {}).items():
        matched = [name for name in explicit_symbols
                   if any(callee == name or callee.endswith("." + name) for callee in callees)]
        if matched:
            rel, qualname = key.split("::", 1)
            add(rel, "caller", 0.84, "calls_target", symbols=[qualname], calls=matched)

    # ファイル名とMarkdown本文索引。20k件で確実に止める。
    scanned = 0
    truncated = False
    doc_files = 0
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS and not d.startswith("."))
        for name in sorted(names):
            scanned += 1
            if scanned > scan_limit:
                truncated = True
                break
            path = Path(dirpath) / name
            suffix = path.suffix.lower()
            if suffix not in TEXT_SUFFIXES:
                continue
            rel = path.relative_to(root).as_posix()
            low = rel.lower()
            stem = path.stem.lower().removeprefix("test_").removesuffix("_test")
            test_like = "tests" in {part.lower() for part in path.parts} or name.lower().startswith("test_")
            if test_like and stem in stems:
                add(rel, "test", 0.82, "matching_test")
            elif any(term in low for term in terms if len(term) >= 3):
                add(rel, "related", 0.68, "task_filename")
            if suffix not in DOC_SUFFIXES:
                continue
            doc_files += 1
            try:
                buffered = buffers.get(str(path.resolve()))
                text = buffered["content"] if buffered is not None else path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            doc = markdown_index(text) if suffix == ".md" else {
                "sections": [], "requirements": [], "links": [], "mermaid": [],
            }
            metadata.setdefault(rel, {}).update(doc)
            haystack = " ".join(
                [rel, *(section["heading"] for section in doc["sections"]),
                 *(req["id"] for req in doc["requirements"])]
            ).lower()
            linked = any(link["target"].split("#", 1)[0].replace("\\", "/").endswith(tuple(explicit_rels))
                         for link in doc["links"]) if explicit_rels else False
            if linked:
                add(rel, "spec", 0.77, "links_target")
            elif any(term in haystack for term in terms if len(term) >= 3):
                add(rel, "spec", 0.76, "matching_section")
            elif any(stem in low for stem in stems if len(stem) >= 3):
                add(rel, "spec", 0.74, "matching_document")
        if truncated:
            break

    related = [(rel, *values) for rel, values in candidates.items()]
    related.sort(key=lambda item: (-item[2], item[0]))
    stats = {
        "code_files": code.get("stats", {}).get("n_files", 0),
        "symbols": code.get("stats", {}).get("n_symbols", 0),
        "document_files": doc_files,
    }
    return related, truncated, metadata, stats
