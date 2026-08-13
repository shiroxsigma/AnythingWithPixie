"""複数Markdown変更のリンク・要件・用語・Mermaid整合性検査。"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from .workset import markdown_index

_TERM_RE = re.compile(r"^\s*\*\*([^*]{2,80})\*\*\s*[:：]\s*(.+?)\s*$")
_REQ_RE = re.compile(r"\b(?:REQ|FR|NFR|SPEC|ADR)-[A-Za-z0-9_.-]+\b", re.I)
_FENCE_RE = re.compile(r"^\s*(```|~~~)\s*([^\s]*)")


def _slug(heading: str) -> str:
    value = heading.strip().lower()
    value = re.sub(r"[^\w\-\sぁ-んァ-ヶ一-龠]", "", value)
    return re.sub(r"[\s-]+", "-", value).strip("-")


def _definitions(text: str) -> list[dict]:
    result = []
    in_fence = False
    for line_no, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _TERM_RE.match(line)
        if match:
            result.append({"term": match.group(1).strip(), "definition": match.group(2).strip(),
                           "line": line_no})
    return result


def _requirement_statements(text: str) -> list[dict]:
    result = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for req in _REQ_RE.findall(line):
            statement = re.sub(r"\s+", " ", line.replace(req, "")).strip(" :-—\t")
            result.append({"id": req.upper(), "statement": statement, "line": line_no})
    return result


def _mermaid_errors(path: str, text: str) -> list[dict]:
    errors = []
    lines = text.splitlines()
    start = None
    marker = ""
    source: list[tuple[int, str]] = []
    for number, line in enumerate(lines, 1):
        match = _FENCE_RE.match(line)
        if start is None:
            if match and match.group(2).lower() == "mermaid":
                start, marker, source = number, match.group(1), []
            continue
        if line.lstrip().startswith(marker):
            joined = "\n".join(value for _, value in source)
            for left, right, label in (("[", "]", "[]"), ("(", ")", "()"), ("{", "}", "{}")):
                if joined.count(left) != joined.count(right):
                    errors.append({"path": path, "line": start, "kind": "mermaid_unbalanced",
                                   "error": f"Mermaidの {label} が対応していません"})
            for line_no, value in source:
                if "-->" in value:
                    before, after = value.split("-->", 1)
                    if not before.strip() or not after.strip():
                        errors.append({"path": path, "line": line_no, "kind": "mermaid_arrow",
                                       "error": "Mermaidの矢印の両端が必要です"})
            start = None
            continue
        source.append((number, line))
    if start is not None:
        errors.append({"path": path, "line": start, "kind": "mermaid_unclosed",
                       "error": "Mermaidコードフェンスが閉じられていません"})
    return errors


def validate_changes(root: Path, changes: list[dict]) -> dict:
    """ChangeSet preview結果中のMarkdownだけを検証する。"""
    root = root.resolve()
    changed = {item["path"]: item["after"] for item in changes
               if item["path"].lower().endswith(".md") and not item.get("delete")}
    deleted = {item["path"] for item in changes if item.get("delete")}
    errors = []
    warnings = []
    matrix = []
    indexes = {path: markdown_index(text) for path, text in changed.items()}

    for path, text in changed.items():
        index = indexes[path]
        matrix.append({"path": path, "sections": len(index["sections"]),
                       "requirements": len(index["requirements"]), "links": len(index["links"]),
                       "mermaid": len(index["mermaid"])})
        errors.extend(_mermaid_errors(path, text))
        slugs = [_slug(item["heading"]) for item in index["sections"]]
        for slug in sorted({value for value in slugs if slugs.count(value) > 1}):
            warnings.append({"path": path, "kind": "duplicate_heading",
                             "warning": f"同じ見出しアンカーが重複しています: #{slug}"})
        doc_dir = (root / path).parent
        for link in index["links"]:
            raw_target = unquote(link["target"])
            file_part, _, anchor = raw_target.partition("#")
            target = (doc_dir / file_part).resolve() if file_part else (root / path).resolve()
            try:
                target_rel = target.relative_to(root).as_posix()
            except ValueError:
                errors.append({"path": path, "line": link["line"], "kind": "link_outside",
                               "error": f"workspace外へのリンクです: {raw_target}"})
                continue
            if target_rel in deleted:
                errors.append({"path": path, "line": link["line"], "kind": "link_deleted",
                               "error": f"削除対象へのリンクです: {raw_target}"})
                continue
            target_text = changed.get(target_rel)
            if target_text is None:
                if not target.is_file():
                    errors.append({"path": path, "line": link["line"], "kind": "link_missing",
                                   "error": f"リンク先が見つかりません: {raw_target}"})
                    continue
                if anchor and target.suffix.lower() == ".md":
                    try:
                        target_text = target.read_text(encoding="utf-8")
                    except (OSError, UnicodeError):
                        target_text = None
            if anchor and target_text is not None and target.suffix.lower() == ".md":
                target_slugs = {_slug(item["heading"]) for item in markdown_index(target_text)["sections"]}
                if anchor.lower() not in target_slugs:
                    errors.append({"path": path, "line": link["line"], "kind": "anchor_missing",
                                   "error": f"見出しアンカーが見つかりません: {raw_target}"})

    requirements: dict[str, list[dict]] = {}
    terms: dict[str, list[dict]] = {}
    for path, text in changed.items():
        for item in _requirement_statements(text):
            requirements.setdefault(item["id"], []).append({"path": path, **item})
        for item in _definitions(text):
            terms.setdefault(item["term"].casefold(), []).append({"path": path, **item})
    for req_id, entries in requirements.items():
        statements = {entry["statement"].casefold() for entry in entries if entry["statement"]}
        paths = {entry["path"] for entry in entries}
        if len(paths) > 1 and len(statements) > 1:
            errors.append({"kind": "requirement_conflict", "id": req_id,
                           "paths": sorted(paths),
                           "error": f"同じ要件ID {req_id} の記述が文書間で一致しません"})
    for entries in terms.values():
        definitions = {entry["definition"].casefold() for entry in entries}
        paths = {entry["path"] for entry in entries}
        if len(paths) > 1 and len(definitions) > 1:
            errors.append({"kind": "term_conflict", "term": entries[0]["term"],
                           "paths": sorted(paths),
                           "error": f"用語「{entries[0]['term']}」の定義が文書間で一致しません"})
    return {"ok": not errors, "errors": errors, "warnings": warnings, "matrix": matrix}
