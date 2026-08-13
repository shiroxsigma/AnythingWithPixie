"""複数ファイル編集を preview → validate → apply → revert する純粋な基盤。

LLM や UI には依存しない。編集計算は tools.py と同じ規則を使い、適用時は全対象を
事前検証・一時ファイルへ stage してから os.replace する。journal は workspace 内の
``.pixie_notes/changesets`` に保存し、途中失敗・プロセス再起動後も復元できる。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tools import _fuzzy_apply

SCHEMA_VERSION = "1"
_JOURNAL_DIR = Path(".pixie_notes") / "changesets"
_MAX_FILES = 100
_MAX_TEXT_CHARS = 2_000_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def _safe_target(root: Path, raw: str) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("path が空です")
    target = Path(raw)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    try:
        rel = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"workspace 外の path は使用できません: {raw}") from exc
    return target, rel


def _read_current(target: Path, buffers: dict[str, dict]) -> tuple[str, bytes, bool, str]:
    """現在内容、元bytes、存在、encodingを返す。bufferはUTF-8の仮想ファイルとして扱う。"""
    buffered = buffers.get(str(target))
    if buffered is not None:
        text = buffered["content"]
        return text, text.encode("utf-8"), target.exists(), "utf-8"
    if not target.exists():
        return "", b"", False, "utf-8"
    if not target.is_file():
        raise ValueError("通常ファイルではありません")
    raw = target.read_bytes()
    for encoding in ("utf-8", "cp932"):
        try:
            return raw.decode(encoding), raw, True, encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("UTF-8/CP932 のテキストとして読めません")


def _heading_span(text: str, heading: str) -> tuple[int, int, int] | None:
    """Markdown見出し本文の [start,end) と見出しレベルを返す。コードフェンス内は無視。"""
    wanted = heading.strip().lstrip("#").strip()
    if not wanted:
        return None
    lines = text.splitlines(keepends=True)
    in_fence = False
    start = end = level = None
    offset = 0
    for line in lines:
        logical = line.rstrip("\r\n")
        stripped = logical.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        match = None if in_fence else re.match(
            r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", logical
        )
        if match:
            current_level = len(match.group(1))
            title = match.group(2).strip()
            if start is None and title == wanted:
                start = offset + len(line)
                level = current_level
            elif start is not None and current_level <= level:
                end = offset
                break
        offset += len(line)
    if start is None:
        return None
    return start, len(text) if end is None else end, int(level)


def _apply_frontmatter(text: str, values: dict) -> str:
    if not isinstance(values, dict) or any(not isinstance(k, str) for k in values):
        raise ValueError("frontmatter values は文字列キーのオブジェクトである必要があります")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    body_start = 0
    current: dict[str, str] = {}
    order: list[str] = []
    if lines and lines[0].strip() == "---":
        try:
            close = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        except StopIteration as exc:
            raise ValueError("frontmatter の閉じる --- がありません") from exc
        for line in lines[1:close]:
            if ":" not in line:
                raise ValueError("複雑なYAML frontmatterは安全に更新できません")
            key, value = line.split(":", 1)
            key = key.strip()
            if key:
                current[key] = value.strip()
                order.append(key)
        body_start = close + 1
    for key, value in values.items():
        if value is None:
            current.pop(key, None)
            order = [item for item in order if item != key]
        else:
            if key not in current:
                order.append(key)
            if isinstance(value, (dict, list)):
                current[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, bool):
                current[key] = "true" if value else "false"
            else:
                current[key] = str(value)
    header = ["---", *(f"{key}: {current[key]}" for key in order if key in current), "---"]
    body = lines[body_start:]
    result = newline.join(header + body)
    return result + (newline if text.endswith(("\n", "\r")) else "")


def _apply_operation(content: str, operation: dict) -> tuple[str, str]:
    if not isinstance(operation, dict):
        raise ValueError("operation はオブジェクトである必要があります")
    kind = operation.get("kind")
    if kind in ("write_file", "replace_file"):
        value = operation.get("content")
        if not isinstance(value, str):
            raise ValueError(f"{kind}.content は文字列である必要があります")
        return value, "write"
    if kind == "append":
        value = operation.get("content")
        if not isinstance(value, str):
            raise ValueError("append.content は文字列である必要があります")
        return content + value, "append"
    if kind == "search_replace":
        search, replace = operation.get("search"), operation.get("replace")
        if not isinstance(search, str) or not search:
            raise ValueError("search_replace.search は空でない文字列が必要です")
        if not isinstance(replace, str):
            raise ValueError("search_replace.replace は文字列が必要です")
        count = content.count(search)
        if count == 1:
            return content.replace(search, replace, 1), "exact"
        if count > 1:
            raise ValueError(f"search が {count} 箇所に一致し、一意に特定できません")
        changed, method = _fuzzy_apply(content, search, replace)
        if changed is None:
            raise ValueError("search が見つかりません")
        return changed, method or "fuzzy"
    if kind == "replace_lines":
        try:
            start, end = int(operation["start_line"]), int(operation["end_line"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("replace_lines には整数の start_line/end_line が必要です") from exc
        value = operation.get("content")
        if not isinstance(value, str):
            raise ValueError("replace_lines.content は文字列が必要です")
        lines = content.splitlines(keepends=True)
        if start < 1 or end < start or start > len(lines):
            raise ValueError(f"行範囲が不正です: {start}-{end} / {len(lines)}行")
        replacement = value.splitlines(keepends=True)
        if value and replacement and not value.endswith(("\n", "\r")):
            replacement[-1] += "\r\n" if "\r\n" in content else "\n"
        return "".join(lines[:start - 1] + replacement + lines[min(end, len(lines)):]), "lines"
    if kind in ("replace_section", "insert_after_heading"):
        heading = operation.get("heading")
        value = operation.get("content")
        if not isinstance(heading, str) or not isinstance(value, str):
            raise ValueError(f"{kind} には heading/content が必要です")
        span = _heading_span(content, heading)
        if span is None:
            raise ValueError(f"見出しが見つかりません: {heading}")
        start, end, _level = span
        newline = "\r\n" if "\r\n" in content else "\n"
        value = value.rstrip("\r\n") + newline if value else ""
        if kind == "replace_section":
            return content[:start] + value + content[end:], "section"
        return content[:start] + value + content[start:], "after_heading"
    if kind == "update_frontmatter":
        return _apply_frontmatter(content, operation.get("values")), "frontmatter"
    if kind == "delete_file":
        return "", "delete"
    raise ValueError(f"未対応の operation.kind です: {kind}")


def preview(root_dir: str, changeset: dict, buffers: dict[str, dict] | None = None) -> dict:
    root = Path(root_dir).resolve()
    buffers = buffers or {}
    if not isinstance(changeset, dict):
        raise TypeError("changeset はオブジェクトである必要があります")
    changes = changeset.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("changeset.changes は空でない配列である必要があります")
    if len(changes) > _MAX_FILES:
        raise ValueError(f"1 ChangeSet は {_MAX_FILES} ファイルまでです")

    result_changes = []
    errors = []
    seen = set()
    for index, change in enumerate(changes):
        if not isinstance(change, dict):
            errors.append({"index": index, "error": "change はオブジェクトである必要があります"})
            continue
        raw_path = change.get("path")
        try:
            target, rel = _safe_target(root, raw_path)
        except (TypeError, ValueError) as exc:
            errors.append({"index": index, "path": raw_path, "error": str(exc)})
            continue
        if rel in seen:
            errors.append({"index": index, "path": rel, "error": "同じpathのchangeが重複しています"})
            continue
        seen.add(rel)
        try:
            before, raw_before, existed, encoding = _read_current(target, buffers)
            operations = change.get("operations")
            if not isinstance(operations, list) or not operations:
                raise ValueError("operations は空でない配列である必要があります")
            after = before
            methods = []
            delete = False
            for operation in operations:
                after, method = _apply_operation(after, operation)
                methods.append(method)
                delete = operation.get("kind") == "delete_file"
            if len(after) > _MAX_TEXT_CHARS:
                raise ValueError(f"適用後内容が上限 {_MAX_TEXT_CHARS} 文字を超えます")
            expected = change.get("base_hash")
            actual = _hash_bytes(raw_before)
            conflict = isinstance(expected, str) and bool(expected) and expected != actual
            result_changes.append({
                "path": rel, "before": before, "after": after, "existed": existed,
                "delete": delete, "encoding": encoding, "base_hash": actual,
                "expected_base_hash": expected,
                "after_hash": None if delete else _hash_bytes(after.encode(encoding)),
                "conflict": conflict, "methods": methods,
            })
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append({"index": index, "path": rel, "error": str(exc)})
    return {
        "schema_version": SCHEMA_VERSION,
        "id": str(changeset.get("id") or f"chg_{uuid.uuid4().hex[:16]}"),
        "ok": not errors and not any(item["conflict"] for item in result_changes),
        "changes": result_changes,
        "errors": errors,
    }


def validate(root_dir: str, changeset: dict, buffers: dict[str, dict] | None = None) -> dict:
    result = preview(root_dir, changeset, buffers)
    conflicts = [{"path": item["path"], "expected": item["expected_base_hash"],
                  "actual": item["base_hash"]} for item in result["changes"] if item["conflict"]]
    result["conflicts"] = conflicts
    result["ok"] = not result["errors"] and not conflicts
    return result


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pixie-tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _journal_path(root: Path, change_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", change_id)[:100]
    return root / _JOURNAL_DIR / f"{safe_id}.json"


def _save_journal(path: Path, journal: dict) -> None:
    _atomic_write(path, json.dumps(journal, ensure_ascii=False, indent=2).encode("utf-8"))


def apply(root_dir: str, changeset: dict, buffers: dict[str, dict] | None = None) -> dict:
    root = Path(root_dir).resolve()
    checked = validate(str(root), changeset, buffers)
    if not checked["ok"]:
        return {**checked, "applied": False}
    entries = []
    for item in checked["changes"]:
        before_raw = item["before"].encode(item["encoding"])
        after_raw = b"" if item["delete"] else item["after"].encode(item["encoding"])
        entries.append({
            "path": item["path"], "existed": item["existed"], "delete": item["delete"],
            "encoding": item["encoding"], "before_b64": base64.b64encode(before_raw).decode("ascii"),
            "before_hash": _hash_bytes(before_raw), "after_hash": None if item["delete"] else _hash_bytes(after_raw),
            "after_b64": base64.b64encode(after_raw).decode("ascii"),
        })
    journal = {"schema_version": SCHEMA_VERSION, "id": checked["id"], "status": "prepared",
               "created_at": _now(), "updated_at": _now(), "entries": entries, "applied_paths": []}
    journal_path = _journal_path(root, checked["id"])
    _save_journal(journal_path, journal)
    try:
        for entry in entries:
            target, _ = _safe_target(root, entry["path"])
            if entry["delete"]:
                if target.exists():
                    target.unlink()
            else:
                _atomic_write(target, base64.b64decode(entry["after_b64"]))
            journal["applied_paths"].append(entry["path"])
            journal["status"] = "applying"
            journal["updated_at"] = _now()
            _save_journal(journal_path, journal)
    except BaseException as exc:
        journal["status"] = "apply_failed"
        journal["error"] = f"{type(exc).__name__}: {exc}"
        journal["updated_at"] = _now()
        _save_journal(journal_path, journal)
        reverted = revert(str(root), checked["id"], force=True)
        return {**checked, "applied": False, "error": journal["error"], "rollback": reverted,
                "journal": str(journal_path.relative_to(root)).replace("\\", "/")}
    journal["status"] = "applied"
    journal["updated_at"] = _now()
    _save_journal(journal_path, journal)
    return {**checked, "applied": True,
            "journal": str(journal_path.relative_to(root)).replace("\\", "/")}


def revert(root_dir: str, change_id: str, force: bool = False) -> dict:
    root = Path(root_dir).resolve()
    path = _journal_path(root, str(change_id))
    if not path.is_file():
        return {"ok": False, "reverted": False, "error": "journal が見つかりません"}
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "reverted": False, "error": f"journal を読めません: {exc}"}
    conflicts = []
    for entry in journal.get("entries", []):
        target, rel = _safe_target(root, entry["path"])
        expected = entry.get("after_hash")
        actual = _hash_bytes(target.read_bytes()) if target.is_file() else None
        if not force and expected != actual:
            conflicts.append({"path": rel, "expected": expected, "actual": actual})
    if conflicts:
        return {"ok": False, "reverted": False, "conflicts": conflicts}
    restored = []
    try:
        for entry in reversed(journal.get("entries", [])):
            target, rel = _safe_target(root, entry["path"])
            if entry.get("existed"):
                _atomic_write(target, base64.b64decode(entry["before_b64"]))
            elif target.exists():
                target.unlink()
            restored.append(rel)
    except (OSError, ValueError) as exc:
        return {"ok": False, "reverted": False, "error": str(exc), "restored": restored}
    journal["status"] = "reverted"
    journal["updated_at"] = _now()
    _save_journal(path, journal)
    return {"ok": True, "reverted": True, "id": journal.get("id"), "restored": restored,
            "journal": str(path.relative_to(root)).replace("\\", "/")}
