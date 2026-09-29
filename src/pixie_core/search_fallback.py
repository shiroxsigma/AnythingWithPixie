"""Bounded file search when Windows has no ripgrep executable."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .turn_control import active_control

MAX_FILES = 2_000
MAX_ENTRIES = 20_000
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_MATCHES = 10_000
MAX_LINE_CHARS = 12_000
MAX_OUTPUT_CHARS = 10_000
_SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", "__pycache__"})


def search(pattern: str, target: Path, display_path: str, *, use_regex: bool,
           file_extensions: str | None, context_lines: int) -> str:
    """Return grep_search-compatible output without unbounded traversal or output."""
    if not pattern:
        return "Error: 検索パターンを指定してください。"
    try:
        matcher = re.compile(pattern) if use_regex else None
    except re.error as exc:
        return f"Error: 正規表現が無効です: {exc}"

    extensions = {"." + part.strip().lstrip(".") for part in (file_extensions or "").split(",") if part.strip(". ")}
    control = active_control.get()
    stack = [target]
    entries_seen = files_seen = bytes_seen = total_matches = 0
    per_file: list[str] = []
    body: list[str] = []
    body_length = 0
    scan_truncated = output_truncated = False

    def add_output(chunk: str) -> None:
        nonlocal body_length, output_truncated
        remaining = 7_500 - body_length
        if remaining <= 0:
            output_truncated = True
        elif len(chunk) > remaining:
            body.append(chunk[:remaining])
            body_length += remaining
            output_truncated = True
        else:
            body.append(chunk)
            body_length += len(chunk)

    while stack:
        if control is not None:
            control.check()
        path = stack.pop()
        if path.is_symlink() or path.name.startswith(".") or path.name in _SKIP_DIRS:
            continue
        try:
            if path.is_dir():
                children = []
                with os.scandir(path) as iterator:
                    for entry in iterator:
                        entries_seen += 1
                        if control is not None and entries_seen % 128 == 0:
                            control.check()
                        if entries_seen > MAX_ENTRIES:
                            scan_truncated = True
                            break
                        if entry.name.startswith(".") or entry.name in _SKIP_DIRS or entry.is_symlink():
                            continue
                        children.append(Path(entry.path))
                stack.extend(sorted(children, key=lambda item: item.name, reverse=True))
                if scan_truncated:
                    break
                continue
            if not path.is_file() or (extensions and path.suffix not in extensions):
                continue
            if files_seen >= MAX_FILES:
                scan_truncated = True
                break
            files_seen += 1
            size = path.stat().st_size
            if size > MAX_FILE_BYTES or bytes_seen + size > MAX_TOTAL_BYTES:
                scan_truncated = True
                continue
            # A file can grow between stat and read; retain the same memory cap.
            with path.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
        except OSError:
            scan_truncated = True
            continue
        if len(raw) > MAX_FILE_BYTES or b"\x00" in raw:
            if len(raw) > MAX_FILE_BYTES:
                scan_truncated = True
            continue
        bytes_seen += len(raw)
        lines = raw.decode("utf-8-sig", errors="replace").splitlines()
        matching: set[int] = set()
        file_matches = 0
        for index, line in enumerate(lines):
            if control is not None and index % 128 == 0:
                control.check()
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS]
                scan_truncated = True
            count = len(list(matcher.finditer(line))) if matcher is not None else line.count(pattern)
            if count:
                matching.add(index)
                accepted = min(count, MAX_MATCHES - total_matches)
                file_matches += accepted
                total_matches += accepted
            if total_matches >= MAX_MATCHES:
                scan_truncated = True
                break
        if not matching:
            if total_matches >= MAX_MATCHES:
                break
            continue
        per_file.append(f"{path}: {file_matches}")
        selected: set[int] = set()
        for index in matching:
            selected.update(range(max(0, index - context_lines), min(len(lines), index + context_lines + 1)))
        add_output(("\n\n" if body else "") + str(path) + "\n")
        previous = -1
        for index in sorted(selected):
            if previous >= 0 and index > previous + 1:
                add_output("--\n")
            marker = ":" if index in matching else "-"
            line = lines[index][:300]
            add_output(f"{index + 1}{marker}{line}\n")
            previous = index
        if total_matches >= MAX_MATCHES:
            break

    if not per_file:
        result = f"No matches found for pattern '{pattern}' in {display_path}."
        return result + (" (truncated=true; scan limit reached)" if scan_truncated else "")

    summary_files = []
    summary_size = 0
    for item in per_file:
        if summary_size + len(item) > 1_500:
            output_truncated = True
            break
        summary_files.append(item)
        summary_size += len(item) + 2
    authority = "partial" if scan_truncated else "authoritative"
    summary = (f"\n\nSearch summary ({authority}): total_matches={total_matches}; "
               f"per_file=[{'; '.join(summary_files)}].")
    if not scan_truncated:
        summary += " Use total_matches as the answer without recalculating it."
    if scan_truncated or output_truncated:
        summary += f" truncated=true (scan={str(scan_truncated).lower()}, output={str(output_truncated).lower()})."
    result = "".join(body).rstrip() + summary
    return result[:MAX_OUTPUT_CHARS]
