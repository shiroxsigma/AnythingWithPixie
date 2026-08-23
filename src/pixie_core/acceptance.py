"""ユーザーが明示した決定論的な受け入れ条件の抽出と検証。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./\\-])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12})"
    r"(?![A-Za-z0-9_./\\-])"
)
_SEMVER_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive(user_text: str, snapshots: dict[str, dict]) -> list[dict]:
    """明示された不変条件とsemantic-version増分を初期内容から決定する。

    条件を確定できない曖昧な文は何も返さない。推測でgateを増やさないことを優先する。
    """
    if not user_text or not snapshots:
        return []
    mentioned = []
    for raw in _PATH_RE.findall(user_text):
        normalized = raw.replace("\\", "/")
        if normalized in snapshots and normalized not in mentioned:
            mentioned.append(normalized)

    conditions: list[dict] = []
    for path in mentioned:
        occurrences = [match.start() for match in re.finditer(re.escape(path), user_text)]
        nearby_segments = [re.split(r"[。\n]", user_text[pos:pos + len(path) + 30], maxsplit=1)[0]
                           for pos in occurrences]
        if any(re.search(r"(?:変更しない|変更せず|保持|そのまま|触らない)", segment)
               for segment in nearby_segments):
            conditions.append({"kind": "unchanged", "path": path,
                               "initial_hash": snapshots[path]["initial_hash"]})

    component = None
    for label, index in (("major", 0), ("メジャー", 0), ("minor", 1), ("マイナー", 1),
                         ("patch", 2), ("パッチ", 2)):
        if re.search(rf"{label}(?:番号)?を\s*\d+\s*(?:増や|上げ|加算)", user_text, re.I):
            component = index
            break
    delta_match = re.search(
        r"(?:major|minor|patch|メジャー|マイナー|パッチ)(?:番号)?を\s*(\d+)\s*(?:増や|上げ|加算)",
        user_text, re.I,
    )
    if component is not None and delta_match:
        delta = int(delta_match.group(1))
        sources = []
        for path in mentioned:
            versions = _SEMVER_RE.findall(snapshots[path]["content"])
            if len(versions) == 1:
                sources.append((path, tuple(map(int, versions[0]))))
        if len(sources) == 1:
            source_path, version = sources[0]
            expected = list(version)
            expected[component] += delta
            conditions.append({
                "kind": "semver_component_increment",
                "source_path": source_path,
                "target_paths": [path for path in mentioned if path != source_path],
                "component": component,
                "delta": delta,
                "expected": ".".join(map(str, expected)),
            })
    return conditions


def validate(root: str | Path, conditions: list[dict]) -> list[str]:
    """満たされていない条件を、修正ループへ渡せる短い説明として返す。"""
    workspace = Path(root).resolve()
    failures = []
    for condition in conditions:
        kind = condition.get("kind")
        if kind == "unchanged":
            path = workspace / condition["path"]
            try:
                actual = _digest(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                actual = None
            if actual != condition["initial_hash"]:
                failures.append(f"{condition['path']} は変更しないという条件に違反しています")
        elif kind == "semver_component_increment":
            expected = condition["expected"]
            targets = condition.get("target_paths") or []
            found = False
            for raw in targets:
                try:
                    if expected in (workspace / raw).read_text(encoding="utf-8"):
                        found = True
                        break
                except (OSError, UnicodeError):
                    continue
            if not found:
                failures.append(
                    f"{condition['source_path']} の値から指定成分を {condition['delta']} 増やした"
                    f"期待値 {expected} が対象ファイルにありません"
                )
    return failures
