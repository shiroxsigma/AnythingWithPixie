"""ユーザーが明示した決定論的な受け入れ条件の抽出と検証。"""
from __future__ import annotations

import hashlib
import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

_SEMVER_RE = re.compile(
    r"(?<![0-9A-Za-z.])(?:v)?(\d+)\.(\d+)\.(\d+)(?![0-9A-Za-z.])"
)
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？；;\r\n]")
_INCREMENT_RE = re.compile(
    r"(?P<component>major|minor|patch|メジャー|マイナー|パッチ)"
    r"(?:\s*(?:version|バージョン))?(?:\s*(?:number|番号|成分))?"
    r"(?:\s*だけ)?\s*(?:を)?\s*(?P<delta>\d+)\s*(?:つ|だけ)?\s*"
    r"(?P<verb>増や|上げ|加算)",
    re.IGNORECASE,
)
_TARGET_ACTION_RE = re.compile(r"(?:追加|追記|反映|記載|書き込|更新)")
_TARGET_TAIL_RE = re.compile(
    r"\s*(?:の\s*見出し\s*(?:の)?\s*直後)?\s*(?:へ|に)"
    r"(?:\s*(?:新しい|新規の?)?\s*(?:バージョン(?:の)?|リリース(?:の)?|"
    r"セクション(?:を)?|見出し(?:を)?))?\s*$"
)
_TARGET_JOIN_RE = re.compile(r"\s*(?:と|、|,|および|及び|ならびに|and)\s*$", re.IGNORECASE)
_UNCHANGED_SUFFIX_RE = re.compile(
    r"\s*(?:自体)?\s*(?:(?:に)?(?:は|を))?\s*(?:絶対に\s*)?"
    r"(?:変更しない(?:で(?:ください)?)?|変更せず|"
    r"そのまま(?:に(?:して(?:ください)?|保つ)?)?|"
    r"保持(?:して(?:ください)?)?|触らない(?:で(?:ください)?)?)"
)
_COMPONENT_INDEX = {
    "major": 0,
    "メジャー": 0,
    "minor": 1,
    "マイナー": 1,
    "patch": 2,
    "パッチ": 2,
}


@dataclass(frozen=True)
class _PathMention:
    path: str
    start: int
    end: int


@dataclass(frozen=True)
class _IncrementOperation:
    component: int
    delta: int
    start: int
    end: int


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_path(raw: str) -> str:
    """プロンプト中の区切りだけを正規化し、snapshot のキーと照合する。"""
    normalized = posixpath.normpath(raw.replace("\\", "/"))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _snapshot_index(snapshots: dict[str, dict]) -> dict[str, str]:
    """正規化したパスから、呼び出し元が使っている元のキーへ引く。"""
    index: dict[str, str] = {}
    ambiguous: set[str] = set()
    for raw in snapshots:
        normalized = _normalize_path(str(raw))
        if normalized in index and index[normalized] != raw:
            ambiguous.add(normalized)
        else:
            index[normalized] = raw
    for normalized in ambiguous:
        index.pop(normalized, None)
    return index


def _path_mentions(user_text: str, snapshots: dict[str, dict]) -> list[_PathMention]:
    """既知のsnapshot pathを原文上で照合し、文字位置とpathの関係を保持する。"""
    index = _snapshot_index(snapshots)
    candidates: list[tuple[_PathMention, int]] = []
    path_chars = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./\\-")
    before_connectors = frozenset("をにへはがとやの")
    after_connectors = frozenset("をにへはがとやので自")
    casefold_counts: dict[str, int] = {}
    for normalized in index:
        folded = normalized.casefold()
        casefold_counts[folded] = casefold_counts.get(folded, 0) + 1
    for normalized, snapshot_path in index.items():
        windows = normalized.replace("/", "\\")
        variants = {normalized, windows, f"./{normalized}", f".\\{windows}"}
        for variant in sorted(variants, key=len, reverse=True):
            if not variant:
                continue
            searches = [(0, 0)]
            if os.name == "nt" and casefold_counts[normalized.casefold()] == 1:
                searches.append((re.IGNORECASE, 1))
            for flags, priority in searches:
                for match in re.finditer(re.escape(variant), user_text, flags):
                    before = user_text[match.start() - 1] if match.start() else ""
                    after = user_text[match.end()] if match.end() < len(user_text) else ""
                    if (
                        before in path_chars
                        or (before and before.isalnum() and before not in before_connectors)
                        or after in path_chars
                        or (after and after.isalnum() and after not in after_connectors)
                    ):
                        continue
                    candidates.append((
                        _PathMention(str(snapshot_path), match.start(), match.end()),
                        priority,
                    ))

    # ``./src/a.py`` と ``src/a.py`` のような重複候補は長い方だけを残す。
    mentions: list[_PathMention] = []
    ordered = sorted(
        candidates,
        key=lambda item: (item[0].start, item[1], -(item[0].end - item[0].start)),
    )
    for candidate, _priority in ordered:
        if any(candidate.start < kept.end and kept.start < candidate.end for kept in mentions):
            continue
        mentions.append(candidate)
    return mentions


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    sentence_start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text, 0, start):
        sentence_start = match.end()
    boundary = _SENTENCE_BOUNDARY_RE.search(text, end)
    sentence_end = boundary.start() if boundary else len(text)
    return sentence_start, sentence_end


def _increments(user_text: str) -> list[_IncrementOperation]:
    operations = []
    for match in _INCREMENT_RE.finditer(user_text):
        delta = int(match.group("delta"))
        if delta <= 0:
            continue
        component = _COMPONENT_INDEX[match.group("component").lower()]
        operations.append(_IncrementOperation(component, delta, match.start(), match.end()))
    return operations


def _single_semver(content: str) -> tuple[int, int, int] | None:
    versions = _SEMVER_RE.findall(content)
    if len(versions) != 1:
        return None
    return tuple(map(int, versions[0]))


def _bump(version: tuple[int, int, int], component: int, delta: int) -> str:
    major, minor, patch = version
    if component == 0:
        major += delta
        minor = 0
        patch = 0
    elif component == 1:
        minor += delta
        patch = 0
    else:
        patch += delta
    return f"{major}.{minor}.{patch}"


def _token_pattern(expected: str) -> re.Pattern[str]:
    # ``v2.7.10`` は一般的な表記として許可する一方、識別子の一部や 2.7.100 は拒否する。
    return re.compile(rf"(?<![0-9A-Za-z.])(?:v)?{re.escape(expected)}(?![0-9A-Za-z.])")


def _is_markdown(path: str) -> bool:
    return Path(path).suffix.lower() in {".md", ".markdown"}


def _expected_count(path: str, content: str, expected: str) -> int:
    token = _token_pattern(expected)
    if not _is_markdown(path):
        return len(token.findall(content))

    count = 0
    for line in content.splitlines():
        if re.match(r"^[ \t]{0,3}#{1,6}[ \t]+", line):
            count += len(token.findall(line))
    return count


def _expected_is_first_subheading(content: str, expected: str) -> bool:
    """expected が最初の Markdown 見出し直後のセクション見出しかを判定する。"""
    headings = [
        line for line in content.splitlines()
        if re.match(r"^[ \t]{0,3}#{1,6}[ \t]+", line)
    ]
    if len(headings) < 2:
        return False
    return bool(_token_pattern(expected).search(headings[1]))


def _derive_unchanged(
    user_text: str,
    mentions: list[_PathMention],
    snapshots: dict[str, dict],
) -> list[dict]:
    conditions: list[dict] = []
    seen: set[str] = set()
    for mention in mentions:
        if mention.path in seen:
            continue
        # 述語はパス直後に係るものだけを受理する。別パスの述語を近接距離で拾わない。
        if not _UNCHANGED_SUFFIX_RE.match(user_text, mention.end):
            continue
        snapshot = snapshots[mention.path]
        content = snapshot.get("content")
        initial_hash = snapshot.get("initial_hash")
        if initial_hash is None and isinstance(content, str):
            initial_hash = _digest(content)
        if not isinstance(initial_hash, str):
            continue
        conditions.append({
            "kind": "unchanged",
            "path": mention.path,
            "initial_hash": initial_hash,
        })
        seen.add(mention.path)
    return conditions


def _derive_semver_increment(
    user_text: str,
    mentions: list[_PathMention],
    snapshots: dict[str, dict],
) -> list[dict]:
    operations = _increments(user_text)
    # 複数操作の合成順序は明示的な IR なしには安全に決められないため、推測しない。
    if len(operations) != 1:
        return []
    operation = operations[0]
    sentence_start, sentence_end = _sentence_bounds(user_text, operation.start, operation.end)

    source_candidates = [
        mention
        for mention in mentions
        if sentence_start <= mention.start and mention.end <= operation.start
    ]
    source: _PathMention | None = None
    version: tuple[int, int, int] | None = None
    for mention in reversed(source_candidates):
        content = snapshots[mention.path].get("content")
        if not isinstance(content, str):
            continue
        candidate_version = _single_semver(content)
        if candidate_version is not None:
            source = mention
            version = candidate_version
            break
    if source is None or version is None:
        return []

    action = _TARGET_ACTION_RE.search(user_text, operation.end, sentence_end)
    if action is None:
        return []
    possible_targets = [
        mention for mention in mentions
        if operation.end <= mention.start < action.start() and mention.path != source.path
    ]
    if not possible_targets:
        return []

    # 追加動詞へ直接係る最後のpathを起点に、明示的な列挙接続だけを遡る。
    # ``README.mdを参考にCHANGELOG.mdへ追加`` の参考資料を対象に混ぜない。
    last_target = possible_targets[-1]
    if not _TARGET_TAIL_RE.fullmatch(user_text[last_target.end:action.start()]):
        return []
    target_mentions = [last_target]
    for mention in reversed(possible_targets[:-1]):
        next_mention = target_mentions[0]
        if not _TARGET_JOIN_RE.fullmatch(user_text[mention.end:next_mention.start]):
            break
        target_mentions.insert(0, mention)

    target_paths: list[str] = []
    for mention in target_mentions:
        if mention.path not in target_paths:
            target_paths.append(mention.path)
    if not target_paths:
        return []

    expected = _bump(version, operation.component, operation.delta)
    initial_target_counts: dict[str, int] = {}
    for path in target_paths:
        content = snapshots[path].get("content")
        if not isinstance(content, str):
            return []
        initial_target_counts[path] = _expected_count(path, content, expected)

    placement = None
    target_to_action = user_text[target_mentions[-1].end:action.end()]
    if re.search(r"見出し\s*(?:の)?\s*直後", target_to_action):
        placement = "first_subheading"

    condition = {
        "kind": "semver_component_increment",
        "source_path": source.path,
        "target_paths": target_paths,
        "component": operation.component,
        "delta": operation.delta,
        "expected": expected,
        "initial_target_counts": initial_target_counts,
    }
    if placement is not None:
        condition["placement"] = placement
    return [condition]


def derive(user_text: str, snapshots: dict[str, dict]) -> list[dict]:
    """明示された不変条件と semantic-version 増分を初期内容から決定する。

    条件を確定できない曖昧な文は何も返さない。推測で gate を増やさないことを優先する。
    """
    if not user_text or not snapshots:
        return []
    mentions = _path_mentions(user_text, snapshots)
    if not mentions:
        return []
    return (
        _derive_unchanged(user_text, mentions, snapshots)
        + _derive_semver_increment(user_text, mentions, snapshots)
    )


def _read_workspace_file(
    workspace: Path,
    raw: str,
    current_snapshots: dict[str, object] | None = None,
) -> str | None:
    if current_snapshots is not None:
        normalized = _normalize_path(raw)
        for snapshot_path, value in current_snapshots.items():
            if _normalize_path(str(snapshot_path)) != normalized:
                continue
            if isinstance(value, str):
                return value
            if isinstance(value, dict) and isinstance(value.get("content"), str):
                return value["content"]
            return None
        return None
    try:
        path = (workspace / raw).resolve()
        path.relative_to(workspace)
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return None


def validate(
    root: str | Path,
    conditions: list[dict],
    *,
    current_snapshots: dict[str, object] | None = None,
) -> list[str]:
    """満たされていない条件を、修正ループへ渡せる短い説明として返す。"""
    workspace = Path(root).resolve()
    failures: list[str] = []
    for condition in conditions:
        kind = condition.get("kind")
        if kind == "unchanged":
            raw = condition.get("path")
            initial_hash = condition.get("initial_hash")
            if not isinstance(raw, str) or not isinstance(initial_hash, str):
                failures.append("不変条件の形式が不正です")
                continue
            content = _read_workspace_file(workspace, raw, current_snapshots)
            if content is None or _digest(content) != initial_hash:
                failures.append(f"{raw} は変更しないという条件に違反しています")
        elif kind == "semver_component_increment":
            expected = condition.get("expected")
            targets = condition.get("target_paths")
            initial_counts = condition.get("initial_target_counts")
            if (
                not isinstance(expected, str)
                or _SEMVER_RE.fullmatch(expected) is None
                or not isinstance(targets, list)
                or not targets
                or not isinstance(initial_counts, dict)
            ):
                failures.append("バージョン増分条件の形式が不正です")
                continue

            for raw in targets:
                if not isinstance(raw, str) or raw not in initial_counts:
                    failures.append("バージョン増分条件の対象または初期値が不正です")
                    continue
                content = _read_workspace_file(workspace, raw, current_snapshots)
                initial_count = initial_counts[raw]
                if content is None or not isinstance(initial_count, int) or initial_count < 0:
                    failures.append(f"{raw} のバージョン増分を確認できません")
                    continue
                actual_count = _expected_count(raw, content, expected)
                if actual_count <= initial_count:
                    source = condition.get("source_path", "元ファイル")
                    delta = condition.get("delta", "指定数")
                    target_description = "Markdown 見出し" if _is_markdown(raw) else "対象ファイル"
                    failures.append(
                        f"{source} の値から指定成分を {delta} 増やした期待値 {expected} が"
                        f"{raw} の新しい{target_description}として追加されていません"
                    )
                elif (
                    condition.get("placement") == "first_subheading"
                    and _is_markdown(raw)
                    and not _expected_is_first_subheading(content, expected)
                ):
                    failures.append(
                        f"期待値 {expected} のセクションが {raw} の見出し直後にありません"
                    )
        else:
            failures.append(f"未対応の受け入れ条件です: {kind!r}")
    return failures
