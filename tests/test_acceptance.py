import hashlib

import pytest

from pixie_core.acceptance import derive, validate


def _snap(content):
    return {"content": content, "initial_hash": hashlib.sha256(content.encode()).hexdigest()}


def _condition(conditions, kind):
    return next(condition for condition in conditions if condition["kind"] == kind)


def test_derives_real_c05_roles_increment_and_preservation(tmp_path):
    version = "__version__ = '2.7.9'\n"
    changelog = "# Changelog\n\n## 2.7.9\n\n- Previous release\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 2.8.0\n\n- Incorrect release\n\n" + changelog.split("\n", 2)[2],
        encoding="utf-8",
    )
    snapshots = {
        "version.py": _snap(version),
        "CHANGELOG.md": _snap(changelog),
    }

    conditions = derive(
        "version.py の現在値を読み、パッチ番号を1増やしたバージョンのセクションを"
        "CHANGELOG.mdの見出し直後へ追加してください。version.py自体は変更しないでください。",
        snapshots,
    )

    increment = _condition(conditions, "semver_component_increment")
    assert increment == {
        "kind": "semver_component_increment",
        "source_path": "version.py",
        "target_paths": ["CHANGELOG.md"],
        "component": 2,
        "delta": 1,
        "expected": "2.7.10",
        "initial_target_counts": {"CHANGELOG.md": 0},
        "placement": "first_subheading",
    }
    assert _condition(conditions, "unchanged")["path"] == "version.py"
    assert any("2.7.10" in failure for failure in validate(tmp_path, conditions))


def test_acceptance_passes_when_relation_and_preservation_hold(tmp_path):
    version = "__version__ = '2.7.9'\n"
    changelog = "# Changelog\n\n## 2.7.9\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 2.7.10\n\n- New release\n\n## 2.7.9\n",
        encoding="utf-8",
    )
    snapshots = {"version.py": _snap(version), "CHANGELOG.md": _snap(changelog)}
    conditions = derive(
        "version.py のパッチ番号を1増やした値をCHANGELOG.mdへ追加。"
        "version.pyは変更しない。",
        snapshots,
    )

    assert validate(tmp_path, conditions) == []


def test_unchanged_predicate_is_tied_to_its_own_path():
    snapshots = {"a.py": _snap("a = 1\n"), "b.py": _snap("b = 1\n")}

    conditions = derive("a.pyを更新し、b.pyは変更しないでください。", snapshots)

    assert [condition["path"] for condition in conditions if condition["kind"] == "unchanged"] == [
        "b.py"
    ]


def test_windows_paths_are_normalized_for_snapshot_lookup():
    snapshots = {
        "src/version.py": _snap("__version__ = '2.7.9'\n"),
        "docs/CHANGELOG.md": _snap("# Changelog\n\n## 2.7.9\n"),
    }

    conditions = derive(
        r"src\version.py のパッチ番号を1増やした値を docs\CHANGELOG.md へ追加。"
        r"src\version.py 自体は変更しない。",
        snapshots,
    )

    increment = _condition(conditions, "semver_component_increment")
    assert increment["source_path"] == "src/version.py"
    assert increment["target_paths"] == ["docs/CHANGELOG.md"]
    assert _condition(conditions, "unchanged")["path"] == "src/version.py"


def test_unicode_and_dot_relative_paths_keep_their_roles():
    snapshots = {
        "src/version.py": _snap("__version__ = '2.7.9'\n"),
        "docs/変更履歴.md": _snap("# 変更履歴\n"),
    }

    conditions = derive(
        "./src/version.py のパッチ番号を1増やした値を"
        "docs/変更履歴.mdへ追加し、./src/version.pyは変更しない。",
        snapshots,
    )

    increment = _condition(conditions, "semver_component_increment")
    assert increment["source_path"] == "src/version.py"
    assert increment["target_paths"] == ["docs/変更履歴.md"]
    assert _condition(conditions, "unchanged")["path"] == "src/version.py"


def test_unicode_path_is_not_matched_as_suffix_of_a_different_filename():
    snapshots = {"変更.md": _snap("# 変更\n")}

    assert derive("大変更.mdは変更しない。", snapshots) == []


def test_ascii_path_is_not_matched_as_suffix_of_a_unicode_filename():
    snapshots = {
        "version.py": _snap("__version__ = '2.7.9'\n"),
        "CHANGELOG.md": _snap("# Changelog\n"),
    }

    conditions = derive(
        "大version.py のパッチ番号を1増やした値をCHANGELOG.mdへ追加。",
        snapshots,
    )

    assert not any(condition["kind"] == "semver_component_increment" for condition in conditions)


def test_exact_case_wins_when_snapshot_paths_differ_only_by_case():
    snapshots = {
        "Foo.py": _snap("UPPER = True\n"),
        "foo.py": _snap("lower = True\n"),
    }

    conditions = derive("foo.pyは変更しない。", snapshots)

    assert [condition["path"] for condition in conditions] == ["foo.py"]


@pytest.mark.parametrize(
    ("label", "expected", "component"),
    [("メジャー", "3.0.0", 0), ("マイナー", "2.8.0", 1), ("パッチ", "2.7.10", 2)],
)
def test_component_increment_resets_lower_components(label, expected, component):
    snapshots = {
        "version.py": _snap("__version__ = '2.7.9'\n"),
        "CHANGELOG.md": _snap("# Changelog\n"),
    }

    conditions = derive(
        f"version.py の{label}番号を1増やした値をCHANGELOG.mdへ追加してください。",
        snapshots,
    )

    increment = _condition(conditions, "semver_component_increment")
    assert increment["component"] == component
    assert increment["delta"] == 1
    assert increment["expected"] == expected


def test_multiple_increment_operations_are_not_mixed_or_guessed():
    snapshots = {
        "version.py": _snap("__version__ = '2.7.9'\n"),
        "CHANGELOG.md": _snap("# Changelog\n"),
    }

    conditions = derive(
        "version.py のパッチ番号を2増やしてからメジャー番号を1増やした値を"
        "CHANGELOG.mdへ追加してください。",
        snapshots,
    )

    assert not any(condition["kind"] == "semver_component_increment" for condition in conditions)


def test_increment_without_an_explicit_target_creates_no_condition():
    snapshots = {"version.py": _snap("__version__ = '2.7.9'\n")}

    conditions = derive("version.py のパッチ番号を1増やしてください。", snapshots)

    assert not any(condition["kind"] == "semver_component_increment" for condition in conditions)


@pytest.mark.parametrize("source", ["version = '2.7.9.1'\n", "address = '192.168.0.1'\n"])
def test_four_part_dotted_values_are_not_treated_as_semver(source):
    snapshots = {
        "version.py": _snap(source),
        "CHANGELOG.md": _snap("# Changelog\n"),
    }

    conditions = derive(
        "version.py のパッチ番号を1増やした値をCHANGELOG.mdへ追加。",
        snapshots,
    )

    assert not any(condition["kind"] == "semver_component_increment" for condition in conditions)


def test_every_explicit_target_must_gain_the_expected_version(tmp_path):
    version = "__version__ = '2.7.9'\n"
    for name in ("A.md", "B.md"):
        (tmp_path / name).write_text("# Changelog\n", encoding="utf-8")
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "A.md").write_text("# Changelog\n\n## 2.7.10\n", encoding="utf-8")
    snapshots = {
        "version.py": _snap(version),
        "A.md": _snap("# Changelog\n"),
        "B.md": _snap("# Changelog\n"),
    }
    conditions = derive(
        "version.py のパッチ番号を1増やした値をA.mdとB.mdへ追加してください。",
        snapshots,
    )

    failures = validate(tmp_path, conditions)

    assert any("B.md" in failure for failure in failures)
    assert not any("A.md" in failure for failure in failures)


def test_reference_path_is_not_misclassified_as_an_edit_target():
    snapshots = {
        "version.py": _snap("__version__ = '2.7.9'\n"),
        "README.md": _snap("# Reference\n"),
        "CHANGELOG.md": _snap("# Changelog\n"),
    }

    conditions = derive(
        "version.py のパッチを1増やした値をREADME.mdを参考に"
        "CHANGELOG.mdへ追加してください。",
        snapshots,
    )

    increment = _condition(conditions, "semver_component_increment")
    assert increment["target_paths"] == ["CHANGELOG.md"]


def test_expected_version_must_be_a_bounded_token(tmp_path):
    version = "__version__ = '2.7.9'\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "release.txt").write_text("next=12.7.100 and x2.7.10y\n", encoding="utf-8")
    snapshots = {"version.py": _snap(version), "release.txt": _snap("")}
    conditions = derive(
        "version.py のパッチ番号を1増やした値をrelease.txtへ追加してください。",
        snapshots,
    )

    assert validate(tmp_path, conditions)


def test_preexisting_expected_version_does_not_count_as_new(tmp_path):
    version = "__version__ = '2.7.9'\n"
    existing = "# Changelog\n\n## 2.7.10\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(existing, encoding="utf-8")
    snapshots = {"version.py": _snap(version), "CHANGELOG.md": _snap(existing)}
    conditions = derive(
        "version.py のパッチ番号を1増やした値をCHANGELOG.mdへ追加してください。",
        snapshots,
    )

    assert validate(tmp_path, conditions)

    (tmp_path / "CHANGELOG.md").write_text(existing + "\n## 2.7.10\n", encoding="utf-8")
    assert validate(tmp_path, conditions) == []


def test_markdown_target_requires_the_expected_version_in_a_heading(tmp_path):
    version = "__version__ = '2.7.9'\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\nThe next release is 2.7.10.\n", encoding="utf-8"
    )
    snapshots = {"version.py": _snap(version), "CHANGELOG.md": _snap("# Changelog\n")}
    conditions = derive(
        "version.py のパッチ番号を1増やした値をCHANGELOG.mdへ追加してください。",
        snapshots,
    )

    assert validate(tmp_path, conditions)

    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## Release 2.7.10\n", encoding="utf-8")
    assert validate(tmp_path, conditions) == []


def test_requested_heading_position_requires_expected_first_subheading(tmp_path):
    version = "__version__ = '2.7.9'\n"
    initial = "# Changelog\n\n## 2.7.9\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        initial + "\n## 2.7.10\n", encoding="utf-8"
    )
    snapshots = {"version.py": _snap(version), "CHANGELOG.md": _snap(initial)}
    conditions = derive(
        "version.py のパッチ番号を1増やした値をCHANGELOG.mdの見出し直後へ追加。",
        snapshots,
    )

    assert validate(tmp_path, conditions)

    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 2.7.10\n\n## 2.7.9\n", encoding="utf-8"
    )
    assert validate(tmp_path, conditions) == []


def test_validation_can_use_unsaved_current_snapshots(tmp_path):
    version = "__version__ = '2.7.9'\n"
    changelog = "# Changelog\n\n## 2.7.9\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    snapshots = {"version.py": _snap(version), "CHANGELOG.md": _snap(changelog)}
    conditions = derive(
        "version.py のパッチ番号を1増やした値をCHANGELOG.mdへ追加し、"
        "version.pyは変更しない。",
        snapshots,
    )
    current = {
        "version.py": {"content": version},
        "CHANGELOG.md": {"content": "# Changelog\n\n## 2.7.10\n\n## 2.7.9\n"},
    }

    assert validate(tmp_path, conditions)
    assert validate(tmp_path, conditions, current_snapshots=current) == []


def test_authoritative_current_snapshots_fail_when_target_is_missing(tmp_path):
    version = "__version__ = '2.7.9'\n"
    initial = "# Changelog\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 2.7.10\n", encoding="utf-8"
    )
    conditions = derive(
        "version.py のパッチ番号を1増やした値をCHANGELOG.mdへ追加。",
        {"version.py": _snap(version), "CHANGELOG.md": _snap(initial)},
    )

    assert validate(
        tmp_path,
        conditions,
        current_snapshots={"version.py": {"content": version}},
    )


def test_unknown_condition_kind_fails_closed(tmp_path):
    assert validate(tmp_path, [{"kind": "future_condition"}]) == [
        "未対応の受け入れ条件です: 'future_condition'"
    ]


def test_ambiguous_numeric_request_does_not_create_condition():
    snapshots = {"data.txt": _snap("2.7.9\n")}
    assert derive("data.txtを更新してください", snapshots) == []
