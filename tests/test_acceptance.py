from pixie_core.acceptance import derive, validate


def _snap(content):
    import hashlib
    return {"content": content, "initial_hash": hashlib.sha256(content.encode()).hexdigest()}


def test_derives_unchanged_and_dynamic_patch_increment(tmp_path):
    (tmp_path / "version.py").write_text("__version__ = '2.7.9'\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 2.8.0\n", encoding="utf-8")
    snapshots = {
        "version.py": _snap("__version__ = '2.7.9'\n"),
        "CHANGELOG.md": _snap("# Changelog\n"),
    }
    conditions = derive(
        "version.py の現在値からパッチ番号を1増やしてCHANGELOG.mdへ追加し、"
        "version.py自体は変更しないでください。",
        snapshots,
    )

    failures = validate(tmp_path, conditions)
    assert any("2.7.10" in failure for failure in failures)
    assert any(condition["kind"] == "unchanged" for condition in conditions)


def test_acceptance_passes_when_relation_and_preservation_hold(tmp_path):
    version = "__version__ = '2.7.9'\n"
    (tmp_path / "version.py").write_text(version, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 2.7.10\n", encoding="utf-8")
    snapshots = {"version.py": _snap(version), "CHANGELOG.md": _snap("# Changelog\n")}
    conditions = derive(
        "version.py のパッチ番号を1増やした値をCHANGELOG.mdへ追加。version.pyは変更しない。",
        snapshots,
    )
    assert validate(tmp_path, conditions) == []


def test_ambiguous_numeric_request_does_not_create_condition():
    snapshots = {"data.txt": _snap("2.7.9\n")}
    assert derive("data.txtを更新してください", snapshots) == []
