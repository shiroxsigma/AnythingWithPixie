"""Cross-agent comparison harness の決定論的部分を検証する。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_RUNNER_PATH = Path(__file__).resolve().parent.parent / "evals" / "comparison" / "runner.py"
_SPEC = importlib.util.spec_from_file_location("comparison_runner", _RUNNER_PATH)
comparison_runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison_runner)


def test_create_workspace_and_basic_checks(tmp_path):
    comparison_runner.create_workspace(
        {"files": {"nested/value.txt": "expected\n", "keep.py": "VALUE = 1\n"}},
        tmp_path,
    )

    assert comparison_runner.run_check(
        tmp_path, {"type": "contains", "path": "nested/value.txt", "text": "expected"}
    )[0]
    assert comparison_runner.run_check(
        tmp_path, {"type": "equals", "path": "keep.py", "text": "VALUE = 1\n"}
    )[0]


def test_json_and_ast_checks(tmp_path):
    (tmp_path / "settings.json").write_text('{"timeout_ms": 3000}', encoding="utf-8")
    (tmp_path / "module.py").write_text("def one():\n    pass\n\ndef two():\n    pass\n", encoding="utf-8")

    assert comparison_runner.run_check(
        tmp_path,
        {"type": "json_value", "path": "settings.json", "key": "timeout_ms", "value": 3000},
    )[0]
    assert comparison_runner.run_check(
        tmp_path,
        {"type": "ast_function_count_at_least", "path": "module.py", "count": 2},
    )[0]


def test_not_contains_any_scans_workspace(tmp_path):
    (tmp_path / "a.py").write_text("new_name()\n", encoding="utf-8")
    assert comparison_runner.run_check(
        tmp_path, {"type": "not_contains_any", "texts": ["old_name"]}
    )[0]

    (tmp_path / "b.py").write_text("old_name()\n", encoding="utf-8")
    assert not comparison_runner.run_check(
        tmp_path, {"type": "not_contains_any", "texts": ["old_name"]}
    )[0]


def test_not_contains_any_ignores_agent_metadata(tmp_path):
    metadata = tmp_path / ".pixie_notes" / "state.json"
    metadata.parent.mkdir()
    metadata.write_text('{"note": "old_name"}', encoding="utf-8")
    assert comparison_runner.run_check(
        tmp_path, {"type": "not_contains_any", "texts": ["old_name"]}
    )[0]


def test_provided_context_contains_each_file_once():
    text = comparison_runner.provided_context(
        {"files": {"a.py": "A = 1\n", "b.md": "# B\n"}}
    )
    assert text.count("--- a.py ---") == 1
    assert text.count("--- b.md ---") == 1
    assert "A = 1" in text and "# B" in text


def test_save_results_replaces_checkpoint_with_latest_completed_trials(tmp_path):
    target = tmp_path / "result.json"
    meta = {"agent": "awp", "track": "provided"}
    first = {"passed": True, "task_id": "c01"}
    comparison_runner.save_results(target, meta, [first])

    second = {"passed": False, "task_id": "c02"}
    payload = comparison_runner.save_results(target, meta, [first, second])
    saved = comparison_runner.json.loads(target.read_text(encoding="utf-8"))

    assert payload["summary"] == {"passed": 1, "total": 2}
    assert saved == payload
    assert not target.with_suffix(".json.tmp").exists()


def test_capture_source_state_identifies_tracked_and_untracked_changes(tmp_path):
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    comparison_runner.initialize_git_repo(tmp_path)

    clean = comparison_runner.capture_source_state(tmp_path)
    assert clean["git_available"] is True
    assert len(clean["git_head"]) >= 40
    assert clean["dirty"] is False
    assert clean["tracked_dirty"] is False
    assert clean["untracked"]["count"] == 0
    assert len(clean["tracked_diff_sha256"]) == 64

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    modified = comparison_runner.capture_source_state(tmp_path)
    assert modified["dirty"] is True
    assert modified["tracked_dirty"] is True
    assert modified["tracked_diff_sha256"] != clean["tracked_diff_sha256"]
    assert modified["state_id"] != clean["state_id"]

    untracked = tmp_path / "new file.py"
    untracked.write_text("FIRST\n", encoding="utf-8")
    added = comparison_runner.capture_source_state(tmp_path)
    assert added["untracked"]["count"] == 1
    assert added["untracked"]["paths"] == ["new file.py"]
    assert added["untracked"]["hash_complete"] is True
    assert added["state_id"] != modified["state_id"]

    untracked.write_text("SECOND\n", encoding="utf-8")
    changed_untracked = comparison_runner.capture_source_state(tmp_path)
    assert changed_untracked["untracked"]["content_sha256"] != added["untracked"]["content_sha256"]
    assert changed_untracked["state_id"] != added["state_id"]


def test_command_and_base_url_provenance_remove_secrets():
    command = comparison_runner.sanitize_command(
        [
            "python",
            "runner.py",
            "--api-key",
            "super-secret",
            "--base-url=https://user:password@example.test:8443/v1?token=secret#fragment",
        ]
    )

    rendered = " ".join(command)
    assert "super-secret" not in rendered
    assert "password" not in rendered
    assert "token=" not in rendered
    assert "fragment" not in rendered
    assert "--api-key <redacted>" in rendered
    assert "--base-url=https://example.test:8443/v1" in rendered


def test_run_metadata_reports_source_consistency_across_repetitions():
    args = SimpleNamespace(
        agent="awp",
        track="provided",
        model="model.gguf",
        base_url="http://user:pass@localhost:8080/v1?api_key=secret",
        task="c02",
        repeat=3,
    )
    source = {
        "git_head": "a" * 40,
        "dirty": True,
        "tracked_diff_sha256": "b" * 64,
        "state_id": "start-state",
    }
    meta = comparison_runner.build_run_metadata(
        args,
        [{"id": "c02"}],
        started_at="2026-08-24T00:00:00+00:00",
        source_state=source,
        command=["runner.py", "--api-key=secret"],
    )

    assert meta["agent"] == "awp"
    assert meta["track"] == "provided"
    assert meta["task"] == "c02"
    assert meta["task_ids"] == ["c02"]
    assert meta["repeat"] == 3
    assert meta["base_url"] == "http://localhost:8080/v1"
    assert meta["base_url_identity"] == "http://localhost:8080/v1"
    assert meta["command"][-1] == "--api-key=<redacted>"
    assert meta["started_at"] == "2026-08-24T00:00:00+00:00"
    assert meta["source_at_start"] == source
    assert meta["source_consistency"]["same_source_for_all_repetitions"] is True

    comparison_runner.observe_source_state(meta, {"state_id": "start-state"})
    assert meta["source_consistency"]["same_source_for_all_repetitions"] is True
    comparison_runner.observe_source_state(meta, {"state_id": "changed-state"})
    assert meta["source_consistency"]["same_source_for_all_repetitions"] is False
    assert meta["source_consistency"]["observed_state_ids"] == [
        "start-state",
        "changed-state",
    ]
