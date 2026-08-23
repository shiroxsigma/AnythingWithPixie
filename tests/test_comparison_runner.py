"""Cross-agent comparison harness の決定論的部分を検証する。"""

import importlib.util
from pathlib import Path


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
