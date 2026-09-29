"""Regression checks for deterministic real-model eval scoring."""

import sys
from pathlib import Path

import pytest

_EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
if str(_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALS_DIR))

from checkers import RunResult, subprocess_stdout_equals


@pytest.mark.parametrize("exit_code, expected_pass", [(0, True), (1, False), (2, False)])
def test_matching_stdout_requires_successful_process(tmp_path, exit_code, expected_pass):
    run = RunResult(task_id="subprocess_exit", workspace_dir=tmp_path)

    passed, detail = subprocess_stdout_equals(
        run,
        ["{python}", "-c", f"print('expected'); raise SystemExit({exit_code})"],
        "expected",
    )

    assert passed is expected_pass
    if exit_code:
        assert f"returncode={exit_code}" in detail


def test_successful_process_with_wrong_stdout_fails(tmp_path):
    run = RunResult(task_id="subprocess_output", workspace_dir=tmp_path)

    passed, _ = subprocess_stdout_equals(
        run, ["{python}", "-c", "print('unexpected')"], "expected"
    )

    assert passed is False
