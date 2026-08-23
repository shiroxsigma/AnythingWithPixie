from pathlib import Path

from tools import apply_search_replace_changeset


def _blocks(*items):
    return "".join(
        f"*** Update File: {path}\n<<<<<<< SEARCH\n{search}=======\n{replace}>>>>>>> REPLACE\n"
        for path, search, replace in items
    )


def test_batch_tool_applies_multiple_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 1\n", encoding="utf-8")
    result = apply_search_replace_changeset(_blocks(
        ("a.py", "A = 1\n", "A = 2\n"),
        ("b.py", "B = 1\n", "B = 2\n"),
    ))
    assert result.startswith("Success:")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 2\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "B = 2\n"


def test_batch_tool_is_all_or_nothing_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 1\n", encoding="utf-8")
    result = apply_search_replace_changeset(_blocks(
        ("a.py", "A = 1\n", "A = 2\n"),
        ("b.py", "missing\n", "B = 2\n"),
    ))
    assert result.startswith("Error:")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "B = 1\n"


def test_batch_tool_rejects_duplicate_search_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("same\nsame\n", encoding="utf-8")
    result = apply_search_replace_changeset(_blocks(("a.txt", "same\n", "new\n")))
    assert result.startswith("Error:")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "same\nsame\n"


def test_batch_tool_accepts_small_model_flat_file_header(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    changes = "--- a.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n>>>>>>> REPLACE\n"
    assert apply_search_replace_changeset(changes).startswith("Success:")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 2\n"


def test_batch_tool_accepts_fence_blank_line_and_marker_whitespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    changes = (
        "```diff\n--- a.py\n\n<<<<<<< SEARCH   \nA = 1\n=======   \n"
        "A = 2\n>>>>>>> REPLACE   \n```\n"
    )
    assert apply_search_replace_changeset(changes).startswith("Success:")


def test_batch_tool_rejects_unparsed_second_block_instead_of_partial_apply(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    changes = _blocks(("a.py", "A = 1\n", "A = 2\n")) + "BROKEN SECOND FILE\n"
    result = apply_search_replace_changeset(changes)
    assert result.startswith("Error:")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 1\n"
