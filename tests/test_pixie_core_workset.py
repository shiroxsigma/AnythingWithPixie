from pathlib import Path

import pytest

from pixie_core._api import Engine


def _engine(workspace: Path) -> Engine:
    return Engine(context=object(), state=object(), workspace=str(workspace))


def test_workset_uses_buffer_without_embedding_content(tmp_path):
    path = tmp_path / "src" / "main.py"
    path.parent.mkdir()
    path.write_text("DISK = True\n", encoding="utf-8")
    engine = _engine(tmp_path)
    engine.set_workspace_snapshot({"buffers": [
        {"path": "src/main.py", "content": "UNSAVED = True\nsecond = 2\n"},
    ]})

    workset = engine.build_workset({
        "task": "このファイルを直す",
        "selected_path": "src/main.py",
        "pinned_paths": ["src/main.py"],
    })

    assert workset["task"] == "このファイルを直す"
    assert len(workset["items"]) == 1
    item = workset["items"][0]
    assert item["path"] == "src/main.py"
    assert item["role"] == "target"
    assert item["buffer"] is True
    assert item["lines"] == 2
    assert item["chars"] == len("UNSAVED = True\nsecond = 2\n")
    assert "content" not in item


def test_workset_reports_missing_outside_and_limit(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    engine = _engine(tmp_path)

    workset = engine.build_workset({
        "pinned_paths": ["a.md", "b.md", "missing.md", str(tmp_path.parent / "outside.md")],
        "max_items": 1,
    })

    assert [item["path"] for item in workset["items"]] == ["a.md"]
    assert {item["reason"] for item in workset["omitted"]} == {"max_items", "outside_workspace"}


@pytest.mark.parametrize("payload", [[], {"pinned_paths": "a.py"}, {"task": 1}])
def test_workset_rejects_invalid_shapes(tmp_path, payload):
    with pytest.raises(TypeError):
        _engine(tmp_path).build_workset(payload)
