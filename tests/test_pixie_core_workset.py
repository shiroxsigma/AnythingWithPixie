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


def test_workset_adds_matching_test_and_document_after_explicit_items(tmp_path):
    source = tmp_path / "src" / "auth.py"
    test = tmp_path / "tests" / "test_auth.py"
    doc = tmp_path / "docs" / "auth-guide.md"
    for path, content in ((source, "def login(): pass\n"),
                          (test, "def test_login(): pass\n"),
                          (doc, "# auth\n")):
        path.parent.mkdir(exist_ok=True)
        path.write_text(content, encoding="utf-8")
    engine = _engine(tmp_path)

    workset = engine.build_workset({"task": "auth のログインを修正", "pinned_paths": ["src/auth.py"]})

    assert [(item["path"], item["role"]) for item in workset["items"]] == [
        ("src/auth.py", "pinned"),
        ("tests/test_auth.py", "test"),
        ("docs/auth-guide.md", "spec"),
    ]
    assert workset["stats"]["auto_added"] == 2


def test_workset_can_disable_related_candidates(tmp_path):
    (tmp_path / "auth.py").write_text("x = 1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_auth.py").write_text("x = 1\n", encoding="utf-8")

    workset = _engine(tmp_path).build_workset({
        "pinned_paths": ["auth.py"], "include_related": False,
    })

    assert [item["path"] for item in workset["items"]] == ["auth.py"]


def test_workset_uses_ast_callers_imports_tests_and_markdown_index(tmp_path):
    files = {
        "src/auth.py": "from src.crypto import verify\n\ndef login(token):\n    return verify(token)\n",
        "src/crypto.py": "def verify(token):\n    return bool(token)\n",
        "src/routes.py": "from src.auth import login\n\ndef post_login(token):\n    return login(token)\n",
        "tests/test_auth.py": "from src.auth import login\n\ndef test_login():\n    assert login('x')\n",
        "docs/auth.md": (
            "# Login contract\nREQ-AUTH-01 must hold.\n[implementation](../src/auth.py)\n\n"
            "```mermaid\nA[Client] --> B[Login]\n```\n"
        ),
    }
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode())

    result = _engine(tmp_path).build_workset({
        "task": "login 認証の変更", "pinned_paths": ["src/auth.py"],
    })
    by_path = {item["path"]: item for item in result["items"]}

    assert by_path["src/routes.py"]["role"] == "caller"
    assert by_path["src/routes.py"]["reason"] == "calls_target"
    assert by_path["src/crypto.py"]["role"] == "dependency"
    assert by_path["tests/test_auth.py"]["role"] in {"caller", "test"}
    doc = by_path["docs/auth.md"]
    assert doc["role"] == "spec"
    assert doc["sections"][0]["heading"] == "Login contract"
    assert doc["requirements"][0]["id"] == "REQ-AUTH-01"
    assert doc["links"][0]["target"] == "../src/auth.py"
    assert doc["mermaid"][0]["ids"] == ["A"]
    assert result["stats"]["symbols"] >= 4
    assert result["stats"]["document_files"] == 1


def test_markdown_index_uses_unsaved_buffer(tmp_path):
    path = tmp_path / "spec.md"
    path.write_bytes(b"# Old\n")
    engine = _engine(tmp_path)
    engine.set_workspace_snapshot({"buffers": [{
        "path": "spec.md", "content": "# New contract\nNFR-PERF-2\n",
    }]})

    result = engine.build_workset({"task": "New contract", "pinned_paths": ["spec.md"]})

    item = result["items"][0]
    assert item["buffer"] is True
    assert item["sections"][0]["heading"] == "New contract"
    assert item["requirements"][0]["id"] == "NFR-PERF-2"


@pytest.mark.parametrize("payload", [[], {"pinned_paths": "a.py"}, {"task": 1}])
def test_workset_rejects_invalid_shapes(tmp_path, payload):
    with pytest.raises(TypeError):
        _engine(tmp_path).build_workset(payload)
