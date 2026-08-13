from pixie_core.workset import markdown_index


def test_heading_ranges_stop_at_same_or_parent_level_and_ignore_fences():
    result = markdown_index(
        "# A\ntext\n## A.1\nchild\n```md\n# not heading\n```\n# B\nend\n"
    )
    assert [(h["heading"], h["range"]) for h in result["sections"]] == [
        ("A", [1, 7]), ("A.1", [3, 7]), ("B", [8, 9]),
    ]


def test_local_links_requirements_and_mermaid_are_structured():
    result = markdown_index(
        "SPEC-42 and [local](../a.md#x), [web](https://example.com).\n"
        "```mermaid\nflowchart LR\nA[One] --> B[Two]\n```\n"
    )
    assert result["requirements"] == [{"id": "SPEC-42", "line": 1}]
    assert result["links"] == [{"label": "local", "target": "../a.md#x", "line": 1}]
    assert result["mermaid"] == [{"range": [2, 5], "ids": ["A"]}]
