"""tools.py の純関数テスト: JITスコアリング、スキーマ生成、検索ヒント。"""

from tools import TOOL_REGISTRY, _build_search_hint, registry_to_openai_tools, score_tools


def test_score_tools_returns_valid_subset():
    result = score_tools("ファイルを読み込んで", top_n=5)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(name in TOOL_REGISTRY for name in result)
    # top_n はスコア上位の追加数; ALWAYS_RECOMMEND が常に加わるため
    # len(result) は top_n + len(ALWAYS_RECOMMEND) 程度になる


def test_score_tools_tool_name_match_boosted():
    # ツール名が直接入力に含まれると高スコア (+5.0)
    result = score_tools("read_file を使って", top_n=5)
    assert "read_file" in result


def test_score_tools_surfaces_delegate_for_explain():
    # 説明・理解系キーワードで delegate_research が推奨セットに浮上
    assert "delegate_research" in score_tools("プロジェクトの構成を説明して")


def test_score_tools_surfaces_delegate_for_detail():
    assert "delegate_research" in score_tools("もっと詳細を教えて")


def test_score_tools_surfaces_delegate_for_overall():
    assert "delegate_research" in score_tools("全体像と構造を把握したい")


def test_registry_to_openai_tools_shape():
    tools = registry_to_openai_tools(["read_file"])
    assert len(tools) == 1
    t = tools[0]
    assert t["type"] == "function"
    fn = t["function"]
    assert fn["name"] == "read_file"
    assert "description" in fn
    assert "parameters" in fn


def test_registry_to_openai_tools_all_when_none():
    tools = registry_to_openai_tools(None)
    assert len(tools) == len(TOOL_REGISTRY)


def test_build_search_hint_returns_string():
    search_lines = ["def foo():"]
    content_lines = ["import os", "def foo():", "    pass", "def bar():"]
    hint = _build_search_hint(search_lines, content_lines)
    assert isinstance(hint, str)


def test_build_search_hint_empty_search_returns_empty():
    assert _build_search_hint([], ["line"]) == ""


def test_new_code_tools_registered():
    """code_index 系の3ツールが TOOL_REGISTRY に登録されている。"""
    assert "map_codebase" in TOOL_REGISTRY
    assert "detect_dead_code" in TOOL_REGISTRY
    assert "read_symbol" in TOOL_REGISTRY
