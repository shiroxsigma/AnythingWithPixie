"""lfm_tooluse.py のテスト（pure・モデル不要）。

LFM2.5 専用 tool use のパーサ/注入器を検証。AST安全性（eval/exec 不使用）含む。
"""

from lfm_tooluse import (
    LFM_ARTIFACT_TAGS,
    LFM_TOOL_END,
    LFM_TOOL_START,
    inject_lfm_tools,
    parse_lfm_tool_calls,
    parse_pythonic_call,
)

# =====================================================
# parse_lfm_tool_calls
# =====================================================

def test_parse_json_single():
    content = f'{LFM_TOOL_START}{{"name": "read_file", "arguments": {{"path": "x.py"}}}}{LFM_TOOL_END}'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["id"].startswith("lfm_read_file_")
    assert cleaned is None  # 全除去


def test_parse_pythonic_single():
    content = f'{LFM_TOOL_START}[read_file(path="x.py")]{LFM_TOOL_END}'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert '"path"' in calls[0]["function"]["arguments"]


def test_parse_pythonic_multiple_in_block():
    content = f'{LFM_TOOL_START}[get_weather(loc="NYC"), get_time(tz="EST")]{LFM_TOOL_END}'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is not None and len(calls) == 2
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[1]["function"]["name"] == "get_time"


def test_parse_pythonic_typed_args():
    content = f'{LFM_TOOL_START}[search(query="hello", limit=10, flag=True)]{LFM_TOOL_END}'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is not None and len(calls) == 1
    args = calls[0]["function"]["arguments"]
    assert "10" in args
    assert "true" in args.lower()


def test_parse_mixed_text_and_call():
    content = f'ファイルを確認します。\n{LFM_TOOL_START}{{"name": "list_directory", "arguments": {{}}}}{LFM_TOOL_END}\n完了'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "list_directory"
    assert "ファイルを確認します" in cleaned
    assert "完了" in cleaned


def test_parse_truncated_no_end_token_does_not_crash():
    # end トークン欠落（max_tokens 到達等）。rescue がクラッシュせず None を返すことを保証
    content = f'{LFM_TOOL_START}{{"name": "run_command", "arguments": {{"command": "ls"'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is None  # 不完全 → 解析不能
    assert isinstance(cleaned, str)


def test_parse_malformed_returns_none():
    content = f'{LFM_TOOL_START}this is not valid json or pythonic{{]{LFM_TOOL_END}'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is None


def test_parse_no_tool_call():
    content = "通常のテキスト応答です。ツールは使いません。"
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is None
    assert cleaned == content


def test_parse_empty_content():
    cleaned, calls = parse_lfm_tool_calls("")
    assert calls is None


def test_tool_call_id_unique_in_block():
    content = (
        f'{LFM_TOOL_START}{{"name": "a", "arguments": {{}}}}{LFM_TOOL_END}'
        f'{LFM_TOOL_START}{{"name": "a", "arguments": {{}}}}{LFM_TOOL_END}'
    )
    cleaned, calls = parse_lfm_tool_calls(content)
    assert len(calls) == 2
    ids = [c["id"] for c in calls]
    assert len(set(ids)) == 2  # 同名でも id 一意


# =====================================================
# parse_pythonic_call（AST安全性）
# =====================================================

def test_pythonic_keyword_args():
    result = parse_pythonic_call('[read_file(path="x.py", start=1)]')
    assert result == [("read_file", {"path": "x.py", "start": 1})]


def test_pythonic_positional_args():
    result = parse_pythonic_call('[read_file("x.py")]')
    assert result is not None
    assert result[0][0] == "read_file"
    assert result[0][1] == {"arg0": "x.py"}


def test_pythonic_reject_attribute():
    # obj.method() は拒否（Attribute — RCE経路）
    assert parse_pythonic_call("[obj.method(x=1)]") is None


def test_pythonic_reject_call_as_arg():
    # f(g()) は拒否（Call-as-arg）
    assert parse_pythonic_call("[f(x=g())]") is None


def test_pythonic_reject_bare_name_value():
    # f(x=y) の y は裸 Name → 拒否（未定義参照）
    assert parse_pythonic_call("[f(x=y)]") is None


def test_pythonic_syntax_error_safe():
    # 閉じ括弧なし → SyntaxError → None（例外伝播しない）
    assert parse_pythonic_call("[func(x=1)") is None


def test_pythonic_list_value():
    result = parse_pythonic_call('[f(items=[1, 2, 3])]')
    assert result == [("f", {"items": [1, 2, 3]})]


def test_pythonic_dict_value():
    result = parse_pythonic_call('[f(opts={"a": 1})]')
    assert result == [("f", {"opts": {"a": 1}})]


# =====================================================
# inject_lfm_tools
# =====================================================

def test_inject_schema_conversion():
    openai_tools = [{
        "type": "function",
        "function": {
            "name": "get_cwd",
            "description": "現在ディレクトリ",
            "parameters": {"type": "object"},
        },
    }]
    result = inject_lfm_tools("BASE", openai_tools)
    assert "BASE" in result
    assert "List of tools:" in result
    assert '"name": "get_cwd"' in result
    assert "parameters" in result
    assert LFM_TOOL_START in result  # 出力指示にトークン含む


def test_inject_empty_tools():
    result = inject_lfm_tools("BASE", [])
    assert "List of tools: []" in result


def test_constants():
    assert LFM_TOOL_START == "<|tool_call_start|>"
    assert LFM_TOOL_END == "<|tool_call_end|>"
    assert LFM_TOOL_START in LFM_ARTIFACT_TAGS
    assert LFM_TOOL_END in LFM_ARTIFACT_TAGS


# =====================================================
# ```json コードブロック（<|tool_call_start|> なしのフォールバック）
# =====================================================

def test_parse_json_code_block_no_tokens():
    # <|tool_call_start|> なし、```json コードブロック内の配列（実運用ログの実例）
    content = '```json\n[\n  {"name": "list_directory", "arguments": {"path": "src"}},\n  {"name": "gather_project_info", "arguments": {"path": "."}}\n]\n```'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is not None and len(calls) == 2
    assert calls[0]["function"]["name"] == "list_directory"
    assert calls[1]["function"]["name"] == "gather_project_info"


def test_parse_json_code_block_single():
    content = '```\n{"name": "get_cwd", "arguments": {}}\n```'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "get_cwd"


def test_parse_code_block_non_tool_json_ignored():
    # name を持たない JSON 配列は tool call とみなさない（説明用 JSON 例の誤検出回避）
    content = '```json\n[{"id": 1, "value": "example"}, {"id": 2}]\n```'
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is None


# =====================================================
# 裸 Pythonic レスキュー（特殊トークンも```も無い content 中の裸呼び出し行）
# フェーズ3 eval パターンA対策: known_tools 指定時のみ有効。誤検知対策が最重要。
# =====================================================

def test_bare_pythonic_rescued_when_known_tool():
    # 実運用ログの実例: 行動宣言テキスト直後に裸の Pythonic 呼び出し行が漏れるケース
    content = (
        "I'll read the `config_sample.py` file to locate the value of "
        "`DEFAULT_TIMEOUT`.\n\n[read_file(path='config_sample.py')]"
    )
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"read_file", "list_directory"})
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert "config_sample.py" in calls[0]["function"]["arguments"]
    # 呼び出し行は除去され、宣言文だけが残る
    assert "read_file(" not in (cleaned or "")
    assert "I'll read" in (cleaned or "")


def test_bare_pythonic_no_brackets_rescued():
    content = "まずファイルを確認します。\nread_file(path=\"x.py\")"
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"read_file"})
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"


def test_bare_pythonic_ignored_when_unknown_tool_name():
    # パース自体は成功するが、関数名が known_tools に無いためツール呼び出しとみなさない
    # （ハルシネーションした架空関数名や単なるコード例の誤検出回避）
    content = "以下のように呼び出します。\n[totally_made_up_func(x=1)]"
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"read_file", "list_directory"})
    assert calls is None
    assert cleaned == content


def test_bare_pythonic_ignored_when_known_tools_none():
    # known_tools 未指定（None）なら従来動作のまま（裸レスキュー無効）
    content = "[read_file(path='x.py')]"
    cleaned, calls = parse_lfm_tool_calls(content)
    assert calls is None
    assert cleaned == content


def test_bare_pythonic_inline_code_example_not_rescued():
    # 文章中に埋め込まれたコード例（行全体が呼び出し式だけで構成されない）は誤検出しない
    content = (
        "例えば Python では print(\"hello\") のようにすれば標準出力に表示できます。"
        "これはツール呼び出しではありません。"
    )
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"read_file", "print"})
    assert calls is None
    assert cleaned == content


def test_bare_pythonic_multiple_calls_in_list_rescued():
    content = "[read_file(path='a.py'), list_directory(path='.')]"
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"read_file", "list_directory"})
    assert calls is not None and len(calls) == 2
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[1]["function"]["name"] == "list_directory"


def test_bare_name_json_line_rescued():
    # eval 05 実測の漏れ形式: `write_file: {"path": ..., "content": ...}` 行 + 完了主張テキスト
    content = (
        'write_file: {"path": "math_utils.py", "content": "def is_even(n):\\n    return n % 2 == 0"}\n'
        "I have created the file math_utils.py containing the requested is_even function."
    )
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"write_file", "read_file"})
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"
    import json as _json
    args = _json.loads(calls[0]["function"]["arguments"])
    assert args["path"] == "math_utils.py"
    assert "is_even" in args["content"]
    assert "write_file:" not in (cleaned or "")


def test_bare_name_json_line_unknown_tool_ignored():
    content = 'made_up_tool: {"x": 1}'
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"write_file"})
    assert calls is None
    assert cleaned == content


def test_bare_name_json_line_non_dict_ignored():
    # JSON が dict でない（配列や壊れた JSON）は不採用
    content = 'write_file: {"path": broken json'
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"write_file"})
    assert calls is None


def test_bare_name_json_prose_with_colon_not_rescued():
    # 「補足: {...}」のような日本語見出し行や、known_tools 外の説明行は無視される
    content = '注意: {"これは": "説明用のJSON例です"}'
    cleaned, calls = parse_lfm_tool_calls(content, known_tools={"read_file", "write_file"})
    assert calls is None
    assert cleaned == content
