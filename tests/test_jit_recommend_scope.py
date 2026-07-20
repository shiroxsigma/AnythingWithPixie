"""JIT 推奨ヒントは「提示しているツール」だけを勧めること。

制限プロファイル（埋め込み側の fixed_tool_set。例: NoteWithPixie の read 専用モード）で
使えないツールを勧めると、モデルがそれを呼ぼうとして空応答になり、engine の強制指示
ループに落ちる（実測: Note モードに search_and_replace を勧めていた）。
"""
import engine


def _suffix(available, jit_input="ファイルの記述を直して置換したい"):
    state = engine.AgentState()
    return engine._build_dynamic_suffix(
        state,
        available_tools=set(available),
        jit_input=jit_input,
        thinking_mode="shallow",
        usage_ratio=0.5,
    )


def _recommended_line(text):
    for line in text.splitlines():
        if line.startswith("【推奨ツール】"):
            return line
    return ""


def test_recommendations_are_limited_to_available_tools():
    readonly = {"list_workspace", "read_note", "grep_workspace", "describe_flows"}
    line = _recommended_line(_suffix(readonly))
    assert "search_and_replace" not in line
    assert "write_file" not in line
    for name in line.replace("【推奨ツール】このリクエストには次のツールが関連度高い可能性があります: ", "").split(", "):
        if name.strip():
            assert name.strip() in readonly


def test_no_recommendation_line_when_nothing_available():
    assert _recommended_line(_suffix(set())) == ""


def test_full_tool_set_still_gets_recommendations():
    """通常（制限なし）のセッションでは従来どおりヒントが出ること。"""
    from registry import TOOL_REGISTRY
    line = _recommended_line(_suffix(set(TOOL_REGISTRY)))
    assert line.startswith("【推奨ツール】")
