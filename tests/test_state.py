"""state.py のテスト: AgentStateBoard, ChatHistory, build_system_prompt。

AgentStateBoard は file_path を tmp_path に注入して I/O を隔離する。
"""

from state import AgentStateBoard, ChatHistory, build_system_prompt

# =====================================================
# AgentStateBoard
# =====================================================

def test_state_board_set_goal_persists_and_reloads(tmp_path):
    path = str(tmp_path / "state.json")
    board = AgentStateBoard(file_path=path)
    board.set_goal("目標X")
    reloaded = AgentStateBoard(file_path=path)
    assert reloaded.goal == "目標X"


def test_state_board_gc_completed_tasks(tmp_path):
    board = AgentStateBoard(file_path=str(tmp_path / "s.json"))
    for i in range(10):
        board.complete_task(f"task {i}")
    assert len(board.completed_tasks) <= AgentStateBoard.MAX_COMPLETED_TASKS


def test_state_board_query_finds_knowledge(tmp_path):
    board = AgentStateBoard(file_path=str(tmp_path / "s.json"))
    board.update(found_knowledge="key1=value1")
    result = board.query("key1")
    assert "key1" in result
    assert "value1" in result


def test_state_board_empty_detection(tmp_path):
    board = AgentStateBoard(file_path=str(tmp_path / "s.json"))
    assert board.is_empty() is True
    board.set_goal("anything")
    assert board.is_empty() is False


def test_state_board_to_injection_text(tmp_path):
    board = AgentStateBoard(file_path=str(tmp_path / "s.json"))
    board.set_goal("ゴール")
    text = board.to_injection_text(max_chars=800)
    assert "ゴール" in text


# =====================================================
# ChatHistory
# =====================================================

def test_chat_history_trim():
    ch = ChatHistory(max_messages=3)
    for i in range(5):
        ch.add("user", f"msg{i}")
    popped = ch.trim()
    assert len(popped) == 2
    assert len(ch.messages) == 3


def test_chat_history_add_and_get():
    ch = ChatHistory()
    ch.add("user", "hello")
    msgs = ch.get_messages({"role": "system", "content": "sys"})
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "hello"


# =====================================================
# build_system_prompt
# =====================================================
#
# prefix cache（KVキャッシュ再利用）安定化のため、build_system_prompt は
# base_prompt をそのまま返すだけの静的な関数になった。state_board / ホワイトボード
# 等の動的コンテキストは、この関数の責務から外れ、engine.py の
# _build_dynamic_suffix() が直近のユーザーメッセージ末尾に注入する方式に変更された
# （system メッセージがセッション内でほぼ不変になることが目的）。

def test_build_system_prompt_base_only():
    result = build_system_prompt("BASE_RULES")
    assert "BASE_RULES" in result


def test_build_system_prompt_returns_base_prompt_verbatim():
    """動的コンテキストの注入は行わず、base_prompt をそのまま返す。"""
    result = build_system_prompt("ABSOLUTE_RULE")
    assert result == "ABSOLUTE_RULE"
