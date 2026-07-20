"""pixie_core API 1.6（会話履歴の外科的編集）の安全網。

組み込み側（CodeWithPixie）が「回答が不要だった往復だけを LLM 文脈から消す」「会話を
要約して次セッションへ引き継ぐ」を行うための追加。壊れると次ターンが丸ごと失敗する
（OpenAI 互換 API は assistant(tool_calls) と tool 応答の対が崩れた履歴を 400 で弾く）ので、
対の繕いを重点的に固める。
"""
import pixie_core
from pixie_core import _api

_SERVER = {"base_url": "http://localhost:1/v1", "model": "test-model"}


def _engine(tmp_path, **kw):
    return pixie_core.create_engine(_SERVER, str(tmp_path), **kw)


def _seed(eng, messages):
    """ChatHistory を直接組む（tool_calls 付きの履歴は load_history では作れないため）。"""
    eng.state.chat_history.messages = list(messages)
    return eng.state.chat_history.messages


def _turn(user, answer):
    """1往復ぶんの素朴な履歴（ツール呼び出しなし）。"""
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def _tool_turn(user, tool_name, result, answer):
    """ツールを1回呼んだ往復（assistant(tool_calls) → tool → assistant）。"""
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "function": {"name": tool_name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": result},
        {"role": "assistant", "content": answer},
    ]


# --- history_size / history_tail（ターン境界の記録） ---

def test_history_size_and_tail(tmp_path):
    eng = _engine(tmp_path)
    msgs = _seed(eng, _turn("q1", "a1") + _turn("q2", "a2"))
    assert eng.history_size() == 4
    tail = eng.history_tail(2)
    assert [m["content"] for m in tail] == ["q2", "a2"]
    # 返るのは履歴が持っている dict そのもの（＝後でその往復を指すハンドルになる）
    assert tail[0] is msgs[2]


def test_history_tail_clamps_and_is_a_new_list(tmp_path):
    eng = _engine(tmp_path)
    _seed(eng, _turn("q", "a"))
    assert eng.history_tail(-5) == eng.history_tail(0)
    assert eng.history_tail(99) == []
    eng.history_tail(0).clear()  # 返り値は複製リスト。捨てても履歴は減らない
    assert eng.history_size() == 2


# --- history_drop（往復の除去） ---

def test_history_drop_removes_only_the_given_turn(tmp_path):
    eng = _engine(tmp_path)
    _seed(eng, _turn("q1", "a1") + _turn("q2", "a2") + _turn("q3", "a3"))
    handles = eng.history_tail(2)[:2]  # 2番目の往復
    assert eng.history_drop(handles) == 2
    assert [m["content"] for m in eng.state.chat_history.messages] == ["q1", "a1", "q3", "a3"]


def test_history_drop_removes_tool_messages_of_that_turn(tmp_path):
    eng = _engine(tmp_path)
    _seed(eng, _turn("q1", "a1") + _tool_turn("q2", "read_file", "…", "a2"))
    assert eng.history_drop(eng.history_tail(2)) == 4
    assert [m["content"] for m in eng.state.chat_history.messages] == ["q1", "a1"]


def test_history_drop_ignores_already_trimmed_messages(tmp_path):
    """自動トリム（check_and_trim_context）で先に落ちたハンドルは黙って無視される。"""
    eng = _engine(tmp_path)
    stale = {"role": "user", "content": "既に落ちた"}
    _seed(eng, _turn("q", "a"))
    assert eng.history_drop([stale]) == 0
    assert eng.history_size() == 2


def test_history_drop_tolerates_non_dict_handles(tmp_path):
    eng = _engine(tmp_path)
    _seed(eng, _turn("q", "a"))
    assert eng.history_drop([None, "x", 1]) == 0
    assert eng.history_drop([]) == 0
    assert eng.history_size() == 2


# --- 対の繕い（_repair_tool_pairs） ---

def test_drop_repairs_orphaned_tool_result(tmp_path):
    """呼び出し元の assistant だけを消しても、孤児の tool 応答は残らない。"""
    eng = _engine(tmp_path)
    msgs = _seed(eng, _tool_turn("q", "read_file", "中身", "a"))
    eng.history_drop([msgs[1]])  # assistant(tool_calls) だけを消す
    roles = [m["role"] for m in eng.state.chat_history.messages]
    assert "tool" not in roles
    assert roles == ["user", "assistant"]


def test_drop_repairs_dangling_tool_calls(tmp_path):
    """tool 応答だけを消したら、応答の来ない tool_calls も残さない。"""
    eng = _engine(tmp_path)
    msgs = _seed(eng, _tool_turn("q", "read_file", "中身", "a"))
    msgs[1]["content"] = "読みます"
    eng.history_drop([msgs[2]])  # tool 応答だけを消す
    kept = eng.state.chat_history.messages
    assert [m["role"] for m in kept] == ["user", "assistant", "assistant"]
    assert "tool_calls" not in kept[1]      # 本文は残し、呼び出しだけ外す
    assert kept[1]["content"] == "読みます"


def test_drop_removes_contentless_dangling_call(tmp_path):
    eng = _engine(tmp_path)
    msgs = _seed(eng, _tool_turn("q", "read_file", "中身", "a"))
    eng.history_drop([msgs[2]])  # 本文が空の assistant(tool_calls) は丸ごと落ちる
    assert [m["content"] for m in eng.state.chat_history.messages] == ["q", "a"]


def test_repair_keeps_multi_call_tool_runs(tmp_path):
    """1つの assistant に複数 tool 応答が続く形（並列ツール実行）を壊さない。"""
    eng = _engine(tmp_path)
    run = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        {"role": "assistant", "content": "a1"},
    ]
    msgs = _seed(eng, _turn("q0", "a0") + run)
    eng.history_drop(msgs[:2])  # 前の往復だけ消す
    assert [m["role"] for m in eng.state.chat_history.messages] == [
        "user", "assistant", "tool", "tool", "assistant"]


# --- history_replace（要約による引き継ぎ） ---

def test_history_replace_swaps_whole_history(tmp_path):
    eng = _engine(tmp_path)
    _seed(eng, _tool_turn("q", "read_file", "中身", "a"))
    eng.history_replace([{"role": "user", "content": "これまでの要約"},
                         {"role": "assistant", "content": "把握しました"}])
    assert [(m["role"], m["content"]) for m in eng.state.chat_history.messages] == [
        ("user", "これまでの要約"), ("assistant", "把握しました")]


def test_history_replace_filters_like_load_history(tmp_path):
    eng = _engine(tmp_path)
    _seed(eng, _turn("q", "a"))
    eng.history_replace([{"role": "tool", "content": "無視"},
                         {"role": "user", "content": ""},
                         {"role": "user", "content": "要約"}])
    assert [m["content"] for m in eng.state.chat_history.messages] == ["要約"]


def test_history_replace_with_empty_clears(tmp_path):
    eng = _engine(tmp_path)
    _seed(eng, _turn("q", "a"))
    eng.history_replace([])
    assert eng.state.chat_history.messages == []
