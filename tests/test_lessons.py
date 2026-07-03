"""lessons.py（教訓ストア）と reflection トリガー条件のテスト。

前半: LessonStore の純粋テスト（add/recall/重複統合(Jaccard)/GC/永続化/injection text）。
後半: run_graph からの reflection 呼出条件（失敗信号あり→呼ぶ・なし→呼ばない・
壊れたJSON応答でもクラッシュしない・generalizable=false は保存しない）を
scripted mock LLM で検証する（test_run_graph_blackbox.py の _MockLLM パターンを踏襲）。
LLM 呼び出しは一切発生しない（すべてモック）。
"""

import copy
import json
import types

import engine
from engine import run_graph
from lessons import LessonStore
from state import AgentState

# =====================================================
# LessonStore — 純粋テスト（LLM不要）
# =====================================================


def test_add_new_lesson_persists_and_reloads(tmp_path):
    path = str(tmp_path / "lessons.json")
    store = LessonStore(file_path=path)
    rec = store.add("read_file前に必ずパスの存在を確認する", ["read_file", "パス"], source="reflection")

    assert rec is not None
    assert rec["lesson"].startswith("read_file")
    assert rec["source"] == "reflection"
    assert rec["hit_count"] == 0
    assert rec["last_used_at"] is None
    assert "id" in rec and "created_at" in rec

    reloaded = LessonStore(file_path=path)
    assert len(reloaded.lessons) == 1
    assert reloaded.lessons[0]["lesson"] == rec["lesson"]
    assert reloaded.lessons[0]["trigger_keywords"] == ["read_file", "パス"]


def test_add_empty_lesson_returns_none(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    assert store.add("   ", ["x"]) is None
    assert store.lessons == []


def test_add_duplicate_lesson_merges_via_jaccard(tmp_path):
    """単語 Jaccard 類似度 >= 0.6 の教訓は新規追加されず、既存の hit_count が +1 される。"""
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    store.add("search_and_replace実行前に必ずread_fileで最新内容を確認すること", ["search_and_replace"])
    rec2 = store.add("search_and_replace実行前には必ずread_fileで最新内容を確認すること", ["read_file"])

    assert len(store.lessons) == 1
    assert store.lessons[0]["hit_count"] == 1
    assert rec2["hit_count"] == 1
    # trigger_keywords がマージされている
    assert "search_and_replace" in store.lessons[0]["trigger_keywords"]
    assert "read_file" in store.lessons[0]["trigger_keywords"]


def test_add_dissimilar_lessons_both_kept(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    store.add("read_file前にパスの存在を確認する", ["read_file"])
    store.add("run_commandはタイムアウトを必ず設定する", ["run_command"])
    assert len(store.lessons) == 2


def test_recall_scores_and_updates_hitcount(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    store.add("write_file実行後は必ずfast_gateの結果を確認する", ["write_file", "fast_gate"])
    store.add("run_commandは対話プログラムの検知に注意する", ["run_command", "対話"])

    results = store.recall("write_fileでファイルを書き込みたい", max_results=3)

    assert results
    assert results[0]["lesson"].startswith("write_file")
    assert results[0]["hit_count"] == 1
    assert results[0]["last_used_at"] is not None


def test_recall_no_match_returns_empty(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    store.add("read_file前にパスの存在を確認する", ["read_file"])
    assert store.recall("まったく無関係なクエリZZZ999", max_results=3) == []


def test_recall_empty_store_returns_empty(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    assert store.recall("何か", max_results=3) == []


def test_gc_enforces_max_items_prefers_low_hitcount(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"), max_items=5)
    for i in range(5):
        store.add(f"教訓その{i}番はユニークな内容Unique{i}です", [f"kw{i}"])
    # 既存の1件だけ何度もrecallしてhit_countを稼ぐ（生存させたい）
    for _ in range(3):
        store.recall("教訓その0番 kw0", max_results=1)
    # 新規教訓を追加してGCを発火させる（6件目 > max_items=5）
    store.add("教訓その5番はユニークな内容Unique5です", ["kw5"])

    assert len(store.lessons) <= 5
    lesson_texts = [r["lesson"] for r in store.lessons]
    assert any("その0番" in t for t in lesson_texts)  # hit_count 高いものは生存


def test_to_injection_text_format(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    store.add("write_file実行後は必ずfast_gateの結果を確認する", ["write_file"])
    text = store.to_injection_text("write_fileでファイルを書きたい", max_chars=600)

    assert text.startswith("【過去の教訓】")
    assert "write_file" in text


def test_to_injection_text_empty_when_no_match(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    store.add("read_file前にパスの存在を確認する", ["read_file"])
    assert store.to_injection_text("無関係なクエリABC999", max_chars=600) == ""


def test_to_injection_text_respects_max_chars(tmp_path):
    store = LessonStore(file_path=str(tmp_path / "lessons.json"))
    for i in range(10):
        store.add(f"教訓{i}: " + "あ" * 100, ["共通キーワード"])
    text = store.to_injection_text("共通キーワード", max_chars=200, max_results=10)
    assert len(text) <= 200


# =====================================================
# reflection — scripted mock LLM でトリガー条件を検証
# =====================================================
# test_run_graph_blackbox.py の _MockLLM パターンを踏襲（LM Studio 互換: 常に generator）。


class _MockLLM:
    """create_chat_completion の戻り値をスクリプト化したモック。

    scripts: list of (content, tool_calls_or_None)。
    reflection 呼出（stream=False, response_format 指定）も node_plan 呼出と
    同じ delta 形式の1チャンクで応答すれば良い（_collect_subquery_response が両対応）。
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.captured_kwargs = []  # 呼出ごとの kwargs（response_format等の検証用）
        self.n_ctx = 32768

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream,
                               tools=None, tool_choice=None, response_format=None, **kw):
        self.captured_kwargs.append({
            "messages": copy.deepcopy(messages),
            "stream": stream,
            "response_format": response_format,
        })
        content, tool_calls = self.scripts.pop(0)
        finish_reason = "tool_calls" if tool_calls else "stop"

        def _gen():
            yield {
                "choices": [{
                    "delta": {"content": content, "tool_calls": tool_calls},
                    "finish_reason": finish_reason,
                }]
            }
        return _gen()

    def estimate_token_count(self, text):
        return len(text) // 3


def _make_ctx(llm, **overrides):
    defaults = dict(
        llm=llm,
        code_mode=False,
        force_deep=False,
        supports_tool_role=False,
        debug_mode=False,
        phase="EXECUTING",
        review_mode=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _output_buf():
    buf = []

    def _fn(text="", end="", flush=True, **kw):
        buf.append(text)
    return _fn, buf


def _fresh_state(user_text="現在の状況について詳しく教えてください"):
    state = AgentState()
    state.chat_history.add("user", user_text)
    return state


def _sys_builder(context, state_board, **kw):
    return "You are a helpful test assistant."


def _run(ctx, state, out_fn=None):
    return run_graph(
        ctx, state,
        output_fn=out_fn,
        system_msg_builder=_sys_builder,
    )


class _SpyLessonStore:
    """get_lesson_store() を差し替えて add() 呼出を記録するスパイ。"""

    def __init__(self):
        self.added = []

    def to_injection_text(self, *a, **kw):
        return ""

    def add(self, lesson, trigger_keywords=None, source=""):
        self.added.append((lesson, trigger_keywords, source))
        return {"lesson": lesson}


def test_reflection_not_called_when_no_failure_signals(monkeypatch):
    """失敗信号が0件のターンでは reflection LLM 呼出自体が発生しない（追加コストゼロ）。"""
    spy = _SpyLessonStore()
    monkeypatch.setattr(engine, "get_lesson_store", lambda: spy)

    llm = _MockLLM([
        ("これは十分に詳しい最終回答です。結論として問題ありません。以上で完了です。", None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state()
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert state.failure_signals == []
    assert llm.scripts == []  # node_plan の1回しか呼ばれていない（scripts を使い切っていない=2回目呼出なし相当を別途確認）
    assert spy.added == []


def test_reflection_called_and_saves_generalizable_lesson(monkeypatch):
    """失敗信号が1件以上あれば final_answer 確定後に reflection LLM が1回呼ばれ、
    generalizable=true の教訓が LessonStore.add() で保存される。"""
    spy = _SpyLessonStore()
    monkeypatch.setattr(engine, "get_lesson_store", lambda: spy)

    reflection_json = json.dumps({
        "lesson": "search_and_replace前にread_fileで最新内容を必ず確認する",
        "trigger_keywords": ["search_and_replace", "read_file"],
        "generalizable": True,
    }, ensure_ascii=False)

    llm = _MockLLM([
        ("これは十分に詳しい最終回答です。結論として問題ありません。以上で完了です。", None),
        (reflection_json, None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state()
    # 失敗信号を注入（run_graph 内で本来自動収集されるものを直接シミュレート）
    state.failure_signals.append("broken_tool_call: ツール呼び出しのフォーマットエラーを検知")
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert llm.scripts == []  # 2回とも消費された（node_plan + reflection）
    assert len(spy.added) == 1
    lesson, keywords, source = spy.added[0]
    assert "search_and_replace" in lesson
    assert "search_and_replace" in keywords
    assert source == "reflection"

    # reflection 呼出は response_format(json_schema) 付き・非ストリーミングだったこと
    reflection_call = llm.captured_kwargs[-1]
    assert reflection_call["stream"] is False
    assert reflection_call["response_format"]["type"] == "json_schema"


def test_reflection_not_generalizable_not_saved(monkeypatch):
    spy = _SpyLessonStore()
    monkeypatch.setattr(engine, "get_lesson_store", lambda: spy)

    reflection_json = json.dumps({
        "lesson": "このタスク固有すぎる話",
        "trigger_keywords": ["固有"],
        "generalizable": False,
    }, ensure_ascii=False)

    llm = _MockLLM([
        ("これは十分に詳しい最終回答です。結論として問題ありません。以上で完了です。", None),
        (reflection_json, None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state()
    state.failure_signals.append("short_answer_guardrail: テスト用の失敗信号")
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert spy.added == []


def test_reflection_broken_json_does_not_crash(monkeypatch):
    """reflection の LLM 応答が壊れたJSONでもクラッシュせず、final_answer は正常に返る。"""
    spy = _SpyLessonStore()
    monkeypatch.setattr(engine, "get_lesson_store", lambda: spy)

    llm = _MockLLM([
        ("これは十分に詳しい最終回答です。結論として問題ありません。以上で完了です。", None),
        ("{ this is not valid json at all !!", None),  # 壊れたJSON応答
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state()
    state.failure_signals.append("loop_guardrail: テスト用の失敗信号")
    out_fn, _ = _output_buf()

    result = _run(ctx, state, out_fn=out_fn)

    assert result  # final_answer は壊れていない
    assert spy.added == []  # 保存はされない


def test_reflection_llm_exception_does_not_crash(monkeypatch):
    """reflection 呼出自体が例外を投げても run_graph の結果に影響しない。"""
    spy = _SpyLessonStore()
    monkeypatch.setattr(engine, "get_lesson_store", lambda: spy)

    class _ExplodingLLM(_MockLLM):
        def create_chat_completion(self, messages, *, max_tokens, temperature, stream,
                                   tools=None, tool_choice=None, response_format=None, **kw):
            if response_format is not None:
                raise RuntimeError("simulated LLM failure during reflection")
            return super().create_chat_completion(
                messages, max_tokens=max_tokens, temperature=temperature, stream=stream,
                tools=tools, tool_choice=tool_choice, response_format=response_format, **kw)

    llm = _ExplodingLLM([
        ("これは十分に詳しい最終回答です。結論として問題ありません。以上で完了です。", None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state()
    state.failure_signals.append("fast_gate: テスト用の失敗信号")
    out_fn, _ = _output_buf()

    result = _run(ctx, state, out_fn=out_fn)

    assert result
    assert spy.added == []


def test_lessons_disabled_flag_skips_reflection(monkeypatch):
    """LESSONS_ENABLED=False の場合、失敗信号があっても reflection 自体を呼ばない。"""
    monkeypatch.setattr(engine, "LESSONS_ENABLED", False)
    spy = _SpyLessonStore()
    monkeypatch.setattr(engine, "get_lesson_store", lambda: spy)

    llm = _MockLLM([
        ("これは十分に詳しい最終回答です。結論として問題ありません。以上で完了です。", None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state()
    state.failure_signals.append("manually injected signal")
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert llm.scripts == []  # reflection 呼出なし（1回しか消費されない）
    assert spy.added == []


def test_broken_tool_call_guardrail_records_failure_signal(monkeypatch):
    """実際のガードレール発火経路（ツール呼び出しフォーマットエラー）が
    state.failure_signals に記録されることを確認する。"""
    spy = _SpyLessonStore()
    monkeypatch.setattr(engine, "get_lesson_store", lambda: spy)

    reflection_json = json.dumps({
        "lesson": "ツール呼び出しのフォーマットエラー時は再試行させる",
        "trigger_keywords": ["tool_call"],
        "generalizable": True,
    }, ensure_ascii=False)

    # tool_call_count > 0 の状態で short_answer_guardrail に阻まれないよう、
    # 完全性スコアが十分高い（>=50）最終回答テキストを使う。
    final_answer_text = "調査の結果、原因が判明しました。\n- 結論: 対応は不要です\n- 理由: 問題なし\n以上がまとめです。"
    llm = _MockLLM([
        ("<tool_call>壊れた呼び出し", None),  # 壊れたツール呼び出しっぽいテキスト → ガードレール発火
        (final_answer_text, None),
        (reflection_json, None),
    ])
    ctx = _make_ctx(llm)
    state = _fresh_state()
    out_fn, _ = _output_buf()

    _run(ctx, state, out_fn=out_fn)

    assert any(s.startswith("broken_tool_call:") for s in state.failure_signals)
    assert len(spy.added) == 1
