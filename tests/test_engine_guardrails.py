"""engine.py のガードレール/コンテキスト純関数のテスト。

最も複雑で未テストだった領域（反復検知・類似度・回答完全性・思考深度・
ツール呼び出しパース等）の挙動を固定するリグレッションテスト。
"""

from engine import (
    _accumulate_tool_calls,
    _answer_completeness_score,
    _build_thinking_notes_block,
    _detect_content_similarity,
    _detect_repetitive_content,
    _has_unclosed_thinking,
    _is_simple_question,
    _looks_like_action_promise,
    _merge_continuation,
    _parse_native_tool_calls,
    _resolve_thinking_mode,
    _safe_parse_args,
    _truncate_thought,
)
from state import AgentState

# =====================================================
# 反復出力検知
# =====================================================

def test_repetitive_identical_lines():
    # len(content) >= 100 のガードを超える十分な長さの反復行
    text = "これは同じ行を繰り返し出力するテスト用の文章です。\n" * 4
    assert _detect_repetitive_content(text) is True


def test_repetitive_varied_text():
    text = ("今日は晴れです。\n明日は雨でしょう。\n"
            "明後日は曇りの予報です。\nその後は分かりません。")
    assert _detect_repetitive_content(text) is False


def test_repetitive_short_text_returns_false():
    assert _detect_repetitive_content("短すぎる") is False


# =====================================================
# コンテンツ類似度 (Jaccard)
# =====================================================

def test_similarity_identical():
    text = "testing similarity function words here"
    assert _detect_content_similarity(text, text) is True


def test_similarity_disjoint():
    a = "alpha bravo charlie delta echo"
    b = "foxtrot golf hotel india juliet"
    assert _detect_content_similarity(a, b, threshold=0.65) is False


def test_similarity_empty():
    assert _detect_content_similarity("", "something") is False


# =====================================================
# 回答完全性スコア
# =====================================================

def test_completeness_substantive_answer():
    content = "## 結論\n原因はXです。なぜなら根拠があります。\n対応案：修正を行う。"
    assert _answer_completeness_score(content, tool_call_count=3) >= 50


def test_completeness_one_liner():
    assert _answer_completeness_score("はい。", tool_call_count=0) < 50


def test_completeness_empty():
    assert _answer_completeness_score("", 0) == 0


def test_completeness_advice_answer_not_flagged():
    # コミットメッセージ提案のような「相談への構造化回答」は完成した最終回答。
    # 調査報告キーワード(結論/対応案/まとめ)を含まなくてもスコア>=50になること。
    content = (
        "コミットメッセージの候補を以下のパターンで提案します。\n\n"
        "### 1. 修正の場合\n"
        "- `fix(engine): ツール実行時のエラーハンドリングを改善`\n"
        "- 具体的な修正内容を簡潔に\n\n"
        "### 2. 機能追加の場合\n"
        "- `feat(engine): 新しい推論モードのロジックを追加`\n\n"
        "### 3. リファクタリングの場合\n"
        "- `refactor(engine): コンテキスト管理ロジックを整理`\n\n"
        "調査が完了しているため、これで最終回答とします。"
    )
    assert _answer_completeness_score(content, tool_call_count=1) >= 50


def test_completeness_structured_explanation_signal():
    # 見出し2以上+箇条書き3以上+150字以上の構造化説明文はシグナル8で加算される。
    # アドバイス系キーワード抜きでも構造だけで十分なスコアになることを担保。
    content = (
        "### 手順その1\n- 最初の段階の説明を丁寧に書きます。\n"
        "### 手順その2\n- 次の段階の説明も丁寧に書きます。\n"
        "### 手順その3\n- 最後の段階の説明を書き記しておきます。\n"
        "これで一通りの流れを網羅しました。"
    )
    assert _answer_completeness_score(content, tool_call_count=0) >= 50


# =====================================================
# 思考strip未完検知 (D)
# =====================================================

def test_unclosed_thinking_marker_detected():
    # StreamFilter.flush が未閉じ<think>時に前置するマーカー
    content = "\n(※思考プロセスが閉じられなかったため内容を表示します)\n生の思考内容..."
    assert _has_unclosed_thinking(content) is True


def test_unclosed_thinking_unbalanced_tag():
    # <think> が開かれたまま切り取られた（閉じタグなし）
    assert _has_unclosed_thinking("<think>推論中です。") is True


def test_unclosed_thinking_balanced_tag():
    # 正しく閉じられた思考ブロックは未完ではない
    assert _has_unclosed_thinking("<think>推論</think>これが回答です。") is False


def test_unclosed_thinking_plain_content():
    assert _has_unclosed_thinking("通常の最終回答です。") is False


def test_unclosed_thinking_empty():
    assert _has_unclosed_thinking("") is False


# =====================================================
# 行動予告検知
# =====================================================

def test_action_promise_next_action():
    assert _looks_like_action_promise("次にファイルを確認します。") is True


def test_action_promise_final_answer_marker():
    assert _looks_like_action_promise("## 結論\n対応案を提示します。") is False


def test_action_promise_empty():
    assert _looks_like_action_promise("") is False


# =====================================================
# 思考深度モード判定
# =====================================================

def test_is_simple_question_cwd():
    assert _is_simple_question("今のディレクトリは？") is True


def test_is_simple_question_complex():
    # "見せて" は simple marker に含まれるため、それを含まない複雑な質問で検証
    assert _is_simple_question("システム全体の設計を比較検討して") is False


def test_resolve_force_deep_overrides():
    s = AgentState()
    assert _resolve_thinking_mode(s, "hello", force_deep=True) == "deep"
    assert s._was_deep is True


def test_resolve_simple_question_is_shallow():
    s = AgentState()
    assert _resolve_thinking_mode(s, "今のディレクトリは？") == "shallow"


def test_resolve_explicit_deep_keyword():
    s = AgentState()
    assert _resolve_thinking_mode(s, "この問題をじっくり考えて") == "deep"


def test_resolve_default_is_shallow():
    s = AgentState()
    assert _resolve_thinking_mode(s, "hello world") == "shallow"


def test_resolve_hysteresis_stays_deep():
    s = AgentState()
    s._was_deep = True
    assert _resolve_thinking_mode(s, "今のディレクトリは？") == "deep"


# =====================================================
# ネイティブツール呼び出しパース (GGUF)
# =====================================================

def test_parse_native_tool_call_block():
    content = '<tool_call\n{"name": "read_file", "arguments": {"path": "x.py"}}\n</tool_call>'
    cleaned, calls = _parse_native_tool_calls(content)
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert cleaned is None  # 全除去 → 空 → None


def test_parse_native_no_tool_call():
    cleaned, calls = _parse_native_tool_calls("通常のテキストです")
    assert calls is None
    assert cleaned == "通常のテキストです"


def test_parse_native_malformed_json():
    content = '<tool_call\n{bad json}\n</tool_call>'
    cleaned, calls = _parse_native_tool_calls(content)
    assert calls is None  # 解析失敗 → ツールなし


# =====================================================
# ストリームチャンク蓄積
# =====================================================

def test_accumulate_content_only():
    chunks = [
        {"choices": [{"delta": {"content": "hello "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
    ]
    content, calls = _accumulate_tool_calls(chunks)
    assert content == "hello world"
    assert calls is None


def test_accumulate_tool_calls_from_deltas():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "read", "arguments": '{"path":'}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ' "x.py"}'}}
        ]}}]},
    ]
    content, calls = _accumulate_tool_calls(chunks)
    assert calls is not None and len(calls) == 1
    assert calls[0]["id"] == "c1"
    assert calls[0]["function"]["name"] == "read"
    assert calls[0]["function"]["arguments"] == '{"path": "x.py"}'


# =====================================================
# 引数の安全なパース
# =====================================================

def test_safe_parse_args_valid_json():
    assert _safe_parse_args({"arguments": '{"a": 1}'}) == {"a": 1}


def test_safe_parse_args_empty_string():
    assert _safe_parse_args({"arguments": ""}) == {}


def test_safe_parse_args_missing_key():
    assert _safe_parse_args({}) == {}


def test_safe_parse_args_invalid_json():
    assert _safe_parse_args({"arguments": "{invalid"}) == {}


def test_safe_parse_args_non_dict_json():
    assert _safe_parse_args({"arguments": "[1, 2, 3]"}) == {}


def test_safe_parse_args_whitespace_only():
    assert _safe_parse_args({"arguments": "   "}) == {}


# =====================================================
# 思考メモの切詰め・構築
# =====================================================

def test_truncate_thought_short_passthrough():
    assert _truncate_thought("短いテキスト") == "短いテキスト"


def test_truncate_thought_empty():
    assert _truncate_thought("") == ""


def test_truncate_thought_long_is_capped():
    text = "あ" * 1000
    result = _truncate_thought(text, max_chars=100)
    assert len(result) <= 100


def test_thinking_notes_empty_returns_empty_string():
    assert _build_thinking_notes_block([]) == ""


def test_thinking_notes_contains_items():
    result = _build_thinking_notes_block(["メモ1", "メモ2"])
    assert "【前回の思考メモ" in result
    assert "メモ1" in result
    assert "メモ2" in result


def test_thinking_notes_caps_total_length():
    notes = ["x" * 300] * 20
    result = _build_thinking_notes_block(notes, max_chars=500)
    assert len(result) <= 600


# =====================================================
# 継続出力の重複結合 (スマート継続)
# =====================================================

def test_merge_continuation_empty():
    assert _merge_continuation("", "abc") == "abc"
    assert _merge_continuation("abc", "") == "abc"


def test_merge_continuation_no_overlap():
    assert _merge_continuation("abc", "def") == "abcdef"


def test_merge_continuation_tail_overlap():
    # 末尾と先頭が重複(>10文字) → 重複除去
    acc = "Lorem ipsum dolor sit amet consectetur"
    new = "amet consectetur adipiscing elit"
    assert _merge_continuation(acc, new) == "Lorem ipsum dolor sit amet consectetur adipiscing elit"


def test_merge_continuation_prefix_repeat():
    # 先頭から再生成（accumulated全体がnewの前置き、>10文字重複）→ 重複除去
    acc = "the quick brown fox"
    new = "the quick brown fox jumps over"
    assert _merge_continuation(acc, new) == "the quick brown fox jumps over"
