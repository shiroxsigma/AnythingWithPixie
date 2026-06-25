"""
AnythingPixie — エージェントエンジンモジュール

ReAct (Plan->Action->Observe) ループエンジン。
ストリーミング、並列ツール実行、コンテキスト管理、ループ検知を統合管理する。

依存: config.py, state.py, tools.py, llm_client.py
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import registry
from config import (
    CONTEXT_BUFFER,
    CONTEXT_CHECKPOINT_THRESHOLD,
    DEEP_THINK_BUDGET_SEC,
    DEFAULT_TRIM_THRESHOLD,
    DESTRUCTIVE_TOOLS,
    MAX_PARALLEL_TOOLS,
    MAX_TOKENS,
    MIN_CONTEXT_TOKENS,
    READONLY_TOOLS,
    TEMPERATURE_LOOP_THRESHOLD,
    TEMPERATURE_MAIN,
    WHITEBOARD_DETAIL_SEPARATOR,
    WHITEBOARD_PATH,
    WHITEBOARD_SYSTEM_PROMPT,
)
from engine_helpers import (
    FILE_EDIT_TOOLS as _FILE_EDIT_TOOLS,
)
from engine_helpers import (
    accumulate_tool_calls as _accumulate_tool_calls,
)
from engine_helpers import (
    default_output_fn as _default_output_fn,
)
from engine_helpers import (
    detect_repetitive_content as _detect_repetitive_content,
)
from engine_helpers import (
    estimate_tokens,
)
from engine_helpers import (
    is_simple_question as _is_simple_question,
)
from engine_helpers import (
    parse_native_tool_calls as _parse_native_tool_calls,
)
from engine_helpers import (
    safe_parse_args as _safe_parse_args,
)
from engine_helpers import (
    strip_all_thinking as _strip_all_thinking,
)
from llm_client import SuppressStderr
from paths import get_data_path
from state import AgentState, build_system_prompt
from subagent import (
    _backup_if_file_edit,
    _collect_subquery_response,
    _execute_analyze_file,
    _execute_delegate_research,
    _is_design_proposal,
    _run_design_review,
    _run_edit_review,
    _run_ruff_check,
    run_verify_fix_loop,
    run_vision_subquery,
)
from tools import (
    TOOL_REGISTRY,
    check_loop_detected,
    execute_builtin_tool,
    generate_behavior_prompt,
    registry_to_openai_tools,
    resize_and_encode_image,
    score_tools,
)


def _merge_continuation(accumulated: str, new_chunk: str) -> str:
    """length継続で途切れた出力を累積に結合する（末尾-先頭の重複を除去）。

    エージェントが「続き」を出せば重複なしで追記。末尾を繰り返してから
    続けた場合は重複を除去。継続プロンプトで「最初からやらない」を強制する
    ことで、全く別の再生成（重複検出困難）を予防する。
    """
    if not accumulated:
        return new_chunk or ""
    if not new_chunk:
        return accumulated
    # accumulated の末尾と new_chunk の先頭の最長一致を探す
    max_check = min(len(accumulated), len(new_chunk), 800)
    overlap = 0
    for n in range(max_check, 10, -1):
        if accumulated[-n:] == new_chunk[:n]:
            overlap = n
            break
    return accumulated + new_chunk[overlap:]


# =====================================================
# ツール結果の圧縮（コンテキスト保護）
# =====================================================

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def _strip_ansi(text: str) -> str:
    """テキストからANSIエスケープシーケンスを除去する。"""
    return _ANSI_RE.sub('', text)


def _compress_tool_result(tool_name: str, tool_args: dict, result: str) -> str:
    """ツール実行結果を履歴に載せる前に処理する（能動圧縮）。

    現ターンは内容をそのまま返す（モデルが結果を利用するため）。ただし read_file の
    大きな結果からアウトライン（def/class）を安価に抽出し、ステートボードの
    file_summaries に登録しておく。これにより後の mask_old_observations で生テキストを
    切り捨てても、ファイル構造の知識はステートボード（毎ターン注入）に残る。
    """
    if tool_name == "read_file":
        try:
            _register_read_outline(tool_args, result)
        except Exception:
            pass  # 圧縮失敗で履歴登録を止めない
    return result


def _register_read_outline(tool_args: dict, result: str) -> None:
    """read_file 結果からアウトライン（def/class 行）を抽出し、ステートボードに登録。

    LLM 呼出なし・決定的。コード構造（def/class）が検出できないファイル（YAML 等）は
    ノイズ回避のため登録しない。`registry._state_board` は関数内参照なので
    常に最新のインスタンスを参照する（set_state_board 後の再束縛に追従）。
    """
    sb = registry._state_board
    if not sb:
        return
    path = tool_args.get("path", "")
    if not path:
        return
    outline = []
    for m in re.finditer(r'(?m)^\s*(?:async\s+)?(?:def|class)\s+\w+', result):
        outline.append(m.group(0).strip())
        if len(outline) >= 15:
            break
    if not outline:
        return  # コード構造なし → 注册しない
    summary_bits = []
    m_hdr = re.search(r'^\[[^\]]+\][^\n]*', result)
    if m_hdr:
        summary_bits.append(m_hdr.group(0).strip()[:120])
    summary_bits.append("構造: " + " | ".join(outline))
    sb.add_file_summary(path, "\n".join(summary_bits))


#: ツール実行結果をユーザー端末にも直接表示するツール
DISPLAY_TOOLS = frozenset({"diff_files"})


def _format_tool_args(args: dict, max_value_len: int = 40) -> str:
    """ツール引数を読みやすい key=value 形式にフォーマットする。

    長い値や複数行の値は「最初の1行... (全体N文字)」に圧縮する。
    """
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if '\n' in v_str or len(v_str) > max_value_len:
            first_line = v_str.split('\n')[0][:max_value_len]
            parts.append(f"{k}={first_line}... ({len(v_str)}文字)")
        else:
            parts.append(f"{k}={v_str}")
    return ", ".join(parts)


# =====================================================
# 無限思考ループ検知
# =====================================================

def _detect_content_similarity(content1: str, content2: str, threshold: float = 0.65) -> bool:
    """2つのテキスト間の類似度を簡易的に判定する（Jaccard類似度）。

    Args:
        content1: 比較対象テキスト1
        content2: 比較対象テキスト2
        threshold: 類似度閾値（0.0-1.0）

    Returns:
        類似度が閾値以上の場合 True
    """
    if not content1 or not content2:
        return False

    def tokenize(text):
        return set(re.findall(r'[\w]{4,}', text.lower()))

    tokens1 = tokenize(content1)
    tokens2 = tokenize(content2)

    if not tokens1 or not tokens2:
        return False

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    similarity = len(intersection) / len(union) if union else 0
    return similarity >= threshold


def _has_balanced_think(text: str) -> bool:
    """<think> ブロックが閉じているか（途中切断でなければ True）。

    思考タイムアウト等で <think> が閉じられなかった場合は False を返す。
    未閉じタグはチャットテンプレートのレンダリングを壊す恐れがあるため、
    その内容は履歴に残さない（Feature A の安全装置）。
    """
    if "<think" not in text:
        return True
    return bool(re.search(r'<think[^>]*>.*?</think', text, flags=re.DOTALL))


# StreamFilter.flush() が未閉じ<think>を検出した際に content に前置するマーカー。
# この文字列が残っている場合、content には生の思考内容が混入しており、
# 完全性スコア判定には使えない（engine.py の StreamFilter.flush と同期すること）。
_UNCLOSED_THINK_MARKER = "(※思考プロセスが閉じられなかったため内容を表示します)"


def _has_unclosed_thinking(content: str) -> bool:
    """思考ブロックが未閉じのまま出力が終了したか（不完全コンテンツの兆候）を判定する。

    StreamFilter.flush が未閉じ <think> を検出した際に付与されるマーカー文字列、
    または <think> タグの不均衡（開いたまま切り取られた）のいずれかで True。
    この状態の content は生の思考プロセスが混入してしており、
    _answer_completeness_score で評価しても不正確になるため、
    短い回答ガードレールではこの状態をスキップする（→ 継続/length判定に委ねる）。
    """
    if not content:
        return False
    if _UNCLOSED_THINK_MARKER in content:
        return True
    if "<think" in content and not _has_balanced_think(content):
        return True
    return False


def _add_assistant_with_think(state, content: str, tool_calls=None) -> None:
    """直前の思考を引き継ぐため、最後の assistant メッセージだけ <think> を残して追加する。

    Feature A の中核。不変量: chat_history 中で <think> を持つ assistant メッセージは
    「直近1件のみ」。追加前に「既存の全 assistant メッセージの <think> を剥がし」てから、
    新しいメッセージを <think> 付きで追加する（未閉じなら剥がす）。

    モデル（Qwen3/DeepSeek）が自分の直前の推論を自然な会話フォーマットで参照できるようにし、
    ターンをまたいだ/ツール反復をまたいだ推論の積み上げを可能にする。
    コストは ~1ブロック（90sタイムアウトで2-3kトークン上限）に抑えられる。
    """
    msgs = state.chat_history.messages
    # 既存 assistant の <think> を全て剥がす（直近1件のみ残す不変量の維持）
    for m in msgs:
        if m.get("role") == "assistant":
            mc = m.get("content")
            if isinstance(mc, str) and "<think" in mc:
                m["content"] = _strip_all_thinking(mc)

    safe = content or ""
    # 新メッセージの <think> は閉じている場合のみ残す
    if "<think" in safe and not _has_balanced_think(safe):
        safe = _strip_all_thinking(safe)
    state.chat_history.add("assistant", safe, tool_calls=tool_calls)


# フェーズ定数
_EXPLORING = "EXPLORING"
_SYNTHESIZING = "SYNTHESIZING"


def _detect_phase(state: AgentState) -> str:
    """実行済みツールの履歴から現在フェーズを推定する。

    EXPLORING: 情報収集中（即断即実）
    SYNTHESIZING: 十分な情報が揃い、深い分析が必要
    """
    if len(state.executed_actions) < 3:
        return _EXPLORING

    recent = state.executed_actions[-5:]
    read_tools = frozenset({
        "read_file", "grep_search", "get_code_outline", "analyze_file",
        "list_directory", "view_tree", "research_code_paths",
    })
    read_count = sum(
        1 for a in recent
        if any(a.startswith(f"{t}:") for t in read_tools)
    )

    if read_count >= 3:
        return _SYNTHESIZING
    return _EXPLORING


def _resolve_thinking_mode(state: AgentState, user_text: str, force_deep: bool = False) -> str:
    """思考深度モードを判定する（shallow / deep）。段階的思考深化。

    判定優先順序（Plan agent 検証で見直した安全順）:
      1. force_deep（/deep コマンド等の明示的指定）
      2. ヒステリシス: 一度deepに入ったらshallowに戻さない（_detect_phase のジッタ対策）
      3. ユーザー明示（「じっくり/深く/設計して/考えて」）
      4. 単純質問（_is_simple_question）→ shallow確定
      5. フェーズ/回数（tool_call_count >= 3 or SYNTHESIZING）
      6. 難易度語（「なぜ/比較/ベスト/リスク/トレードオフ/設計」）
      7. デフォルト → shallow

    一度 deep と判定されたら state._was_deep = True を立てる。
    """
    if force_deep:
        state._was_deep = True
        return "deep"
    if getattr(state, "_was_deep", False):
        return "deep"

    if user_text:
        # 3. ユーザー明示
        if any(k in user_text for k in ("じっくり", "深く", "設計して", "考えて")):
            state._was_deep = True
            return "deep"
        # 4. 単純質問 → shallow確定
        if _is_simple_question(user_text):
            return "shallow"
        # 6. 難易度語
        if any(k in user_text for k in ("なぜ", "比較", "ベスト", "リスク", "トレードオフ", "設計")):
            state._was_deep = True
            return "deep"

    # 5. フェーズ/回数
    if state.tool_call_count >= 3 or _detect_phase(state) == _SYNTHESIZING:
        state._was_deep = True
        return "deep"

    return "shallow"


def _truncate_thought(text: str, max_chars: int = 400) -> str:
    """<think>末尾を文境界で丸めて抽出する（ノイズ削減・コンテキスト保護）。

    Qwen3 等の <think> は冒頭に "Wait, let me reconsider..." 等の迷いを含むため、
    結論に近い末尾側を残す。
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # 末尾 max_chars 文字を取り、直近の文境界で先頭を整える
    tail = text[-max_chars:]
    for sep in ("\n", "。", "．", "！", "？", ". ", "! ", "? "):
        idx = tail.find(sep)
        if 0 <= idx < max_chars // 2:
            tail = tail[idx + len(sep):]
            break
    return tail.strip()


def _build_thinking_notes_block(thinking_notes: list[str], max_chars: int = 1500) -> str:
    """前回の思考メモをシステムプロンプト注入用に組み立てる（コンテキスト保護付き）。

    新しい方から積み上げ、合計 max_chars を超えたら古い方を捨てる。
    """
    if not thinking_notes:
        return ""
    parts = []
    total = 0
    for note in reversed(thinking_notes):
        note = note.strip()
        if not note:
            continue
        block = f"- {note}"
        if total + len(block) > max_chars:
            break
        parts.insert(0, block)
        total += len(block)
    if not parts:
        return ""
    return "【前回の思考メモ（推論を引き継げ）】\n" + "\n".join(parts)


def _looks_like_action_promise(content: str) -> bool:
    """ツール呼び出しなしの自然文が「次に〜します」型の行動予告かを判定する。

    最終回答（「結論」「まとめ」「対応案」等を含む）は除外する。
    """
    if not content:
        return False
    text = _strip_all_thinking(content).strip()

    # 最終回答らしい語がある場合は除外
    final_markers = [
        "結論", "まとめ", "最終回答", "調査結果", "原因", "対応案",
        "改善案", "ベストプラクティス", "総括", "概要",
    ]
    if any(m in text for m in final_markers):
        return False

    patterns = [
        r"次[には].*(確認|解析|調査|読み込|実行|見て|調べ)",
        r"これから.*(確認|解析|調査|読み込|実行|見て|調べ)",
        r"引き続き.*(確認|解析|調査|読み込|実行|見て|調べ)",
        r"まずは.*(確認|解析|調査|読み込|実行|見て|調べ)",
        r"(確認|解析|調査|読み込み|実行)していきます",
        r"(確認|解析|調査|読み込み|実行)します",
    ]
    return any(re.search(p, text) for p in patterns)


def _get_last_user_text(state: AgentState) -> str:
    """直近の通常ユーザー入力を取得する。システム指示は除外する。"""
    for msg in reversed(state.chat_history.messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")

        if isinstance(content, list):
            text = " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            text = str(content)

        if text.startswith("【システム"):
            continue
        return text.strip()
    return ""


def _is_simple_direct_answer_sufficient(user_text: str, answer: str, state: AgentState) -> bool:
    """単純な情報取得質問は、短くても完了回答として扱う。

    「今のディレクトリは？」「ファイルの内容は？」「何が入ってる？」等の
    単純質問に対し、対応するツールが既に実行済みで回答に結果が含まれていれば、
    ガードレール（短い回答・行動予告）を迂回して final_answer とする。
    """
    if not user_text or not answer:
        return False

    q = user_text.strip().lower()
    a = answer.strip()

    # --- cwd / pwd 系 ---
    cwd_questions = [
        "今のディレクトリ", "現在のディレクトリ", "カレントディレクトリ",
        "作業ディレクトリ", "cwd", "pwd", "どこのディレクトリ",
    ]
    if any(k in q for k in cwd_questions):
        used_get_cwd = any(
            str(action).startswith("get_cwd:")
            for action in getattr(state, "executed_actions", [])
        )
        has_path = bool(re.search(r"[A-Z]:\\|/[\w.-]+", a))
        if used_get_cwd and has_path:
            return True

    # --- list_directory / 何が入ってる系 ---
    list_questions = [
        "何が入って", "なにが入って", "ファイル一覧", "ファイル構成",
        "ディレクトリの中", "ディレクトリの中身", "何がある",
        "ファイルがある", "フォルダの中",
    ]
    if any(k in q for k in list_questions):
        used_list_dir = any(
            str(action).startswith("list_directory:")
            for action in getattr(state, "executed_actions", [])
        )
        # 回答にファイル名らしきものが含まれていれば十分
        has_file_entries = bool(re.search(r"[\w]+\.\w+", a))
        if used_list_dir and has_file_entries:
            return True

    # --- read_file / 内容確認系 ---
    read_questions = [
        "内容を教えて", "中身を教えて", "読んで", "見せて",
        "内容は", "中身は", "ファイルを見",
    ]
    if any(k in q for k in read_questions):
        used_read = any(
            str(action).startswith("read_file:")
            for action in getattr(state, "executed_actions", [])
        )
        # 回答が一定文字数以上あれば内容を反映しているとみなす
        if used_read and len(a) >= 30:
            return True

    # --- analyze_file / 要約系 ---
    summary_questions = [
        "要約", "まとめて", "概要", "サマリ", "まとめ",
        "要点", "ポイント", "解説して", "説明して",
    ]
    if any(k in q for k in summary_questions):
        used_analyze = any(
            str(action).startswith("analyze_file:")
            for action in getattr(state, "executed_actions", [])
        )
        if used_analyze and len(a) >= 100:
            return True

    return False


def _answer_completeness_score(content: str, tool_call_count: int) -> int:
    """回答の「最終回答らしさ」を0-100のスコアで評価する。

    単なる長さではなく、最終回答としての構造（結論・根拠・対応案等）で採点する。
    行動予告（「次に〜します」）は減点する。
    スコア >= 50 なら完全な回答とみなす。

    Signals:
        1. 最終回答らしい語句 (+35) — 結論・まとめ・対応案・提案・選択肢等
        2. 根拠や説明 (+20) — なぜなら・理由・つまり等
        3. Markdown書式 (+15) — 見出し・箇条書き・セクション構造
        4. 文字数 (+0~20) — min(len, 400) // 20
        5. ツール実行回数 (+5) — 3回以上なら情報蓄積済み
        6. 文末マーカー (+10) — 。！？. ! ? 等
        7. 具体データ (+10) — パス・数値等
        8. 構造化された説明/提案文 (+20) — 見出し2以上+箇条書き3以上+150字以上
        Penalty:
        - 行動予告 (-50) — 「次に〜します」等
    """
    if not content:
        return 0

    score = 0
    stripped = content.strip()

    # シグナル1: 最終回答らしい構造
    # 調査報告系（結論/対応案/まとめ）に加え、アドバイス・説明系
    # （提案/選択肢/パターン/手順/アドバイス等）も最終回答語句として扱う。
    # コミットメッセージ提案のような「相談への回答」が誤って不完全判定されるのを防ぐ。
    if re.search(r"(結論|原因|調査結果|対応案|改善案|まとめ|"
                 r"おすすめ|ベストプラクティス|総括|概要|結論として|"
                 r"提案|選択肢|以下の通り|解決策|アドバイス|パターン|手順)", stripped):
        score += 35

    # シグナル2: 根拠や説明がある
    if re.search(r"(なぜなら|理由|つまり|具体的には|このため|問題は"
                 r"|したがって|一方|対して)", stripped):
        score += 20

    # シグナル3: Markdown書式（見出し・箇条書き・セクション構造）
    if re.search(r"(^|\n)#{1,3}\s|(^|\n)-\s|(^|\n)\d+\.|\*\*[^*]+\*\*|``", stripped):
        score += 15

    # シグナル4: 文字数（長さだけで完了扱いしないよう上限を下げる）
    score += min(len(stripped), 400) // 20

    # シグナル5: ツール実行回数（情報蓄積の目安だが主力にしない）
    if tool_call_count > 2:
        score += 5

    # シグナル6: 文末マーカー（補助）
    if stripped and stripped[-1] in '。！？.!?\'"」』':
        score += 10

    # シグナル7: 具体データ（パス・ファイル名・数値）
    if re.search(r'[A-Za-z]:\\|/[\w-]+/|\d{2,}|[\w-]+\.\w{1,5}', stripped):
        score += 10

    # シグナル8: 構造化された説明/提案ドキュメント
    # 見出しと箇条書きで組み立てられた十分な長さの説明文は、
    # 調査報告キーワードを含まなくても完成した最終回答とみなす。
    _headings = len(re.findall(r'(?:^|\n)#{1,3}\s', stripped))
    _list_items = len(re.findall(r'(?:^|\n)(?:[-*]|\d+\.)\s', stripped))
    if _headings >= 2 and _list_items >= 3 and len(stripped) >= 150:
        score += 20

    # ペナルティ: 行動予告は大幅減点
    if _looks_like_action_promise(stripped):
        score -= 50

    return max(0, min(score, 100))


# =====================================================
# StreamFilter — 思考ブロックフィルタ
# =====================================================

class StreamFilter:
    """LLMのストリーミング出力から思考ブロックをリアルタイムに除去するフィルター。

    対応する思考ブロック形式:
    - 絵文字形式（U+1F9E0 ... U+1FAE7）
    - <|channel>thought...<channel|> (Gemma形式)
    - <think...>...</think (Qwen3.5 / QwQ / DeepSeek形式)
    - <|tool_response|> 等のモデルアーティファクトも除去

    Args:
        remove_thinking: True の場合、思考ブロックを非表示にする
        start_in_think: フィルタ開始時に既に思考内部にいるか
        capture_thinking: True の場合、思考ブロックの内容を captured_thoughts に保存する
    """

    # # モデル固有の思考タグ（絵文字形式に事前変換される）
    # _THINK_START_REPLACEMENTS = [
    #     ("<|channel>thought", "\U0001f9e0"),
    #     ("<|channel", "\U0001f9e0"),
    #     ("<think", "\U0001f9e0"),
    #     ("<think\n", "\U0001f9e0"),
    # ]
    # _THINK_END_REPLACEMENTS = [
    #     ("<channel|>", "\U0001fae7"),
    #     ("</think", "\U0001fae7"),
    #     ("</think\n", "\U0001fae7"),
    #     ("</think >", "\U0001fae7"),
    # ]

    # 1. 内部マーカーを絵文字から特殊な文字列に変更
    MARKER_START = "___THINK_START_INTERNAL___"
    MARKER_END = "___THINK_END_INTERNAL___"

    # 2. 置換リストをこのマーカーを使用するように更新
    _THINK_START_REPLACEMENTS = [
        ("<|channel>thought", MARKER_START),
        ("<|channel", MARKER_START),
        ("<think", MARKER_START),
        ("<think\n", MARKER_START),
    ]
    _THINK_END_REPLACEMENTS = [
        ("<channel|>", MARKER_END),
        ("</think", MARKER_END),
        ("</think\n", MARKER_END),
        ("</think >", MARKER_END),
    ]

    # 除去すべきモデルアーティファクト
    _ARTIFACT_TAGS = ["<|tool_response>", "<|end_tool_response>", "<|tool_call|>", "<|tool_call_start|>", "<|tool_call_end|>"]  # [LFM専用] 末尾2要素

    def __init__(self, remove_thinking=True, start_in_think=False, capture_thinking=False):
        self.remove_thinking = remove_thinking
        self.in_think = start_in_think
        self.buffer = ""
        self.thought_buffer = ""
        self.capture_thinking = capture_thinking
        self.captured_thoughts: list[str] = []  # 捕捉された思考内容のリスト

    @classmethod
    def _preprocess(cls, text: str) -> str:
        """モデル固有の思考タグを絵文字形式に変換し、アーティファクトを除去する。"""
        for old, new in cls._THINK_START_REPLACEMENTS:
            text = text.replace(old, new)
        for old, new in cls._THINK_END_REPLACEMENTS:
            text = text.replace(old, new)
        for tag in cls._ARTIFACT_TAGS:
            text = text.replace(tag, "")
        return text

    def process(self, text):
        if not self.remove_thinking:
            return text

        text = self._preprocess(text)
        self.buffer += text
        output = ""

        while self.buffer:
            if not self.in_think:
                # 🧠 絵文字ではなく内部マーカーを探す
                start_idx = self.buffer.find(self.MARKER_START)
                if start_idx != -1:
                    output += self.buffer[:start_idx]
                    self.in_think = True
                    self.buffer = self.buffer[start_idx + len(self.MARKER_START):]
                    self.thought_buffer = ""
                else:
                    # 分割チェックもマーカーの長さに合わせる
                    partial_match = False
                    for i in range(1, len(self.MARKER_START)):
                        if self.buffer.endswith(self.MARKER_START[:i]):
                            output += self.buffer[:-i]
                            self.buffer = self.buffer[-i:]
                            partial_match = True
                            break
                    if not partial_match:
                        output += self.buffer
                        self.buffer = ""
                    break
            else:
                # 終わりも同様に内部マーカーで判定
                end_idx = self.buffer.find(self.MARKER_END)
                if end_idx != -1:
                    self.in_think = False
                    if self.capture_thinking:
                        thought_content = self.buffer[:end_idx].strip()
                        if thought_content:
                            self.captured_thoughts.append(thought_content)
                    self.buffer = self.buffer[end_idx + len(self.MARKER_END):]
                    if self.buffer.startswith("\n"):
                        self.buffer = self.buffer[1:]
                    self.thought_buffer = ""
                else:
                    partial_match = False
                    for i in range(1, len(self.MARKER_END)):
                        if self.buffer.endswith(self.MARKER_END[:i]):
                            self.thought_buffer += self.buffer[:-i]
                            self.buffer = self.buffer[-i:]
                            partial_match = True
                            break
                    if not partial_match:
                        self.thought_buffer += self.buffer
                        self.buffer = ""
                    break
        return output

    def flush(self):
        if not self.remove_thinking:
            return self.buffer

        result = ""
        if self.in_think:
            # 思考プロセスが閉じられずに終了した場合
            if self.thought_buffer or self.buffer:
                result = "\n(※思考プロセスが閉じられなかったため内容を表示します)\n" + self.thought_buffer + self.buffer
                if self.capture_thinking and self.thought_buffer.strip():
                    self.captured_thoughts.append(self.thought_buffer.strip())
        else:
            result = self.buffer

        self.thought_buffer = ""
        self.buffer = ""
        self.in_think = False
        return result

    def get_last_thought(self) -> str:
        """最後にキャプチャされた思考内容を返す。"""
        return self.captured_thoughts[-1] if self.captured_thoughts else ""

    def clear_captured_thoughts(self):
        """キャプチャされた思考内容をクリアする。"""
        self.captured_thoughts.clear()


# =====================================================
# プロンプト構築
# =====================================================

def build_base_prompt(context, jit_user_input=None, available_tools=None, thinking_mode="shallow", mode="normal") -> str:
    """フェーズに応じたベースプロンプトを組み立てる（Function Calling版）。

    ツール定義は tools パラメータで別枠送信されるため、
    システムプロンプトには動作ルールのみを含める。

    Args:
        context: AppContext（phase 属性を参照）
        jit_user_input: JITツールスコアリング用のユーザー入力（未使用、後方互換）
        available_tools: 利用可能なツール名のセット（動的プロンプト生成に使用）
        thinking_mode: "shallow"（即断即実・簡潔）または "deep"（複数仮説を推論）

    Returns:
        ベースプロンプト文字列
    """
    return generate_behavior_prompt(available_tools=available_tools, thinking_mode=thinking_mode, mode=mode)


def build_system_text(context, state_board=None, jit_user_input=None, available_tools=None, thinking_mode="shallow", mode="normal") -> str:
    """完全なシステムプロンプトを組み立てる。

    Args:
        context: AppContext
        state_board: AgentStateBoard インスタンス（Noneの場合は空のボードを使用）
        jit_user_input: JITツールスコアリング用のユーザー入力
        available_tools: 利用可能なツール名のセット（動的プロンプト生成に使用）
        thinking_mode: "shallow" または "deep"（基本方針の切り替え）

    Returns:
        システムプロンプト文字列
    """
    base_prompt = build_base_prompt(context, jit_user_input=jit_user_input, available_tools=available_tools, thinking_mode=thinking_mode, mode=mode)
    whiteboard = load_whiteboard_summary(max_chars=1500)
    return build_system_prompt(
        base_prompt,
        state_board=state_board,
        whiteboard_summary=whiteboard,
    )


# =====================================================
# コンテキスト管理 — ユーティリティ
# =====================================================

def _messages_to_text(messages: list[dict]) -> str:
    """メッセージ配列をテキスト文字列に変換する（画像データ除外）。"""
    parts = []
    for m in messages:
        role = m.get("role", "")
        text = _extract_text_from_message(m)
        if role == "tool":
            parts.append(f"tool_result: {text}")
        else:
            parts.append(f"{role}: {text}")
    return "\n".join(parts)


def _strip_think(text: str) -> str:
    """<thinkブロックをテキストから除去する。"""
    return re.sub(r'<think.*?</think', '', text, flags=re.DOTALL).strip()


def _extract_text_from_message(msg: dict) -> str:
    """メッセージからテキスト部分のみを抽出する（画像データ除外）。"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _strip_think(content)
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(_strip_think(item.get("text", "")))
        return " ".join(texts)
    else:
        return _strip_think(str(content))


def get_total_context(llm) -> int:
    """LLMの総コンテキスト長を取得（取得できない場合はconfig.N_CTXを仮定）。"""
    from config import N_CTX
    try:
        if hasattr(llm, 'n_ctx'):
            total_ctx = llm.n_ctx() if callable(llm.n_ctx) else llm.n_ctx
            return int(total_ctx) if total_ctx else N_CTX
        return N_CTX
    except Exception:
        return N_CTX


def _dynamic_tool_cap(usage_ratio: float) -> int:
    """コンテキスト使用率（= current_tokens / safe_max）から、ツール結果1件あたりの
    文字上限を逆算する。余裕があるほど大きく読ませ（中規模ファイルの全文読みを許容）、
    逼迫するほど1件を絞って単発での圧迫を防ぐ。閾値は node_plan の予算ヒント(0.40/0.65)と整合。
    """
    if usage_ratio < 0.40:
        return 16000   # 余裕あり: 全文読みを許容
    elif usage_ratio < 0.65:
        return 12000   # 標準
    else:
        return 6000    # 容量注意: 1件あたりを絞る


# =====================================================
# ホワイトボード型コンテキスト要約
# =====================================================

def _update_whiteboard(llm, popped_messages: list[dict]):
    """切り捨てられたメッセージからホワイトボードを更新する（2段階圧縮）。"""
    print("\n[システム通知] ホワイトボード (CONTEXT_SUMMARY.md) を更新しています...")

    new_log = _messages_to_text(popped_messages)
    if len(new_log) > 6000:
        new_log = new_log[:6000] + "\n...[長すぎるため切り捨て]..."

    existing_board = ""
    if os.path.exists(WHITEBOARD_PATH):
        try:
            with open(WHITEBOARD_PATH, encoding="utf-8") as f:
                existing_board = f.read()
        except Exception:
            pass

    if existing_board:
        user_prompt = (
            f"以下の「既存ホワイトボード」と「新しい会話ログ」を統合して、"
            f"新しいホワイトボードを生成してください。\n\n"
            f"【既存ホワイトボード】\n{existing_board}\n\n"
            f"【新しい会話ログ（切り捨てられた履歴）】\n{new_log}"
        )
    else:
        user_prompt = (
            f"以下の会話ログからホワイトボードを新規作成してください。\n\n"
            f"【会話ログ（切り捨てられた履歴）】\n{new_log}"
        )

    try:
        with SuppressStderr():
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": WHITEBOARD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.2,
                stream=False
            )

        board_content = ""
        if hasattr(response, '__iter__') and not isinstance(response, dict):
            for chunk in response:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "content" in delta:
                        board_content += delta["content"]
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                board_content = message.get("content", "")

        if board_content.strip():
            if "<!-- DETAIL_SECTION -->" not in board_content:
                board_content = board_content.rstrip() + WHITEBOARD_DETAIL_SEPARATOR

            with open(WHITEBOARD_PATH, "w", encoding="utf-8") as f:
                f.write(board_content)
            print("[システム通知] ホワイトボードの更新が完了しました。")
        else:
            print("[警告] ホワイトボードの生成結果が空でした。既存の内容を維持します。")
    except Exception as e:
        print(f"\n[警告] ホワイトボードの更新に失敗しました: {e}")


def load_whiteboard_summary(max_chars: int = 1500) -> str:
    """ホワイトボードの上部セクション（コンテキスト注入用）のみを読み込む。"""
    if not os.path.exists(WHITEBOARD_PATH):
        return ""

    try:
        with open(WHITEBOARD_PATH, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return ""

    if "<!-- DETAIL_SECTION -->" in content:
        upper = content.split("<!-- DETAIL_SECTION -->")[0].strip()
    else:
        upper = content.strip()

    if len(upper) > max_chars:
        upper = upper[:max_chars] + "\n...[注入上限により省略。詳細はgrep_searchでCONTEXT_SUMMARY.mdを検索]..."

    return upper


# =====================================================
# 観測のマスキング（コンテキスト管理）
# =====================================================

def mask_old_observations(messages: list[dict], keep_recent: int = 1) -> list[dict]:
    """古いツール実行結果を要約に置換してコンテキストを節約する。"""
    if not messages:
        return messages

    tool_result_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            tool_result_indices.append(i)

    mask_from = 0
    if len(tool_result_indices) > keep_recent:
        mask_from = len(tool_result_indices) - keep_recent

    for idx in tool_result_indices[:mask_from]:
        msg = messages[idx]
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 100:
            # 行番号付き read_file 結果: ヘッダ + 正確な再取得ヒントを残す
            # （ファイル構造のアウトラインは _compress_tool_result でステートボードにも保存済み）
            m_last = list(re.finditer(r'(?m)^(\d+):\s', content))
            if m_last:
                last_line = int(m_last[-1].group(1))
                m_hdr = re.search(r'^\[[^\]]+\][^\n]*', content)
                hdr = m_hdr.group(0) if m_hdr else "(file read)"
                masked = (f"{hdr}\n... [古い読込を圧縮: {last_line}行目まで表示済み。"
                          f"再参照時は read_file(path, start_line=...) で取得（構造はステートボード参照）] ...")
            else:
                masked = content[:80] + "\n... [Observation masked] (grep_searchで検索可能) ..."
            messages[idx] = {
                "role": "tool",
                "content": masked,
                "tool_call_id": msg.get("tool_call_id", ""),
            }

    return messages


def check_context_checkpoint(
    llm,
    messages: list[dict],
    state_board=None,
) -> str | None:
    """コンテキスト使用量をチェックし、チェックポイント通知を返す。"""
    prompt_text = _messages_to_text(messages)

    token_count = estimate_tokens(llm, prompt_text)

    total_ctx = get_total_context(llm)
    safe_max = max(1000, int(total_ctx) - int(MAX_TOKENS) - CONTEXT_BUFFER)

    threshold = int(safe_max * CONTEXT_CHECKPOINT_THRESHOLD)

    if token_count >= threshold:
        if state_board:
            try:
                state_board._save()
            except Exception:
                pass
        return (f"\n[システム通知] コンテキスト使用量が {token_count}/{safe_max} トークン "
                f"({token_count/safe_max:.0%}) に達しました。"
                f"ステートは自動保存済みです。/clear で会話履歴をリセットしてください"
                f"（ステートは引き継がれます）。\n")

    return None


def check_and_trim_context(llm, messages: list[dict], max_context: int = DEFAULT_TRIM_THRESHOLD) -> list[dict]:
    """推論前にトークン数を計算し、上限を超えそうなら古い履歴を削る。"""
    prompt_text = _messages_to_text(messages)

    token_count = estimate_tokens(llm, prompt_text)

    # Phase 1: ソフトトリム（観測マスキング）
    if token_count > max_context * 0.7:
        mask_old_observations(messages, keep_recent=1)
        prompt_text = _messages_to_text(messages)
        token_count = estimate_tokens(llm, prompt_text)

    # Phase 2: ハードトリム（古いメッセージ削除）
    if token_count > max_context:
        print("\n[システム通知] コンテキスト上限に接近しています。古い履歴を削除してホワイトボードに退避します...")

        popped_messages = []

        while len(messages) > 3 and token_count > max_context:
            popped = messages.pop(1)
            popped_messages.append(popped)
            # assistant(tool_calls) の後に続く tool メッセージも一緒に削除
            if popped.get("role") == "assistant" and popped.get("tool_calls"):
                while (len(messages) > 2
                       and messages[1].get("role") == "tool"):
                    popped_messages.append(messages.pop(1))

            prompt_text = _messages_to_text(messages)
            token_count = estimate_tokens(llm, prompt_text)

        if popped_messages:
            _update_whiteboard(llm, popped_messages)

    return messages


# =====================================================
# ツール実行（インターセプト）
# =====================================================

#: /review モードでレビュー対象とする編集ツールの明示集合。
#: _FILE_EDIT_TOOLS から write_sections（別経路でセクション毎生成＝単一の変更案なし）を除く。
#: 将来 _FILE_EDIT_TOOLS が増えても意図せずレビューが走らないよう、独立集合とする。
_REVIEWABLE_EDITS = frozenset({"write_file", "replace_lines", "search_and_replace", "append_to_file"})


def execute_tool(context, tool_name: str, tool_args: dict, output_fn) -> str:
    """ツールを実行し、結果文字列を返す。

    analyze_file, view_image はインターセプトしてサブクエリ処理を行う。
    それ以外は execute_builtin_tool に委譲する。

    Args:
        context: AppContext（llm, use_vision 等を保持）
        tool_name: 実行するツール名
        tool_args: ツールに渡す引数辞書
        output_fn: テキスト出力用コールバック

    Returns:
        ツール実行結果の文字列
    """
    # インターセプト: view_image（VLMサブクエリ方式）
    if tool_name == "view_image" and context.use_vision:
        img_path = tool_args.get("path")
        try:
            output_fn("[System] 画像を裏で解析中...\n", end="", flush=True)
            img_data_url = resize_and_encode_image(img_path)
            analysis_prompt = tool_args.get("analysis_prompt")
            with SuppressStderr():
                image_description = run_vision_subquery(context.llm, img_data_url, prompt=analysis_prompt)
            return f"画像 ({os.path.basename(img_path)}) の解析結果:\n{image_description}"
        except Exception as e:
            return f"画像の読み込みに失敗しました: {e}"

    # インターセプト: analyze_file（テキストサブクエリ方式 + キャッシュチェック）
    elif tool_name == "analyze_file":
        return _execute_analyze_file(context, tool_args, output_fn)

    # インターセプト: write_sections（セクションごとサブクエリ生成）
    elif tool_name == "write_sections":
        return _execute_write_sections(context, tool_args, output_fn)

    # インターセプト: delegate_research（独立サブエージェントで調査委譲）
    elif tool_name == "delegate_research":
        return _execute_delegate_research(context, tool_args, output_fn)

    # 通常のツール実行（ファイル書き込み系は事前にバックアップ）
    else:
        _backup_if_file_edit(tool_name, tool_args)
        result = execute_builtin_tool(tool_name, tool_args)
        # 編集後に ruff で構文/未定義名を検査し、違反を observation に付加
        if tool_name in _FILE_EDIT_TOOLS and not result.startswith("Error"):
            ruff_out = _run_ruff_check(tool_args.get("path", ""))
            if ruff_out:
                result = f"{result}\n{ruff_out}"
            # /review モード: 読み取り専用レビューアで編集を検証し、判定を observation に付加
            # （observe-only・編集は実行済み。失敗時は "" が返り何も付加されない）
            if getattr(context, "review_mode", False) and tool_name in _REVIEWABLE_EDITS:
                verdict = _run_edit_review(context, tool_name, tool_args, output_fn)
                if verdict:
                    result = f"{result}\n{verdict}"
            # /verify モード: 実行ベース検証 → 自動修正ループ（.py のみ・observe-and-fix）。
            # review の後に走り、実行エラーがあれば自動で編集し直す。失敗時は "" で何も付加しない
            if getattr(context, "verify_mode", False) and tool_name in _REVIEWABLE_EDITS:
                try:
                    sb = registry._state_board
                    goal = getattr(sb, "goal", "") if sb else ""
                except Exception:
                    goal = ""
                verify_summary = run_verify_fix_loop(
                    context, tool_args.get("path", ""), tool_name, tool_args, goal, output_fn
                )
                if verify_summary:
                    result = f"{result}\n{verify_summary}"
        return result


def _execute_write_sections(context, tool_args: dict, output_fn) -> str:
    """write_sections のインターセプト処理。

    各セクションを個別のサブクエリで生成し、ファイルに順次書き込む。
    メインのReActコンテキストを汚さずに長文ドキュメントを生成できる。
    """
    from pathlib import Path as PathLib

    file_path = str(tool_args.get("path", ""))
    sections = tool_args.get("sections", [])
    doc_context = tool_args.get("context", "")

    if not file_path:
        return "Error: path は必須です。"
    if not sections or not isinstance(sections, list):
        return "Error: sections は空でない配列で指定してください。"

    target = PathLib(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    written_sections = []
    total_chars = 0
    errors = []

    for i, section in enumerate(sections):
        heading = section.get("heading", f"## セクション {i + 1}")
        instruction = section.get("instruction", "")

        output_fn(f"[System] セクション '{heading}' を生成中 ({i + 1}/{len(sections)})...\n",
                  end="", flush=True)

        # サブクエリ用プロンプト構築
        system_prompt = (
            "あなたは優秀なドキュメント作成アシスタントです。"
            "指定されたセクションの見出しと指示に従って、詳細で充実した内容をMarkdown形式で出力してください。\n"
            "厳守事項:\n"
            "- 見出し行自身は出力しない（呼び出し側で出力するため）\n"
            "- 「...」や「（省略）」「（以下同様）」等の省略記号は絶対に使わない\n"
            "- 箇条書きだけでなく、説明文も含める\n"
            "- 日本語で記述する\n"
            "- 読者が知りたい全ての情報を過不足なく書く"
        )

        user_parts = ["以下のセクションの本文を書いてください。\n"]
        if doc_context:
            user_parts.append(f"【ドキュメント全体の文脈】\n{doc_context}\n")
        if i > 0:
            prev_headings = [s.get("heading", "") for s in sections[:i] if s.get("heading")]
            if prev_headings:
                user_parts.append("【これまでのセクション見出し】\n" + "\n".join(prev_headings) + "\n")
        user_parts.append(f"【セクション見出し】\n{heading}\n")
        if instruction:
            user_parts.append(f"【このセクションで書く内容の指示】\n{instruction}")

        user_prompt = "\n".join(user_parts)

        try:
            with SuppressStderr():
                response = context.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.5,
                    stream=False,
                )

            content = _collect_subquery_response(response)

            if not content or not content.strip():
                content = f"（{heading} の内容生成に失敗しました）"
                errors.append(heading)
        except Exception as e:
            content = f"（生成エラー: {e}）"
            errors.append(heading)

        # ファイルに書き込み
        section_text = f"\n\n{heading}\n\n{content.strip()}"
        if i == 0:
            target.write_text(section_text.lstrip('\n'), encoding="utf-8")
        else:
            with open(target, 'a', encoding='utf-8') as f:
                f.write(section_text)

        total_chars += len(content.strip())
        written_sections.append(heading)
        output_fn(f"  ✓ {heading} ({len(content.strip())}文字)\n", end="", flush=True)

    # 結果サマリー
    result_lines = [
        f"Success: {file_path} に {len(written_sections)} セクション (計{total_chars}文字) を書き込みました。",
    ]
    for ws in written_sections:
        result_lines.append(f"  - {ws}")
    if errors:
        result_lines.append(f"警告: {len(errors)} セクションで生成エラーが発生しました: {', '.join(errors)}")

    return "\n".join(result_lines)


# =====================================================
# 並列/直列実行エンジン
# =====================================================

def classify_tools(tool_calls: list[dict]) -> tuple[list[dict], list[dict]]:
    """ツール呼び出しを読み取り専用(並列)と破壊的操作(直列)に分類する。
       大きなファイルの analyze_file はコンテキスト溢れを防ぐため動的に直列へ回す。
    """
    readonly = []
    destructive = []

    # 並列処理を許容する最大ファイルサイズ（バイト）
    # 例: 8192バイト (約8KB)。これより大きいファイルは順番(直列)に処理される
    PARALLEL_MAX_SIZE = 8192

    for call in tool_calls:
        func = call.get("function", {})
        tool_name = func.get("name", "")

        # --- 動的ルーティング: analyze_file のサイズチェック ---
        if tool_name == "analyze_file":
            try:
                args = _safe_parse_args(func)
                file_path = args.get("path", "")
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > PARALLEL_MAX_SIZE:
                        # サイズが大きい場合は直列(destructive)リストへ逃がす
                        destructive.append(call)
                        continue
            except Exception:
                pass  # パースエラー等があれば安全のため通常の分類へ
        # -----------------------------------------------------

        if tool_name in READONLY_TOOLS:
            readonly.append(call)
        elif tool_name in DESTRUCTIVE_TOOLS:
            destructive.append(call)
        else:
            # 未知のツールは安全のため直列で実行
            destructive.append(call)

    return readonly, destructive


def execute_parallel(
    tool_calls: list[dict],
    executor_fn,
    max_workers: int = MAX_PARALLEL_TOOLS,
) -> list[tuple[dict, str]]:
    """読み取り専用ツールを並列実行し、結果を元の順序で返す。

    Args:
        tool_calls: 読み取り専用ツール呼び出し辞書のリスト
        executor_fn: ツール実行関数。シグネチャ:
                     executor_fn(tool_name: str, tool_args: dict) -> str
        max_workers: 最大並列スレッド数（デフォルト: MAX_PARALLEL_TOOLS）

    Returns:
        [(tool_call, result_str), ...] のリスト（元の順序）
    """
    if not tool_calls:
        return []

    results = [None] * len(tool_calls)

    # スレッド数はツール数と max_workers の小さい方
    workers = min(len(tool_calls), max_workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {}
        for i, call in enumerate(tool_calls):
            func = call.get("function", {})
            tool_name = func.get("name", "")
            tool_args = _safe_parse_args(func)
            future = pool.submit(executor_fn, tool_name, tool_args)
            future_to_index[future] = i

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result_str = future.result()
            except Exception as e:
                result_str = f"[並列実行エラー] {e}"
            results[idx] = (tool_calls[idx], result_str)

    # None が混入していないか確認（念のため）
    return [(call, res) for call, res in results if res is not None]


def execute_serial(
    tool_calls: list[dict],
    executor_fn,
) -> list[tuple[dict, str]]:
    """ツールを直列（1つずつ）に実行し、結果を返す。

    破壊的操作ツールの実行や、並列不可のツールに使用する。

    Args:
        tool_calls: ツール呼び出し辞書のリスト
        executor_fn: ツール実行関数。シグネチャ:
                     executor_fn(tool_name: str, tool_args: dict) -> str

    Returns:
        [(tool_call, result_str), ...] のリスト
    """
    results = []
    for call in tool_calls:
        func = call.get("function", {})
        tool_name = func.get("name", "")
        tool_args = _safe_parse_args(func)
        try:
            result_str = executor_fn(tool_name, tool_args)
        except Exception as e:
            result_str = f"[実行エラー] {e}"
        results.append((call, result_str))
    return results


# =====================================================
# State Graphエージェントエンジン
# =====================================================

def _extract_latest_user_input(messages: list[dict]) -> str:
    """メッセージ履歴から最新のユーザー入力テキストを抽出する（JITスコアリング用）。"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Vision形式: [{"type": "text", ...}, {"type": "image_url", ...}]
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return " ".join(texts)
    return ""


# =====================================================
# デバッグ — コンテキストダンプ
# =====================================================

def _dump_debug_context(context, state, system_msg, messages, messages_for_llm,
                        tool_names, tools, jit_input, safe_max, total_ctx):
    """デバッグモード時にLLMへの入力コンテキストをファイルにダンプする。

    context.debug_mode が "summary" または "full" の場合に呼び出される。
    ファイルパス: .pixie_notes/debug/turn_NNN.md
    """
    mode = getattr(context, 'debug_mode', 'summary')
    turn = getattr(context, 'debug_turn', 0)

    debug_dir = get_data_path("debug")
    os.makedirs(debug_dir, exist_ok=True)
    filepath = os.path.join(debug_dir, f"turn_{turn:03d}.md")

    lines = []
    lines.append(f"=== Debug Turn {turn} ===")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Mode: {mode}")
    lines.append("")

    # --- System Prompt ---
    lines.append("--- System Prompt ---")
    if system_msg:
        sys_content = system_msg.get("content", "")
        lines.append(f"Length: {len(sys_content)} chars")
        lines.append(f"StateBoard: {'active' if state.state_board and not state.state_board.is_empty() else 'empty'}")
        lines.append(f"Whiteboard: {'loaded' if 'ホワイトボード' in sys_content else 'none'}")
        if mode == "full":
            lines.append("")
            lines.append("```")
            lines.append(sys_content)
            lines.append("```")
        else:
            lines.append(f"Preview: {sys_content[:100]}...")
    else:
        lines.append("(no system message)")
    lines.append("")

    # --- Messages ---
    target_msgs = messages_for_llm if messages_for_llm else messages
    lines.append(f"--- Messages ({len(target_msgs)} messages) ---")
    for i, msg in enumerate(target_msgs, 1):
        role = msg.get("role", "unknown")
        content = _extract_text_from_message(msg)
        char_len = len(content)
        # Feature A 観測性: 履歴に <think> が保持されているか（carryover）
        raw_content = msg.get("content", "")
        carryover_note = ""
        if role == "assistant" and isinstance(raw_content, str) and "<think" in raw_content:
            carryover_note = f" [carryover: <think> preserved, {len(raw_content)} chars]"
        if mode == "full":
            lines.append(f"\n[{i}] {role} ({char_len} chars){carryover_note}:")
            lines.append("```")
            lines.append(content if content else "(empty)")
            lines.append("```")
        else:
            preview = content[:100].replace("\n", "\\n") if content else "(empty)"
            label = role
            if role == "tool":
                # tool role の場合、tool_call_id があれば表示
                tc_id = msg.get("tool_call_id", "")
                label = f"tool (id={tc_id[:8]}...)" if tc_id else "tool"
            lines.append(f"[{i}] {label}: \"{preview}\" ({char_len} chars){carryover_note}")
    lines.append("")

    # --- JIT Tool Selection ---
    lines.append("--- JIT Tool Selection ---")
    if tool_names:
        # スコア情報を取得するため再スコアリング
        if jit_input:
            input_lower = jit_input.lower()
            input_tokens = set(re.findall(r'[\w]+', input_lower))
            for name in tool_names:
                entry = TOOL_REGISTRY.get(name, {})
                searchable = (entry.get("prompt_desc", "") + " " + entry.get("description", "")).lower()
                desc_tokens = set(re.findall(r'[\w]+', searchable))
                overlap = input_tokens & desc_tokens
                score = len(overlap)
                if name.lower() in input_lower:
                    score += 5
                lines.append(f"  {name}: {score:.1f}")
        else:
            lines.append(f"  (no JIT input — tools: {', '.join(tool_names)})")
    else:
        lines.append("  (no tool filtering)")
    if tools:
        lines.append(f"Tool schemas sent: {len(tools)} / {len(TOOL_REGISTRY)} total")
    lines.append("")

    # --- Context Usage ---
    lines.append("--- Context Usage ---")
    total_chars = sum(len(_extract_text_from_message(m)) for m in target_msgs)
    est_tokens = total_chars // 3
    lines.append(f"Estimated chars: {total_chars}")
    lines.append(f"Estimated tokens: ~{est_tokens} / safe_max {safe_max} ({est_tokens / max(safe_max, 1) * 100:.0f}%)")
    lines.append(f"Total context window: {total_ctx}")
    lines.append("")

    # --- State Board Content ---
    if state.state_board and not state.state_board.is_empty():
        lines.append("--- State Board ---")
        lines.append(state.state_board.to_injection_text(max_chars=2000))
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def node_plan(context, state: AgentState, *, show_thinking: bool = True, max_tokens: int = MAX_TOKENS, output_fn=None, system_msg_builder=None) -> tuple[str | None, list[dict] | None]:
    """Plan ノード: LLMに次のアクションを考えさせる（Function Calling版）。

    tools パラメータでツール定義を別枠送信し、
    LLMからの tool_calls / content をストリーミングで受信する。

    Args:
        context: AppContext
        state: AgentState
        show_thinking: 思考ブロック表示フラグ
        max_tokens: LLM生成時の最大トークン数
        output_fn: テキスト出力用コールバック
        system_msg_builder: システムプロンプト構築関数（jit_user_input 引数対応）

    Returns:
        (content, tool_calls) タプル
        - content: テキスト回答（ツール呼び出し時も含む）、Noneの場合あり
        - tool_calls: OpenAI形式のツール呼び出しリスト、ツール呼び出しなしの場合はNone
    """
    if output_fn is None:
        output_fn = _default_output_fn

    # 最新のユーザー入力を抽出（JITスコアリング用）
    jit_input = _extract_latest_user_input(state.chat_history.messages)

    # JITツールフィルタリング — プロンプト構築の前に実行して整合性を保証
    # ツール選択: /code モードは固定 CODE_TOOL_SET（JITスコアリングをバイパス）
    code_mode = getattr(context, 'code_mode', False)
    if code_mode:
        from config import CODE_TOOL_SET
        available_tools = set(CODE_TOOL_SET)
    else:
        available_tools = set(score_tools(jit_input, top_n=5)) if jit_input else None
    tools = registry_to_openai_tools(list(available_tools) if available_tools else None)

    # 思考深度モードの判定（段階的思考深化）。/code モードは強制 deep
    thinking_mode = _resolve_thinking_mode(
        state, jit_input, force_deep=(getattr(context, 'force_deep', False) or code_mode))

    system_msg = None
    if system_msg_builder:
        system_text = system_msg_builder(
            context, state.state_board,
            jit_user_input=jit_input,
            available_tools=available_tools,
            thinking_mode=thinking_mode,
            mode="code" if code_mode else "normal",
        )
        # ※ thinking_notes の先頭注入は廃止（Feature A）: 直前の <think> を履歴に
        #    残すようにしたため重複解消。システムプロンプト先頭が安定化し、
        #    将来のプレフィックス/KVキャッシュ再利用にも寄与する。
        system_msg = {"role": "system", "content": system_text}

    messages = state.chat_history.get_messages(system_msg)

    # コンテキストのトリミング
    total_ctx = get_total_context(context.llm)
    safe_max = max(MIN_CONTEXT_TOKENS, int(total_ctx) - int(max_tokens) - CONTEXT_BUFFER)
    messages = check_and_trim_context(context.llm, messages, max_context=safe_max)

    # チェックポイント通知
    checkpoint = check_context_checkpoint(context.llm, messages, state_board=state.state_board)
    if checkpoint:
        output_fn(checkpoint, end="", flush=True)

    # トリミング後のメッセージをchat_historyに反映
    if messages and messages[0].get("role") == "system":
        state.chat_history.messages = messages[1:]
    else:
        state.chat_history.messages = messages

    # コンテキスト使用率に応じた読み取り戦略ヒントをシステムメッセージに注入
    # （available_tools に基づいて動的にツール名を参照）
    _at = available_tools or set(TOOL_REGISTRY.keys())
    prompt_text = _messages_to_text(messages)
    token_count = estimate_tokens(context.llm, prompt_text)
    usage_ratio = token_count / safe_max if safe_max > 0 else 1.0

    # ツール結果の文字上限をコンテキスト使用率から逆算して動的設定
    # （この直後に実行される Action の並列/直列実行で参照される）
    registry.set_tool_result_max_chars(_dynamic_tool_cap(usage_ratio))

    if usage_ratio < 0.40:
        _outline = next((t for t in ("get_code_outline", "research_code_paths") if t in _at), None)
        _analysis = "analyze_file" if "analyze_file" in _at else None
        hint_parts = [f"【コンテキスト使用率: {usage_ratio:.0%} — 余裕あり】"]
        if "read_file" in _at:
            hint_parts.append("複数ファイルの並列 read_file も有効です。")
            if state.tool_call_count < 3:
                hint_parts.append(
                    "コンテキストに余裕があるため、read_file は start_line/end_line を省略して全文を読んで構いません。"
                    "search_and_replace を使う場合は特に、正確なコードを得るために全文読みを推奨します。"
                )
        structure_tools = [f"`{t}`" for t in (_outline, _analysis) if t]
        if structure_tools:
            hint_parts.append(f"構造把握なら {'、'.join(structure_tools)} を優先してください。")
        budget_hint = "\n\n" + "".join(hint_parts)
    elif usage_ratio > 0.65:
        hint_parts = [f"【コンテキスト使用率: {usage_ratio:.0%} — 容量注意】"]
        if "read_file" in _at:
            hint_parts.append("read_file は start_line/end_line で必要最小限の範囲だけ読んでください。")
        _outline = next((t for t in ("get_code_outline", "research_code_paths") if t in _at), None)
        _analysis = "analyze_file" if "analyze_file" in _at else None
        alt_tools = [f"`{t}`" for t in (_outline, _analysis) if t]
        if alt_tools:
            hint_parts.append(f"全文が不要なら {'、'.join(alt_tools)} を使用してください。")
        budget_hint = "\n\n" + "".join(hint_parts)
    else:
        budget_hint = None

    if budget_hint:
        system_msg = {"role": "system", "content": system_text + budget_hint}

    # deep モードの深化プロンプト注入（文字化けしていたSYNTHESIZING hintを正常化）
    if thinking_mode == "deep" and system_msg is not None:
        deep_hint = (
            "\n\n【現在のフェーズ: 統合分析（深度思考）】\n"
            "これまでのツール実行結果で十分な情報が揃いました。以下の思考プロセスを踏んでください:\n"
            "1. <think> ブロック内で、収集した事実を統合し、複数の仮説を立てて深く推論してください。\n"
            "2. 各仮説の根拠と反証を比較し、最も妥当な結論を導いてください。\n"
            "3. 結論がまとまったら update_state(found_knowledge='...') で記録してください。\n"
            "4. 必要なアクション（search_and_replace 等）を実行してください。\n"
            "※ じっくり考えてください。急いでツールを呼ぶ必要はありません。"
        )
        system_msg = {"role": "system", "content": system_msg["content"] + deep_hint}

    # 定期的な状態整理リマインダー（deepモード時は長考の邪魔になるためスキップ）
    REPORT_INTERVAL = 3
    if thinking_mode != "deep" and state.tool_call_count % REPORT_INTERVAL == 0:
        reminder_content = (
            "【システム強制指示】裏でのツール実行が連続しています。"
            "次のツールを呼び出す前に、必ず「これまでに何が分かったか」「今から何をするか」をユーザーに向けて日本語で簡潔に報告してください。"
            "※JSONやツール呼び出しだけでなく、必ず自然言語での説明を含めること。"
        )
        msgs = state.chat_history.messages
        if not msgs or msgs[-1].get("content") != reminder_content:
            state.chat_history.add("user", reminder_content)
            # 強制的にテキストを出させるため、一時的に tools を空にして送信する（任意）

    messages = state.chat_history.get_messages(system_msg)

    # 動的温度: ツール呼び出しが閾値を超えたら温度を下げてループ抑制
    # deepモードでは創発的な深い推論を阻害しないよう下限を 0.5 に引き上げる
    temp = TEMPERATURE_MAIN
    temp_floor = 0.5 if thinking_mode == "deep" else 0.3
    if state.tool_call_count > TEMPERATURE_LOOP_THRESHOLD:
        temp = max(temp_floor, temp - (state.tool_call_count - TEMPERATURE_LOOP_THRESHOLD) * 0.05)

    # モデル互換性: role="tool" の変換
    # supports_tool_role=True の場合（Qwen3/Gemma-FC等）はそのまま送信
    # False の場合（LM Studio + 非対応モデル）は role="user" に変換
    # （role="assistant" だとAIがツール結果を真似てエコーするため）
    messages_for_llm = []
    for msg in messages:
        if msg.get("role") == "tool" and not getattr(context, "supports_tool_role", False):
            tool_content = msg.get("content", "")
            messages_for_llm.append({
                "role": "user",
                "content": f"[ツール結果]\n{tool_content}",
            })
        else:
            messages_for_llm.append(msg)

    # デバッグモード: LLM呼び出し直前にコンテキストをファイルにダンプ
    if getattr(context, 'debug_mode', False):
        _dump_debug_context(context, state, system_msg, messages, messages_for_llm,
                            available_tools, tools, jit_input, safe_max, total_ctx)

    ai_prompt_printed = False
    think_timeout = False  # deepモードの <think> タイムアウト検知

    # フェーズ検出用状態（Prefill / Thinking / Generating）
    _phase = "prefill"
    _prefill_start = None
    _prefill_secs = 0.0
    _thinking_start = None
    _thinking_total = 0.0
    _indicator_on = False  # \r で上書き可能な行が画面にあるか

    # Prefill開始タイマー
    _prefill_start = time.monotonic()
    output_fn("  ⏳ Prefill...", end="", flush=True)
    _indicator_on = True

    # deepモード時は <think> の分を含めて max_tokens を増やす
    # （tool_choice="auto" + tools渡しが <think> を短く切る対策。両バックエンドで確実に効く）
    if thinking_mode == "deep":
        effective_max_tokens = min(max_tokens * 2, total_ctx // 2)
    else:
        effective_max_tokens = max_tokens

    with SuppressStderr():
        response = context.llm.create_chat_completion(
            messages=messages_for_llm,
            max_tokens=effective_max_tokens,
            temperature=temp,
            stream=True,
            tools=tools,
            tool_choice="auto",
        )

    stream_filter = StreamFilter(remove_thinking=not show_thinking, start_in_think=False, capture_thinking=True)
    stream_chunks = []
    finish_reason = None
    # ストリーミング反復検知用のバッファ
    accumulated_raw = ""
    last_repcheck_pos = 0
    repetition_detected = False
    REPCHECK_INTERVAL = 200  # N文字ごとに反復チェック

    for chunk in response:
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})

        # 最初のチャンク = Prefill完了
        if _phase == "prefill":
            _prefill_secs = time.monotonic() - _prefill_start
            if _indicator_on:
                output_fn(f"\r  ✅ Prefill: {_prefill_secs:.1f}s  ", end="", flush=True)

        # テキストコンテンツをストリーム表示
        if delta.get("content"):
            raw_text = delta["content"]
            accumulated_raw += raw_text
            filtered = stream_filter.process(raw_text)

            # フェーズ遷移の追跡
            _now_in_think = stream_filter.in_think

            # /think ON の場合、StreamFilter はパススルー（in_think は変化しない）
            # → prefill 完了後は直接 generating に遷移
            if show_thinking and _phase in ("prefill", "thinking"):
                _phase = "generating"

            # → 思考フェーズに入った（/think OFF の場合のみインジケータ表示）
            if _now_in_think and _phase != "thinking":
                _phase = "thinking"
                _thinking_start = time.monotonic()
                if not show_thinking and _indicator_on:
                    output_fn("\r  🧠 Thinking...  ", end="", flush=True)

            # → 思考フェーズから出た（生成開始）
            if not _now_in_think and _phase == "thinking":
                if _thinking_start:
                    _thinking_total += time.monotonic() - _thinking_start
                    _thinking_start = None
                _phase = "generating"

            # think タイムアウト（deep モードの無限長考防止）
            if thinking_mode == "deep" and _thinking_start is not None:
                elapsed = _thinking_total + (time.monotonic() - _thinking_start)
                if elapsed > DEEP_THINK_BUDGET_SEC:
                    think_timeout = True
                    # 思考状態をクリーンアップ（未閉じ<think>のflush表示を防ぐ）
                    stream_filter.in_think = False
                    stream_filter.thought_buffer = ""
                    stream_filter.buffer = ""
                    output_fn("\n[システム通知: 思考時間が上限に達したため、結論生成に移ります。]\n", end="", flush=True)
                    break

            # → thinkなしモデル: prefill 直後にテキストが出た場合は generating に遷移
            if _phase == "prefill" and filtered:
                _phase = "generating"

            # → 生成フェーズでテキストを出力
            if _phase == "generating":
                if filtered:
                    if _indicator_on:
                        output_fn("\r                \r", end="", flush=True)
                        _indicator_on = False
                    if not ai_prompt_printed:
                        output_fn("AI: ", end="", flush=True)
                        ai_prompt_printed = True
                    output_fn(filtered, end="", flush=True)

            # 反復パターンの定期チェック（思考ブロックを除外して検知）
            if len(accumulated_raw) - last_repcheck_pos >= REPCHECK_INTERVAL:
                visible_text = _strip_all_thinking(accumulated_raw)
                rep_min = 5 if thinking_mode == "deep" else 3
                if _detect_repetitive_content(visible_text, min_repeats=rep_min):
                    repetition_detected = True
                    output_fn("\n[システム通知: 出力の反復ループを検知。生成を中断します。]\n", end="", flush=True)
                    break
                last_repcheck_pos = len(accumulated_raw)

        stream_chunks.append(chunk)

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
            if finish_reason == "tool_calls":
                break
            if finish_reason == "length":
                # 出力上限に到達する前に反復検知されていないか確認（思考ブロック除外）
                visible_at_length = _strip_all_thinking(accumulated_raw)
                if _detect_repetitive_content(visible_at_length):
                    repetition_detected = True
                    output_fn("\n[システム通知: 出力の反復ループを検知。継続生成を中止します。]\n", end="", flush=True)
                    break
                output_fn("\n\n[システム通知: 出力が最大文字数(トークン上限)に達したため途中で終了しました。自動で続きを生成します。]", end="", flush=True)

    # フェーズインジケータの残りをクリア
    if _indicator_on:
        output_fn("\r                \r", end="", flush=True)
        _indicator_on = False

    final_text = stream_filter.flush()
    if final_text:
        if not ai_prompt_printed:
            output_fn("AI: ", end="", flush=True)
            ai_prompt_printed = True
        output_fn(final_text, end="", flush=True)
    if ai_prompt_printed:
        output_fn("\n", end="", flush=True)

    # チャンクから content と tool_calls を蓄積・抽出
    content, tool_calls = _accumulate_tool_calls(stream_chunks)

    # デバッグモード: タイミング情報をダンプファイルに追記
    if getattr(context, 'debug_mode', False):
        turn = getattr(context, 'debug_turn', 0)
        debug_dir = get_data_path("debug")
        filepath = os.path.join(debug_dir, f"turn_{turn:03d}.md")
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write("\n--- Timing ---\n")
                f.write(f"Prefill: {_prefill_secs:.3f}s\n")
                f.write(f"Thinking: {_thinking_total:.3f}s\n")
        except Exception:
            pass

    # GGUFモデル（Qwen3.5等）は tool_calls を構造化して返さない場合がある。
    # テキストから <tool_call...> ブロックを抽出して補完する。
    if tool_calls is None and content:
        content, tool_calls = _parse_native_tool_calls(content)
        if tool_calls:
            finish_reason = "tool_calls"

    state.last_response = content or ""

    # deep モードで <think> を捕捉していれば、次ターンへ引き継ぐ（末尾抽出）
    if thinking_mode == "deep":
        last_thought = stream_filter.get_last_thought()
        if last_thought and len(last_thought) > 30:
            snippet = _truncate_thought(last_thought, max_chars=400)
            if snippet:
                state.thinking_notes.append(snippet)
                state.thinking_notes = state.thinking_notes[-2:]  # 直近2件のみ保持

    # think タイムアウト時: 思考を破棄して結論生成を促す
    if think_timeout:
        state.chat_history.add("user",
            "【システム指示】推論に十分な時間をかけました。これまでの思考を整理し、"
            "結論・根拠・対応案を日本語で出力してください。これ以上 <think> で推論する必要はありません。")
        state.guardrail_cooldown = 1
        return content or "", None

    # 反復検知時: 内容をクリーンにして強制的にツール呼び出しを促す
    if repetition_detected and not tool_calls:
        # 反復内容を履歴に追加しない（汚染防止）
        clean_content = _strip_all_thinking(content or "")
        if clean_content:
            # 短縮版のみ履歴に追加
            short = clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
            state.chat_history.add("assistant", short)
        state.chat_history.add("user",
            "【システム強制指示】あなたは直前の出力で同じ内容を何度も繰り返しました。"
            "即座に最も適切なツールを1つだけ呼び出してください。"
            "迷わず、検討せず、最初に思いついたツールを即実行してください。"
            "それ以上考える必要はありません。")
        state.guardrail_cooldown = 2
        return content or "", None

    # 出力が途中で切れた場合の継続処理
    if finish_reason == "length" and not tool_calls:
        _add_assistant_with_think(state, content or "")
        state.chat_history.add("user", "出力が途中で切れました。続きを出力してください。")
        state.phase = "NEEDS_CONTINUATION"
        return content or "", None

    return content or "", tool_calls


def node_action(context, state: AgentState, tool_call: dict, *, output_fn=None) -> str:
    """Action ノード: Planで指定されたツールを安全に実行する。

    execute_tool に処理を委譲するラッパー。
    tool_call dict から tool_name / tool_args を抽出して渡す。

    Args:
        context: AppContext
        state: AgentState
        tool_call: ツール情報辞書（tool_name + 引数）
        output_fn: テキスト出力用コールバック

    Returns:
        ツール実行結果の文字列
    """
    if output_fn is None:
        output_fn = _default_output_fn

    tool_name = tool_call.get("tool_name")
    tool_args = {k: v for k, v in tool_call.items() if k != "tool_name"}

    return execute_tool(context, tool_name, tool_args, output_fn)


def node_observe(state: AgentState, tool_name: str, tool_result: str, *, output_fn=None) -> str:
    """Observe ノード: ツール実行結果を評価し、状態を更新する。

    - AgentStateBoard のタスクリストを更新（完了・未完了）
    - ループ検知
    - エラー記録
    - 次のフェーズの決定

    Args:
        state: AgentState
        tool_name: 実行されたツール名
        tool_result: ツールの実行結果
        output_fn: テキスト出力用コールバック

    Returns:
        次のフェーズ ("PLANNING" = 続行, "DONE" = 終了)
    """
    if output_fn is None:
        output_fn = _default_output_fn

    # エラー検出と記録
    if tool_result.startswith("Error:"):
        state.state_board.add_error(f"{tool_name}: {tool_result[:200]}")

    # update_state が呼ばれた場合の処理
    if tool_name == "update_state":
        pass  # state_board はツール内で直接更新される

    state.tool_call_count += 1

    # 最大ツール呼び出し回数チェック
    if state.tool_call_count >= state.max_tool_calls:
        state.exit_reason = f"max_tool_calls_reached (連続実行上限 {state.max_tool_calls}回に到達)"
        output_fn(f"\n[System] ReActループ終了: {state.exit_reason}\n", end="", flush=True)
        return "DONE"

    return "PLANNING"


def run_graph(context, state: AgentState, *, show_thinking: bool = True, max_tokens: int = MAX_TOKENS, output_fn=None, system_msg_builder=None, interactive_fn=None) -> str:
    """State Graphの実行エンジン（Function Calling版）。

    Plan -> Action -> Observe のサイクルを回す。
    ツール呼び出しは tools パラメータ経由で行い、
    レスポンスの tool_calls を直接使用する。

    Args:
        context: AppContext
        state: AgentState
        show_thinking: 思考ブロック表示フラグ
        max_tokens: LLM生成時の最大トークン数
        output_fn: テキスト出力用コールバック
        system_msg_builder: システムプロンプト構築関数
        interactive_fn: 半自動モード用コールバック。ツール実行前に呼び出される。
                        (tool_calls, content) -> (approved_calls, user_override)
                        Noneの場合は完全自律動作（従来の動作）

    Returns:
        最終的なLLMの回答テキスト
    """
    if output_fn is None:
        output_fn = _default_output_fn

    state.phase = "PLANNING"
    code_mode = getattr(context, 'code_mode', False)  # /code モード: 一部ガードレールを緩和
    final_answer = ""
    last_substantive_content = ""  # フォールバック用: 最後の有意なコンテンツを保持
    # 安全カウンター: tool_call_count に依存しない全体反復上限
    total_iterations = 0
    max_total_iterations = state.max_tool_calls + 10  # 継続やスキップ分の余裕

    while state.tool_call_count < state.max_tool_calls:
        total_iterations += 1
        if total_iterations > max_total_iterations:
            state.exit_reason = f"iteration_limit (全体反復上限 {max_total_iterations} に到達)"
            output_fn(f"\n[System] ReActループ終了: {state.exit_reason}\n", end="", flush=True)
            break
        # ========== Plan ノード ==========
        # デバッグモード用ターンカウンター
        if getattr(context, 'debug_mode', False):
            context.debug_turn = getattr(context, 'debug_turn', 0) + 1
        content, tool_calls = node_plan(
            context, state,
            show_thinking=show_thinking,
            max_tokens=max_tokens,
            output_fn=output_fn,
            system_msg_builder=system_msg_builder,
        )

        # 有意なコンテンツを追跡（空回答時のフォールバック用）
        if content:
            stripped = _strip_all_thinking(content).strip()
            if len(stripped) >= 50:
                last_substantive_content = stripped

        # 出力が途中で切れた場合 -> もう一度Planに戻る
        if state.phase == "NEEDS_CONTINUATION":
            state.continuation_count += 1
            # 途切れた分を累積バッファに結合（重複除去）— final_answer 復元用
            state.accumulated_content = _merge_continuation(state.accumulated_content, content or "")
            if state.continuation_count >= 8:
                # 継続が8回に達したら強制終了（無限継続防止）
                state.exit_reason = f"continuation_limit (継続生成が{state.continuation_count}回に到達)"
                output_fn(f"\n[System] ReActループ終了: {state.exit_reason}\n", end="", flush=True)
                final_answer = state.accumulated_content or content or ""
                _add_assistant_with_think(state, final_answer)
                break
            # 直前の出力の末尾をヒントに「続きだけ」を強く促す（再生成防止）
            tail_hint = ""
            if content:
                last_lines = [l for l in content.split('\n') if l.strip()]
                if last_lines:
                    tail_hint = f"\n前回の出力の末尾:\n```\n{last_lines[-1]}\n```\nこの行の直後から続けてください。"
            state.chat_history.add("user",
                f"出力が途中で切れました。**前回の末尾からの続きだけ**を出力してください。"
                f"**絶対に最初からやり直さないこと**（既に出力済みの部分は繰り返さない）。{tail_hint}")
            state.phase = "PLANNING"
            continue

        if tool_calls:
            # ========== ツール呼び出しがあり ==========
            # ツール未呼び出しカウンターをリセット
            state.no_tool_count = 0
            state.continuation_count = 0

            # ========== 半自動モード: ユーザー承認 ==========
            if interactive_fn:
                approved_calls, user_override = interactive_fn(tool_calls, content)
                if user_override:
                    # ユーザーが独自の指示を入力 → ツールをスキップして次のPlanへ
                    state.chat_history.add("assistant", content or "")
                    state.chat_history.add("user", user_override)
                    state.tool_call_count += 1
                    state.phase = "PLANNING"
                    continue
                if not approved_calls:
                    # ユーザーが全ツールを却下 → ループ終了
                    clean_content = _strip_all_thinking(content or "")
                    if _looks_like_action_promise(clean_content):
                        # 行動予告だけの content は履歴に残さず、クリーンな停止文にする
                        final_answer = "ツール実行がキャンセルされたため、ここで停止します。"
                    else:
                        final_answer = clean_content or content or "ツール実行がキャンセルされたため、ここで停止します。"
                    state.chat_history.add("assistant", final_answer)
                    state.exit_reason = "user_rejected (ユーザーがツール実行を却下)"
                    output_fn(f"[System] ReActループ終了: {state.exit_reason}\n", end="", flush=True)
                    break
                tool_calls = approved_calls

            # アシスタントメッセージを履歴に追加（content + tool_calls の両方）
            # Feature A: 直前の <think> を履歴に残して推論を引き継ぐ
            _add_assistant_with_think(state, content, tool_calls=tool_calls)

            # Turn間のコンテンツ追跡（類似性検知用）
            if content and hasattr(state, 'recent_contents'):
                clean_c = _strip_all_thinking(content or "")
                if clean_c:
                    state.recent_contents.append(clean_c[:500])

            # ========== ループ検知 + バッチ内重複排除 + ツール名バリデーション ==========
            loop_detected = False
            valid_calls = []
            skipped_count = 0
            seen_in_batch = set()  # 同一ターン内の重複排除用

            # 1ターンあたりのツール呼び出し上限（ハルシネーション対策）
            tool_calls_per_turn = min(len(tool_calls), 10)
            if len(tool_calls) > tool_calls_per_turn:
                output_fn(f"[System] ツール呼び出しが多すぎるため、最初の{tool_calls_per_turn}件のみ実行します。\n", end="", flush=True)
                # 超過分の tool_call_id にダミー結果を追加（メッセージシーケンス整合性維持）
                for tc_extra in tool_calls[tool_calls_per_turn:]:
                    state.chat_history.add(
                        role="tool",
                        content="[スキップ] 1ターンの上限に達したため未実行。",
                        tool_call_id=tc_extra["id"],
                    )
                tool_calls = tool_calls[:tool_calls_per_turn]

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = _safe_parse_args(func)
                current_action = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

                # ツール名バリデーション（存在しないツールの呼び出しを拒否）
                if tool_name not in TOOL_REGISTRY:
                    skipped_count += 1
                    output_fn(f"[System] ツール '{tool_name}' は存在しません — スキップします。\n", end="", flush=True)
                    state.chat_history.add(
                        role="tool",
                        content=f"Error: ツール '{tool_name}' は存在しません。利用可能なツール一覧を確認してください。",
                        tool_call_id=tc["id"],
                    )
                    continue

                # バッチ内重複排除（同じターンでの同一呼び出しは1回のみ実行）
                if current_action in seen_in_batch:
                    skipped_count += 1
                    state.chat_history.add(
                        role="tool",
                        content="[重複] 同じ呼び出しが既に実行されます。",
                        tool_call_id=tc["id"],
                    )
                    continue

                # クロスターン ループ検知
                if check_loop_detected(state.executed_actions, current_action):
                    state.loop_warn_count += 1
                    skipped_count += 1
                    if state.loop_warn_count >= 3:
                        state.exit_reason = f"loop_force_exit ({tool_name} の無限ループが3回検知)"
                        output_fn(f"[System] {tool_name} のループを3回検知。別のツールに切り替えます。\n", end="", flush=True)
                        state.chat_history.add(
                            role="tool",
                            content="Error: 無限ループ検知によりスキップされました。",
                            tool_call_id=tc["id"],
                        )
                        loop_detected = True
                        break
                    output_fn(f"[System] {tool_name} の連続ループを検知 — スキップします。\n", end="", flush=True)
                    state.chat_history.add(
                        role="tool",
                        content="[スキップ] 同じ呼び出しを検知しました。",
                        tool_call_id=tc["id"],
                    )
                else:
                    valid_calls.append(tc)
                    state.executed_actions.append(current_action)
                    seen_in_batch.add(current_action)

            # ========== Action ノード（有効な呼び出しは常に実行） ==========
            all_results = []
            if valid_calls:
                state.phase = "EXECUTING"
                readonly_calls, destructive_calls = classify_tools(valid_calls)

                if readonly_calls:
                    def _exec_tool_fn(tool_name, tool_args):
                        return node_action(context, state, {"tool_name": tool_name, **tool_args}, output_fn=output_fn)

                    parallel_results = execute_parallel(readonly_calls, _exec_tool_fn)
                    all_results.extend(parallel_results)
                    if len(readonly_calls) > 1:
                        output_fn(f"[System] {len(readonly_calls)}件の読み取りツールを並列実行しました\n", end="", flush=True)

                if destructive_calls:
                    for dc in destructive_calls:
                        func = dc.get("function", {})
                        tool_name = func.get("name", "")
                        tool_args = _safe_parse_args(func)
                        tool_result = node_action(context, state, {"tool_name": tool_name, **tool_args}, output_fn=output_fn)
                        all_results.append((dc, tool_result))

                # ========== Observe ノード + ツール結果をチャット履歴に追加 ==========
                state.phase = "OBSERVING"
                next_phase = "PLANNING"
                for tc, result in all_results:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    compressed = _compress_tool_result(tool_name, _safe_parse_args(func), result)

                    # 表示系ツール: 端末に結果を直接出力し、履歴にはANSI除去版を追加
                    if tool_name in DISPLAY_TOOLS:
                        output_fn(f"\n{result}\n", end="", flush=True)
                        compressed = _strip_ansi(compressed)

                    # role="tool" で結果を追加（tool_call_idで紐付け）
                    state.chat_history.add(
                        role="tool",
                        content=compressed,
                        tool_call_id=tc["id"],
                    )

                    # ステップ番号付きコンパクト表示
                    state.step_count += 1
                    args_str = _safe_parse_args(func)
                    compact_args = _format_tool_args(args_str)
                    is_destructive = tool_name in DESTRUCTIVE_TOOLS
                    if result.startswith("Error:"):
                        status = "✗"
                    elif is_destructive:
                        status = "✎"  # 書き込み系ツール
                    else:
                        status = "✓"  # 読み取り系ツール
                    output_fn(f"  [{state.step_count}] {tool_name}({compact_args}) → {len(compressed)}文字 {status}\n",
                              end="", flush=True)
                    next_phase = node_observe(state, tool_name, result, output_fn=output_fn)

                if next_phase == "DONE":
                    break

            # ========== ループ検知後の処理（実行済みの結果はchat_historyに反映済み） ==========
            if loop_detected:
                state.chat_history.add("user", "【警告】無限ループ検知。別のアプローチをとってください。")
                state.tool_call_count += 1
                state.phase = "PLANNING"
                continue

            if skipped_count == 0:
                state.loop_warn_count = 0

            if not valid_calls:
                output_fn(f"[System] {skipped_count}件スキップ。別のアプローチを促します。\n", end="", flush=True)
                state.chat_history.add("user",
                    "【システム】全ツール呼び出しがスキップされました。"
                    "別のツールを使用するか、最終回答を出力してください。")
                state.tool_call_count += 1
                state.phase = "PLANNING"
                continue

            state.phase = "PLANNING"
            continue

        else:
            # ========== ツール呼び出しなし ==========
            is_synthesizing = _detect_phase(state) == _SYNTHESIZING

            if not content or not content.strip():
                if last_substantive_content:
                    # フォールバック: 直前の有意なコンテンツを最終回答として使用
                    final_answer = last_substantive_content
                    state.exit_reason = f"fallback_response (ツール実行 {state.tool_call_count}回後、直前の回答を使用)"
                    state.chat_history.add("assistant", final_answer)
                    output_fn("\n[System] 空の応答でした。直前の回答を最終回答として使用します。\n", end="", flush=True)
                else:
                    state.exit_reason = f"empty_response (ツール実行 {state.tool_call_count}回目)"
                    output_fn(f"\n[System] ReActループ終了: {state.exit_reason}\n", end="", flush=True)
                break

            state.no_tool_count += 1

            # --- 反復コンテンツ検知 ---
            # 生成内容が反復パターンを含む場合、強制的にツール実行を促す
            clean_content = _strip_all_thinking(content)
            # SYNTHESIZING（詳細分析中）は正当な長文レポートの列挙パターンを反復と誤認しないよう免除
            is_repetitive = (_detect_repetitive_content(clean_content)
                             if state.continuation_count == 0 and not is_synthesizing
                             else False)

            # --- 単純質問は短くても最終回答として許可 ---
            # 「今のディレクトリは？」→ get_cwd() だけ完了、のように
            # 単純な情報取得質問に必要なツールが既に実行済みならガードレールを迂回する
            user_text = _get_last_user_text(state)
            if _is_simple_direct_answer_sufficient(user_text, clean_content, state):
                state.exit_reason = f"final_answer_simple_direct (ツール実行 {state.tool_call_count}回後)"
                final_answer = clean_content or content
                _add_assistant_with_think(state, content or final_answer)
                output_fn(f"[System] ReActループ終了: {state.exit_reason}\n", end="", flush=True)
                break

            # --- クロスターン コンテンツ類似性検知 ---
            # クールダウン中 or 完全性スコアが高い場合は類似度チェックを免除
            # SYNTHESIZING中はスキップ（分析のために類似表現が連続するのは正常）
            # ※ is_similar_to_previous / current_score は無条件参照されるため、
            #    SYNTHESIZING時にも安全なデフォルトで初期化しておく（UnboundLocalError防止）。
            is_similar_to_previous = False
            current_score = 100  # SYNTHESIZING時は高スコア扱いで類似度チェックを免除
            if not is_synthesizing:
                current_score = _answer_completeness_score(clean_content, state.tool_call_count)
            if state.guardrail_cooldown > 0:
                state.guardrail_cooldown -= 1
            elif current_score < 50 and state.continuation_count == 0:
                if hasattr(state, 'recent_contents') and state.recent_contents:
                    for prev in state.recent_contents[-2:]:
                        if _detect_content_similarity(clean_content, prev, threshold=0.75):
                            is_similar_to_previous = True
                            break

            # 反復または類似コンテンツの検知 -> 強制アクション
            if (is_repetitive or is_similar_to_previous) and state.no_tool_count < 4:
                output_fn(f"\n[System] 同一内容の反復出力を検知。強制的にツールを実行させます ({state.no_tool_count}/4)。\n", end="", flush=True)
                # 履歴を汚さないよう短縮版のみ追加
                short = clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
                state.chat_history.add("assistant", short)
                state.chat_history.add("user",
                    "【システム強制指示】あなたは同じ内容を繰り返し出力しています。即座に行動してください。\n"
                    "1. まず `list_directory` で現在のディレクトリ構造を確認してください。\n"
                    "2. または `get_cwd` で現在のパスを確認してください。\n"
                    "絶対に再検討しないでください。最初に思いついたツールを即実行してください。")
                state.tool_call_count += 1
                state.phase = "PLANNING"
                state.guardrail_cooldown = 2
                # 履歴に記録
                if hasattr(state, 'recent_contents'):
                    state.recent_contents.append(clean_content[:500])
                continue

            # --- 壊れたツール呼び出しの検知 ---
            # パーサー通過後のテキストにツール呼び出しのタグが残っている場合、フォーマットエラーとみなす
            has_partial_tool_call = "<tool_call" in content or "⬡" in content or "<|tool_call_start|>" in content  # [LFM専用]

            if has_partial_tool_call and state.no_tool_count < 3:
                output_fn(f"\n[System] ツール呼び出しのフォーマットエラーを検知しました。"
                          f"再試行します({state.no_tool_count}/3)。\n", end="", flush=True)
                state.chat_history.add("assistant", _strip_all_thinking(content))
                state.chat_history.add("user",
                    "【システム警告】ツールを呼び出そうとしてフォーマットが崩れています。"
                    "タグの閉じ忘れや、JSONの構文エラーがないか確認し、正しい形式でツールを呼び直してください。")
                state.tool_call_count += 1
                state.phase = "PLANNING"
                state.guardrail_cooldown = 2
                if hasattr(state, 'recent_contents'):
                    state.recent_contents.append(clean_content[:500])
                continue

            # --- 行動予告だけで終わっている応答の検知 ---
            # 「次に〜します」と書いているのに tool_call がない場合、
            # final_answer にせずツール呼び出しを促す
            # ※ tool_call_count > 0 の条件を外す: reset_for_new_turn() で
            #    tool_call_count が 0 にリセットされるため、2回目以降のターンで
            #    行動予告を検知できなくなる問題を回避する
            if (not is_synthesizing
                    and _looks_like_action_promise(clean_content)
                    and state.no_tool_count < 2):
                output_fn(
                    f"\n[System] 次の行動を宣言していますが tool_call がありません。"
                    f"実際にツールを呼び出させます ({state.no_tool_count}/2)。\n",
                    end="", flush=True)
                state.chat_history.add("assistant", clean_content[:300])
                state.chat_history.add("user",
                    "【システム指示】あなたは次に行う作業を宣言しましたが、"
                    "実際の tool_call がありません。"
                    "宣言だけで終わらず、対応するツールを今すぐ呼び出してください。"
                    "調査が完了している場合のみ、最終回答として"
                    "結論・根拠・対応案をまとめてください。")
                state.no_tool_count += 1
                state.phase = "PLANNING"
                state.guardrail_cooldown = 1
                if hasattr(state, 'recent_contents'):
                    state.recent_contents.append(clean_content[:500])
                continue

            # --- 短い回答の検知 (不完全な応答の可能性) ---
            # スコアベース判定: 完全性スコア < 50 のみガードレール発火
            # ※ 思考stripが未完（max_tokens到達等で<think>が閉じられなかった）の場合、
            #    clean_content に生の思考内容が混入しスコアが不正確になるため発火しない。
            #    このケースは継続/length判定の経路に委ねる。
            score_threshold = 30 if is_synthesizing else 50
            if (not code_mode
                    and state.tool_call_count > 0
                    and not _has_unclosed_thinking(content)
                    and _answer_completeness_score(clean_content, state.tool_call_count) < score_threshold
                    and state.no_tool_count < 3):
                output_fn(f"\n[System] 回答が短すぎます。引き続きツールを使用してください ({state.no_tool_count}/3)。\n",
                          end="", flush=True)
                state.chat_history.add("assistant", clean_content)
                state.chat_history.add("user",
                    "【システム指示】前回の回答が短すぎるため、不完全と判断されました。"
                    "引き続き必要なツールを使用して調査を続けてください。"
                    "調査が完了したら、十分な情報を含む最終回答を出力してください。")
                state.phase = "PLANNING"
                state.guardrail_cooldown = 2
                if hasattr(state, 'recent_contents'):
                    state.recent_contents.append(clean_content[:500])
                continue

            # ※ コードのみ応答検知（日本語文字数<5で強制再説明）は廃止:
            #    粗いプロキシで正確な簡潔回答を誘爆するため。ファイルエコー検出が必要なら
            #    「直前の tool_result との類似度」で検出する方針（_detect_content_similarity 転用）。

            # --- 通常の最終回答 ---
            state.exit_reason = f"final_answer (ツール実行 {state.tool_call_count}回後)"
            # length継続で累積した場合は結合して完全な回答を復元
            if state.accumulated_content:
                final_answer = _merge_continuation(state.accumulated_content, clean_content)
                hist_content = final_answer  # 継続時は結合済み全文（think含む可能性）を履歴へ
            else:
                final_answer = clean_content or content
                hist_content = content or final_answer  # 継続以外は生content(think付き)で引き継ぎ

            if getattr(context, 'phase', 'EXECUTING') == "PLANNING":
                with open(get_data_path("PLANNING.md"), "w", encoding="utf-8") as f:
                    f.write(final_answer)
                output_fn("[System] 計画を PLANNING.md に保存しました。\n", end="", flush=True)

                context.phase = "PLANNING_WAIT_OK"

            _add_assistant_with_think(state, hist_content)
            # /review モード: 設計/提案らしい final_answer を読取専用レビューアで批判。
            # observe-only・判定は _run_design_review 内で端末表示済み。ここでは履歴へ
            # 【システム: デザインレビュー】として注入（_get_last_user_text がスキップ＝非汚染。
            # 次ターンでエージェントが自己修正の余地を持つ）。
            if getattr(context, "review_mode", False):
                _dr_user = _get_last_user_text(state)
                if _is_design_proposal(final_answer, _dr_user, code_mode):
                    _dr_verdict = _run_design_review(context, final_answer, _dr_user, output_fn)
                    if _dr_verdict:
                        state.chat_history.add(
                            "user",
                            f"【システム: デザインレビュー】\n{_dr_verdict}\n"
                            f"※ 必要ならこの指摘を反映して設計を見直してください。"
                        )
            output_fn(f"[System] ReActループ終了: {state.exit_reason}\n", end="", flush=True)
            break

    if state.tool_call_count >= state.max_tool_calls:
        state.exit_reason = f"max_tool_calls_reached (連続実行上限 {state.max_tool_calls}回に到達)"
        output_fn(f"\n[System] ReActループ終了: {state.exit_reason}\n", end="", flush=True)

    return final_answer
