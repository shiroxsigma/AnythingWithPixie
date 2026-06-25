"""エンジン共通の純粋関数群（LLM/状態に依存しない解析ユーティリティ）。

engine.py と subagent.py の両方から参照される関数を集約し、循環 import を解消する。
解析純粋関数が中心。default_output_fn（CLI 出力）等の軽量ヘルパ・定数も含む。
estimate_tokens は llm オブジェクトのメソッドを呼ぶが引数で受け取るためグローバル状態に
依存しない。test_engine_guardrails.py で保護済み。

依存: 標準ライブラリのみ（json, re, collections）。
"""

import json
import re
from collections import Counter

# =====================================================
# ネイティブツール呼び出しパーサー（GGUFモデル用）
# =====================================================

# Qwen3.5系のツール呼び出しフォーマット（2種類）:
# 1. ASCII形式:   <tool_call\n{json}\n</tool_call (>は省略可)
# 2. 特殊トークン: (U+2B21)\n{json}\n(U+2B22)
_NATIVE_TOOL_CALL_RES = [
    re.compile(r'<tool_call\s*\n(.*?)\n\s*</tool_call\s*>?', re.DOTALL),
    re.compile(r'⬡\s*\n(.*?)\n\s*⬢', re.DOTALL),
]


def safe_parse_args(func: dict) -> dict:
    """ツール呼び出しの arguments を安全にパースする。

    空文字列・空白のみ・不正JSON の場合も空 dict を返し、
    JSONDecodeError が main.py まで伝播するのを防ぐ。
    """
    raw = func.get("arguments", "")
    if not raw or not raw.strip():
        return {}
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def detect_repetitive_content(content: str, min_repeats: int = 3) -> bool:
    """LLM出力内の反復パターンを検出する（無限思考ループ検知）。

    ローカルLLMが「Wait... Actually...」のような迷いを繰り返す
    思考ループに陥った場合に True を返す。

    Args:
        content: 検査対象のテキスト
        min_repeats: 反復回数の閾値

    Returns:
        反復パターンが検出された場合 True
    """
    if not content or len(content) < 100:
        return False

    lines = content.split('\n')
    non_empty = [l.strip() for l in lines if l.strip()]

    if len(non_empty) < min_repeats:
        return False

    # 1. 同一行の反復（意味のある長さの行のみ）
    line_counts = Counter(non_empty)
    for line, count in line_counts.items():
        if count >= min_repeats and len(line) > 5:
            return True

    # 2. 「Wait/Actually/Let's go」パターンの反復
    hesitation_count = 0
    hesitation_words = ['Wait', 'Actually', 'Let\'s go', 'Okay, let', 'Final Decision',
                        'Final Choice', 'Final Final', 'One more', 'No, let',
                        'Ready', 'Let\'y go', 'Final-final']
    for word in hesitation_words:
        hesitation_count += content.lower().count(word.lower())
    if hesitation_count >= min_repeats * 2:
        return True

    # 3. 同じコマンド/コードブロックの頻出
    command_pattern = re.findall(r'`([^`]+)`', content)
    cmd_counts = Counter(command_pattern)
    for cmd, count in cmd_counts.items():
        if count >= min_repeats * 2 and len(cmd) > 2:
            return True

    # 4. "---" セパレータの過剰な出現
    sep_count = content.count('---')
    if sep_count >= min_repeats * 4:
        return True

    # 5. [Command] マーカーの反復
    command_markers = content.count('[Command]')
    if command_markers >= min_repeats:
        return True

    return False


def strip_all_thinking(text: str) -> str:
    """全てのthinkingブロックを除去する（履歴汚染防止用）"""
    # Qwen/DeepSeek等の <think> タグ形式を削除
    cleaned = re.sub(r'<think[^>]*>?.*?</think[^>]*>?', '', text, flags=re.DOTALL)
    # 未閉じの思考ブロックも除去（max_tokens到達時など）
    cleaned = re.sub(r'<think[^>]*>.*$', '', cleaned, flags=re.DOTALL)

    # 【修正】絵文字形式の削除（🧠...）を削除。
    # これにより AI が出力した「🧠理由💬」が履歴に残るようになります。
    return cleaned.strip()


def estimate_tokens(llm, text: str) -> int:
    """テキストのトークン数を概算または正確に取得。

    estimate_token_count (正確 or 正直な概算) を優先し、なければ tokenize + len、
    最後に文字数概算でフォールバックする。
    """
    if hasattr(llm, 'estimate_token_count'):
        try:
            return llm.estimate_token_count(text)
        except Exception:
            pass
    if hasattr(llm, 'tokenize'):
        try:
            tokens = llm.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception:
            pass
    return len(text) // 3


def parse_native_tool_calls(content: str) -> tuple[str | None, list[dict] | None]:
    """LLMのテキスト出力からネイティブツール呼び出しを抽出する。

    GGUFモデル（Qwen3.5等）は Function Calling に対応したチャットテンプレートを
    使用しているが、llama-cpp-python がツール呼び出しを構造化して返さない場合がある。
    この関数はテキストから <tool_call...> ブロックを正規表現で抽出し、
    OpenAI tool_calls 形式に変換する。

    Args:
        content: LLMの出力テキスト

    Returns:
        (cleaned_content, tool_calls)
        - cleaned_content: ツール呼び出しブロックを除去したテキスト（空ならNone）
        - tool_calls: OpenAI形式のツール呼び出しリスト（見つからなければNone）
    """
    if not content:
        return content, None

    tool_calls = []
    cleaned = content

    for pattern in _NATIVE_TOOL_CALL_RES:
        for match in pattern.finditer(cleaned):
            json_str = match.group(1).strip()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            name = data.get("name", "")
            arguments = data.get("arguments", {})

            if not name:
                continue

            tool_calls.append({
                "id": f"native_{name}_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments),
                },
            })

            # マッチしたブロックをテキストから除去
            cleaned = cleaned.replace(match.group(0), "", 1)

    if not tool_calls:
        return content, None

    # 除去後のテキストを整形（末尾の空白を削除）
    cleaned = cleaned.strip()
    return cleaned if cleaned else None, tool_calls


def accumulate_tool_calls(stream_chunks: list[dict]) -> tuple[str | None, list[dict] | None]:
    """ストリーミングチャンクからテキストとtool_callsを蓄積・抽出する。

    Args:
        stream_chunks: LLMから受け取った生チャンクのリスト

    Returns:
        (content_text, tool_calls)
        - content_text: 蓄積されたテキスト（None if no content）
        - tool_calls: OpenAI形式のツール呼び出しリスト（None if no tool_calls）
    """
    content_parts = []
    tool_calls_map = {}  # index -> {id, type, function: {name, arguments}}

    for chunk in stream_chunks:
        choice = chunk.get("choices", [{}])[0] if "choices" in chunk else {}
        delta = choice.get("delta", {})

        if delta.get("content"):
            content_parts.append(delta["content"])

        if delta.get("tool_calls"):
            for tc_delta in delta["tool_calls"]:
                idx = tc_delta["index"]
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": tc_delta.get("id", f"call_{idx}"),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_calls_map[idx]
                if tc_delta.get("id"):
                    entry["id"] = tc_delta["id"]
                func = tc_delta.get("function", {})
                if func.get("name"):
                    entry["function"]["name"] += func["name"]
                if func.get("arguments"):
                    entry["function"]["arguments"] += func["arguments"]

    content = "".join(content_parts) if content_parts else None
    tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())] if tool_calls_map else None

    return content, tool_calls


# =====================================================
# 軽量ヘルパ・定数（engine 本体と subagent の共用）
# =====================================================

def is_simple_question(user_text: str) -> bool:
    """ユーザー入力が単純な情報取得質問かを判定する（answer不要・軽量）。

    _is_simple_direct_answer_sufficient は answer も参照するが、思考深度モードの
    判定時点ではまだ answer がないため、user_text 単体で判定する軽量版。
    単純質問は shallow のまま即実行させる（「簡単なのは残す」方針）。
    """
    if not user_text:
        return False
    q = user_text.strip().lower()
    simple_markers = [
        "今のディレクトリ", "現在のディレクトリ", "カレントディレクトリ",
        "作業ディレクトリ", "cwd", "pwd", "どこのディレクトリ",
        "何が入って", "なにが入って", "ファイル一覧", "ファイル構成",
        "ディレクトリの中", "ディレクトリの中身", "何がある", "フォルダの中",
        "内容を教えて", "中身を教えて", "読んで", "見せて",
    ]
    return any(m in q for m in simple_markers)


def default_output_fn(text, end="", flush=True):
    """デフォルトの出力関数（CLI用: print）"""
    print(text, end=end, flush=flush)


# ファイル編集を行うツール名のセット（review/verify のトリガ判定用）
FILE_EDIT_TOOLS = {"write_file", "replace_lines", "search_and_replace", "append_to_file", "write_sections"}
