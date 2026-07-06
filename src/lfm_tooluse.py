"""AnythingPixie — LFM2.5 (Liquid Foundation Model) 専用 tool use モジュール。

engine.py の LFM ブランチから遅延 import される。本モジュールが欠落/エラーでも
engine.py は動作（LFM 機能のみ停止）。不要になったら本ファイルを削除 +
engine.py / main.py / llm_client.py の「# [LFM専用]」行を削除すれば完全除去できる
（Disposable Design）。

フェーズ2実測 (Sub PC LM Studio, lfm2.5-8b-a1b): native tools= パラメータだけで
構造化 tool_calls が返ることを確認したため（LM Studio がツール呼び出し特殊トークンを
内部変換する）、engine.py は現在 inject_lfm_tools を呼び出していない
（tools= への一本化。system プロンプトへの二重注入を避けるため）。
本モジュールの parse/inject 関数群は、llama-server 直結など tools= パラメータを
ネイティブ解釈しない環境向けの保険として温存している（system 注入 + テキストパースの
フォールバック経路。engine.py 側で再度 inject_lfm_tools を呼べば即座に有効化できる）。

LFM2.5 仕様 (Liquid 公式 docs):
- ツール定義: system プロンプトに JSON (List of tools: [{name,description,parameters}])
- ツール呼び出し: <|tool_call_start|>...<|tool_call_end|>。中身は Pythonic [func(kw=val)] または JSON
- チャット: ChatML (<|im_end|> EOS)。ツール結果は role="tool"

セキュリティ: モデル出力の Pythonic 文字列は ast.parse + 制限ウォークで解析
（eval/exec 不使用）。Attribute/Subscript/BinOp/Call-as-arg/裸Name を拒否。
"""

from __future__ import annotations

import ast
import json
import re

# 公開定数（engine.py の StreamFilter / partial 検出が参照。削除時にこれら参照箇所も削除）
LFM_TOOL_START = "<|tool_call_start|>"
LFM_TOOL_END = "<|tool_call_end|>"
LFM_ARTIFACT_TAGS = [LFM_TOOL_START, LFM_TOOL_END]

# 主正規表現（end トークンあり）+ 副正規表現（end 欠落 rescue。truncated 対策）
_LFM_TOOL_CALL_RE = re.compile(
    rf"{re.escape(LFM_TOOL_START)}(.*?){re.escape(LFM_TOOL_END)}",
    re.DOTALL,
)
_LFM_TOOL_CALL_RE_OPEN = re.compile(
    rf"{re.escape(LFM_TOOL_START)}(.*)$",
    re.DOTALL,
)
# ```json コードブロック内のツール配列（<|tool_call_start|> なしのフォールバック）
# LFM がトークンを使わず ```json [...] ``` で出力する場合の rescue
_LFM_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\s*```", re.DOTALL)

# _literal_value が受け入れる「安全なリテラル」AST ノード
_SAFE_LITERAL_NODES = (ast.Constant, ast.List, ast.Tuple, ast.Dict)


def _literal_value(node):
    """AST ノードから安全なリテラル値を抽出（Constant/List/Tuple/Dict のみ）。

    Name/Attribute/Call 等の副作用のあるノードは拒否（None を返す）。
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_value(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _literal_value(k): _literal_value(v)
            for k, v in zip(node.keys, node.values, strict=True)
            if k is not None
        }
    return None


def parse_pythonic_call(raw: str) -> list[tuple[str, dict]] | None:
    """Pythonic `[func(kw=val), func2(kw2=val2)]` を安全に解析（ast + 制限ウォーク）。

    eval/exec 不使用。Attribute/Subscript/BinOp/Call-as-arg/裸Name を拒否。
    SyntaxError/ValueError 含む全例外をキャッチし None を返す（呼出元で通常テキスト扱い）。
    位置引数 func("v") とキーワード引数 func(k="v") の両方に対応（位置は argN 仮名）。
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        tree = ast.parse(raw, mode="eval")
    except (SyntaxError, ValueError):
        return None
    expr = tree.body
    # List 式 [f1(), f2()] または単一 Call 式 f() を許容
    if isinstance(expr, ast.List):
        calls = expr.elts
    elif isinstance(expr, ast.Call):
        calls = [expr]
    else:
        return None

    results: list[tuple[str, dict]] = []
    for elt in calls:
        if not isinstance(elt, ast.Call):
            return None
        # func は Name のみ（Attribute obj.method / Subscript 拒否）
        if not isinstance(elt.func, ast.Name):
            return None
        name = elt.func.id
        args: dict = {}
        # 位置引数: 名前不明のため arg0, arg1 ... 仮名（値のみ安全に抽出）
        for i, a in enumerate(elt.args):
            if not isinstance(a, _SAFE_LITERAL_NODES):
                return None  # Name/Call 等の危険ノードを拒否
            args[f"arg{i}"] = _literal_value(a)
        # キーワード引数
        for kw in elt.keywords:
            if kw.arg is None:
                return None  # **kwargs は拒否
            if not isinstance(kw.value, _SAFE_LITERAL_NODES):
                return None
            args[kw.arg] = _literal_value(kw.value)
        results.append((name, args))
    return results if results else None


def _try_json(raw: str, start_idx: int) -> list[dict] | None:
    """JSON 形式の tool call を解析（{"name":..,"arguments":..} またはその list）。"""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    items = data if isinstance(data, list) else [data]
    result: list[dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        if not name:
            return None
        args = item.get("arguments", item.get("parameters", {}))
        if isinstance(args, str):
            # arguments が JSON 文字列なら妥当性確認、でなければ文字列化
            try:
                json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = json.dumps(args, ensure_ascii=False)
        else:
            args = json.dumps(args, ensure_ascii=False)
        result.append({
            "id": f"lfm_{name}_{start_idx + i}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    return result if result else None


def _parse_block(raw: str, start_idx: int) -> list[dict]:
    """1ブロック（start〜end の中身）を JSON → Pythonic の順で解析し OpenAI 形式へ。"""
    raw = raw.strip()
    if not raw:
        return []
    parsed = _try_json(raw, start_idx)  # 1) JSON 優先（解析確実）
    if parsed:
        return parsed
    calls = parse_pythonic_call(raw)  # 2) Pythonic フォールバック
    if not calls:
        return []
    return [
        {
            "id": f"lfm_{name}_{start_idx + i}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }
        for i, (name, args) in enumerate(calls)
    ]


def parse_lfm_tool_calls(content: str) -> tuple[str | None, list[dict] | None]:
    """LFM2.5 の <|tool_call_start|>...<|tool_call_end|> を抽出・解析。

    Returns:
        (cleaned, tool_calls)。tool_calls は OpenAI 形式 [{id,type,function:{name,arguments}}]。
        id は lfm_{name}_{n}（同一 assistant メッセージ内で一意）。
        解析失敗/呼び出しなし時は (content, None)。
    """
    if not content or (LFM_TOOL_START not in content and "```" not in content):
        return content, None

    tool_calls: list[dict] = []
    cleaned = content

    # 主: end トークンあり
    for match in list(_LFM_TOOL_CALL_RE.finditer(cleaned)):
        parsed = _parse_block(match.group(1), len(tool_calls))
        if parsed:
            tool_calls.extend(parsed)
            cleaned = cleaned.replace(match.group(0), "", 1)

    # 副: end 欠落（truncated）の rescue — 主で拾えなかった start が残っていれば
    if not tool_calls:
        for match in list(_LFM_TOOL_CALL_RE_OPEN.finditer(cleaned)):
            parsed = _parse_block(match.group(1), len(tool_calls))
            if parsed:
                tool_calls.extend(parsed)
                cleaned = cleaned.replace(match.group(0), "", 1)

    # 第3: <|tool_call_start|> なしでも ```json コードブロック内のツール配列を検出
    # （LFM がトークンを使わず ```json [...] ``` で出力する場合のフォールバック。
    #   name/arguments を持つ dict 配列のみ tool call とみなし、説明用 JSON 例は除外）
    if not tool_calls:
        for match in list(_LFM_JSON_BLOCK_RE.finditer(cleaned)):
            parsed = _try_json(match.group(1).strip(), len(tool_calls))
            if parsed:
                tool_calls.extend(parsed)
                cleaned = cleaned.replace(match.group(0), "", 1)

    if not tool_calls:
        return content, None

    cleaned = cleaned.strip()
    return (cleaned if cleaned else None), tool_calls


def inject_lfm_tools(system_text: str, openai_tools: list[dict]) -> str:
    """OpenAI ツールスキーマ → LFM 平たい形に変換し system プロンプトへ注入。

    LFM 推奨: system に `List of tools: [{name, description, parameters}, ...]` +
    JSON 形式での tool call 出力を指示（解析の確実性のため。Pythonic もフォールバック対応）。
    """
    lfm_tools = []
    for t in openai_tools:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        params = fn.get("parameters") or fn.get("input") or {}
        lfm_tools.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": params,
        })
    tools_str = json.dumps(lfm_tools, ensure_ascii=False)
    instruction = (
        "\n\n【利用可能なツール（LFM2.5 tool use）】\n"
        f"List of tools: {tools_str}\n"
        "ツールを呼ぶときは以下の形式で出力してください（JSON）:\n"
        f"{LFM_TOOL_START}{{\"name\": \"<ツール名>\", \"arguments\": {{<引数>}}}}{LFM_TOOL_END}\n"
        "※ 1ブロックに複数ツールをまとめても構いません。"
    )
    return system_text + instruction
