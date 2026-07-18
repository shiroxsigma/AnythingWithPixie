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

フェーズ3実測: <|tool_call_start|> も ```コードブロックも使わず、"I'll read the file...
[read_file(path='x.py')]" のように裸の Pythonic 呼び出しが content に漏れるケースを確認。
parse_lfm_tool_calls に known_tools（実在ツール名集合）を渡すと、行全体がその呼び出し式
だけで構成され、かつ関数名が known_tools に含まれる場合に限り rescue する（第4段）。
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

# 裸 Pythonic ツール呼び出し行の候補抽出（特殊トークンも```も無い content 中の rescue 用）。
# 行「全体」が `[func(...), func2(...)]` または `func(...)` だけで構成される場合のみ候補にする
# （文章中に埋め込まれた `print("hello")` のようなコード例を誤検知しないため）。
_BARE_LINE_CANDIDATE_RE = re.compile(
    r"^\[.*\]$|^[A-Za-z_][A-Za-z0-9_]*\(.*\)$"
)

# 裸 `tool_name: {json引数}` 行の候補抽出（フェーズ3 eval 05 で実測されたもう1つの漏れ形式:
# `write_file: {"path": "math_utils.py", "content": "..."}` が content にテキストとして漏れる）。
# こちらも行全体一致 + known_tools フィルタ + JSON 引数の dict パース成功時のみ採用する。
_BARE_JSON_LINE_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\{.*\})\s*$"
)

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


def _rescue_bare_pythonic_lines(
    text: str, known_tools: set[str], start_idx: int = 0
) -> tuple[str, list[dict]]:
    """特殊トークンも```も無い content 中の「裸の Pythonic ツール呼び出し行」を rescue する。

    フェーズ3 eval で確認された失敗パターン（例: "I'll read the file...\\n\\n
    [read_file(path='x.py')]" のように、<|tool_call_start|> も ```json ブロックも
    使わず裸の Pythonic 呼び出しが content に漏れる）への対処。

    対応する漏れ形式（いずれも行全体一致のみ・部分一致は不採用）:
    1. Pythonic: `[read_file(path='x.py')]` または `read_file(path='x.py')`
    2. name:json: `write_file: {"path": "x.py", "content": "..."}`（eval 05 実測）

    誤検知対策（最重要）:
    - 行「全体」がその呼び出し式（List または単一Call、あるいは name: {json}）だけで
      構成される場合に限り候補にする（文章中に埋め込まれた `print("hello")` のような
      コード例は行全体一致しないため無視される）。
    - 候補行は AST 安全パース（parse_pythonic_call）または json.loads で解析し、
      パース成功しても**関数名が known_tools に含まれる場合のみ**採用する
      （ハルシネーションした架空の関数名やただのコード例を誤ってツール呼び出し化しない）。
    - 1行でも known_tools 不一致の呼び出しを含む場合、その行は丸ごと不採用（部分採用しない）。

    Returns:
        (呼び出し行を除去した残りテキスト, 追加された tool_calls のリスト)。
        該当行が無ければ (text, []) を返す。
    """
    tool_calls: list[dict] = []
    kept_lines: list[str] = []
    for line in text.split("\n"):
        candidate = line.strip()
        if not candidate:
            kept_lines.append(line)
            continue

        # 形式1: Pythonic 呼び出し行
        if _BARE_LINE_CANDIDATE_RE.match(candidate):
            calls = parse_pythonic_call(candidate)
            if calls and all(name in known_tools for name, _ in calls):
                for name, args in calls:
                    tool_calls.append({
                        "id": f"lfm_{name}_{start_idx + len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })
                continue  # 採用: この行は残りテキストから除去

        # 形式2: `tool_name: {json引数}` 行
        m = _BARE_JSON_LINE_RE.match(candidate)
        if m and m.group(1) in known_tools:
            try:
                args = json.loads(m.group(2))
            except (json.JSONDecodeError, ValueError):
                args = None
            if isinstance(args, dict):
                name = m.group(1)
                tool_calls.append({
                    "id": f"lfm_{name}_{start_idx + len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                })
                continue  # 採用: この行は残りテキストから除去

        kept_lines.append(line)
    return "\n".join(kept_lines), tool_calls


# --- プロトコル JSON レスキュー（別エージェント形式の生 JSON からの tool call 抽出） -------
# 小型モデルが学習時に混入した「別エージェント形式」のプロトコル JSON をツール呼び出しとして
# テキスト出力する崩壊モードがある（LFM2.5 実測）。native tool_calls も <|tool_call_start|> も
# ```json フェンスも使わず、トップレベルの JSON オブジェクト内に tool 呼び出し配列を埋める:
#   {"plan": "...", "commands": [{"name": "grep_search", "arguments": {...}}]}
#   {"tool_calls": [{"tool_name": "read_file", "arguments": {...}}]}
#   {"steps": [{"thought": "...", "tool_calls": [{...}]}]}
# これらを known_tools フィルタつきで rescue する（誤検知対策は既存段と同じく known_tools 一致）。
_PROTOCOL_CONTAINER_KEYS = ("tool_calls", "commands")  # 呼び出し配列を持つキー
_PROTOCOL_NEST_KEYS = ("steps",)  # 各要素がさらに container キーを持ちうるネストキー
_PROTOCOL_MAX_CALLS = 10  # 巨大 JSON からの過剰抽出を防ぐ上限（engine の1ターン上限と同じ）

# JSON として無効な「孤立バックスラッシュ」を補修するための正規表現。
# Windows パス（"D:\Workspace"）や正規表現（"\["）が引数に入ると strict JSON が壊れるため、
# 有効なエスケープ（\" \\ \/ \b \f \n \r \t \uXXXX）以外のバックスラッシュのみ二重化する。
_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?![\"\\/bfnrtu])')


def _iter_json_objects(text: str):
    """text 中のトップレベル {...} を balanced-brace 走査で列挙（文字列リテラル/エスケープ考慮）。

    連結された複数オブジェクト（}{）や、引数値に含まれる入れ子 {} を正しく分離する。
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def _loads_lenient(raw: str):
    """json.loads を試し、失敗時のみ孤立バックスラッシュを補修して再試行（安全な最小補修）。"""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return json.loads(_INVALID_JSON_ESCAPE_RE.sub(r"\\\\", raw))
    except (json.JSONDecodeError, ValueError):
        return None


def _calls_from_items(items, known_tools: set[str], start_idx: int) -> list[dict]:
    """{name|tool_name, arguments|parameters} の list を OpenAI 形式 tool_calls へ（known_tools 一致のみ）。"""
    out: list[dict] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool_name")
        if not name or name not in known_tools:
            continue
        args = item.get("arguments", item.get("parameters", {}))
        if isinstance(args, str):
            try:
                json.loads(args)
                args_str = args
            except (json.JSONDecodeError, ValueError):
                args_str = json.dumps(args, ensure_ascii=False)
        elif isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)
        else:
            args_str = "{}"
        out.append({
            "id": f"lfm_{name}_{start_idx + len(out)}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return out


def _rescue_protocol_json(text: str, known_tools: set[str]) -> list[dict]:
    """別エージェント形式のプロトコル JSON オブジェクトからツール呼び出しを抽出する。

    重複（同一 name+arguments）は除去し、最大 _PROTOCOL_MAX_CALLS 件で打ち切る。
    """
    tool_calls: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(calls: list[dict]):
        for c in calls:
            key = (c["function"]["name"], c["function"]["arguments"])
            if key in seen:
                continue
            seen.add(key)
            c["id"] = f"lfm_{c['function']['name']}_{len(tool_calls)}"
            tool_calls.append(c)

    for raw in _iter_json_objects(text):
        if len(tool_calls) >= _PROTOCOL_MAX_CALLS:
            break
        obj = _loads_lenient(raw)
        if not isinstance(obj, dict):
            continue
        for key in _PROTOCOL_CONTAINER_KEYS:
            _add(_calls_from_items(obj.get(key), known_tools, len(tool_calls)))
        for nest in _PROTOCOL_NEST_KEYS:
            steps = obj.get(nest)
            if isinstance(steps, list):
                for step in steps:
                    if isinstance(step, dict):
                        for key in _PROTOCOL_CONTAINER_KEYS:
                            _add(_calls_from_items(step.get(key), known_tools, len(tool_calls)))
    return tool_calls[:_PROTOCOL_MAX_CALLS]


def parse_lfm_tool_calls(
    content: str, known_tools: set[str] | None = None
) -> tuple[str | None, list[dict] | None]:
    """LFM2.5 の <|tool_call_start|>...<|tool_call_end|> を抽出・解析。

    Args:
        content: モデル生成テキスト。
        known_tools: 実在するツール名の集合。指定時のみ、特殊トークンも```も無い
            「裸の Pythonic ツール呼び出し行」の rescue（第4段）を有効にする
            （誤検知対策のため既定 None = 無効・従来動作のまま）。

    Returns:
        (cleaned, tool_calls)。tool_calls は OpenAI 形式 [{id,type,function:{name,arguments}}]。
        id は lfm_{name}_{n}（同一 assistant メッセージ内で一意）。
        解析失敗/呼び出しなし時は (content, None)。
    """
    if not content:
        return content, None
    has_markers = LFM_TOOL_START in content or "```" in content
    if not has_markers and not known_tools:
        return content, None

    tool_calls: list[dict] = []
    cleaned = content

    if has_markers:
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

    # 第4: 裸 Pythonic レスキュー（特殊トークンも```も無い content 中の裸呼び出し行）。
    # known_tools 指定時のみ有効（誤検知対策の要。呼び出し元の engine.py が
    # TOOL_REGISTRY のキー集合を渡す）。
    if not tool_calls and known_tools:
        cleaned, bare_calls = _rescue_bare_pythonic_lines(cleaned, known_tools)
        tool_calls.extend(bare_calls)

    # 第5: プロトコル JSON レスキュー（別エージェント形式の {..."commands":[...]...} オブジェクト）。
    # 全文がプロトコル JSON の scaffolding（plan/analysis 等）のため、rescue 成功時は
    # 残テキストを表示しない（cleaned=None）。known_tools 指定時のみ有効。
    if not tool_calls and known_tools:
        proto_calls = _rescue_protocol_json(content, known_tools)
        if proto_calls:
            return None, proto_calls

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
