"""ツールレジストリと共有グローバル状態のインフラモジュール。

tools.py / code_tool.py / engine.py が共通して参照する、他に依存を持たない
純粋なグローバル状態を集約する。本モジュールへの切り出しにより、
tools ↔ code_tool の循環 import（かつて tools.py 末尾にあった ``import code_tool``）
を解消し、code_tool の7ツールをトップレベルで安全にロードできるようにする。

依存: なし（標準ライブラリのみ）。他モジュールの最下層。
"""

# ============================
# ツールレジストリ
# ============================
TOOL_REGISTRY = {}


def register_tool(
    name: str, description: str, schema: dict, category: str = "core", prompt_desc: str = None, pack: str | None = None
):
    """ツール登録デコレータ。

    Args:
        name: ツール名（LLMが指定する識別子）
        description: ツールの説明文（LLMのFunction Callingスキーマ用）
        schema: 引数スキーマ（OpenAI Function Calling形式）
        category: "core" または "extended"（拡張ツールは inspect_tool で詳細取得）
        prompt_desc: プロンプトに表示する1行サマリー（Noneならdescriptionを使用）
        pack: 所属ツールパック名（例: "manga"）。None（既定）はコアツールを意味し、
              常時 get_active_tool_names() の対象になる（従来動作と完全互換）。
              pack 指定時は、そのパックが active_packs に含まれる時のみ対象になる。
    """

    def decorator(func):
        TOOL_REGISTRY[name] = {
            "func": func,
            "description": description,
            "schema": schema,
            "category": category,
            "prompt_desc": prompt_desc or description,
            "pack": pack,
        }
        return func

    return decorator


def get_active_tool_names(active_packs: set = None) -> frozenset:
    """コアツール（pack未指定）+ active_packs に含まれるパックのツール名を返す。

    active_packs が空/None の場合はコアツールのみ（パック未有効時の従来動作と同一集合）。
    戻り値は frozenset（呼び出し側で sorted() して決定論的な順序にすること）。
    """
    active = active_packs or set()
    return frozenset(
        name for name, entry in TOOL_REGISTRY.items() if entry.get("pack") is None or entry.get("pack") in active
    )


def get_active_tool_names_ordered(active_packs: set = None) -> list:
    """get_active_tool_names の順序保証版（TOOL_REGISTRY の登録順=import順を維持する）。

    active_packs が空/None の時（パック未有効セッション）は、他プロセス内で
    /pack により一時的にロードされたパックモジュールが TOOL_REGISTRY に残っていても、
    それらを除いた「コアツールのみ・登録順」の一覧を返す。これにより
    ``registry_to_openai_tools(None)`` の「フィルタなし＝TOOL_REGISTRY全件」という
    従来のショートカット的意味に依存せず、パック未使用時のツール一覧・並び順を
    実装前と完全に一致させられる（sorted() での並び替えは行わない）。
    """
    active = active_packs or set()
    return [name for name, entry in TOOL_REGISTRY.items() if entry.get("pack") is None or entry.get("pack") in active]


# ============================
# ステートボード参照（外部から注入: main.py の set_state_board 経由）
# ============================
_state_board = None


def set_state_board(sb):
    """外部からステートボードインスタンスを注入する。"""
    global _state_board
    _state_board = sb


# ============================
# 動的ツール結果上限（engine.node_plan がコンテキスト使用率に応じて設定）
# ============================
_dynamic_max_chars = None  # None のとき TOOL_RESULT_MAX_CHARS にフォールバック


def set_tool_result_max_chars(n: int):
    """engine.node_plan が呼ぶ。コンテキスト使用率から逆算した1件あたりの文字上限を設定。

    並列ツール実行中は読み取り専用（node_plan が単一スレッドで1回だけ設定し、
    その直後のAction実行が並列で参照する）ためスレッドセーフ。
    """
    global _dynamic_max_chars
    _dynamic_max_chars = n
