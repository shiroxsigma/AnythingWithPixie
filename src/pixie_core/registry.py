"""ツールレジストリと共有グローバル状態のインフラモジュール。

tools.py / code_tool.py / engine.py が共通して参照する、他に依存を持たない
純粋なグローバル状態を集約する。本モジュールへの切り出しにより、
tools ↔ code_tool の循環 import（かつて tools.py 末尾にあった ``import code_tool``）
を解消し、code_tool の7ツールをトップレベルで安全にロードできるようにする。

依存: なし（標準ライブラリのみ）。他モジュールの最下層。
"""

import contextvars

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
# ステートボード参照 / 動的ツール結果上限（コンテキスト別・マルチセッション対応）
# ============================
# かつてはプロセスグローバル変数だったが、1プロセスで複数セッション（会話）を並行実行する
# 埋め込み利用（CodeWithPixie 等）に対応するため ContextVar 化した。engine/subagent/tools は
# 従来どおり `registry._state_board` / `registry._dynamic_max_chars` として参照するが、
# モジュールの __getattr__（PEP 562）が現在の実行コンテキストの値へ解決する。
# 並列ツール実行（engine.execute_parallel）は contextvars.copy_context() で
# コンテキストをワーカースレッドへ伝播するため、並列 Action からも同一ターンの値が見える。
# CLI（単一スレッドで全ターンを実行）では従来と完全に同じ挙動になる。
_state_board_var: contextvars.ContextVar = contextvars.ContextVar("pixie_state_board", default=None)
_dynamic_max_chars_var: contextvars.ContextVar = contextvars.ContextVar("pixie_dynamic_max_chars", default=None)


def set_state_board(sb):
    """現在の実行コンテキストにステートボードインスタンスを束縛する。

    埋め込み時は各ターンを実行するスレッド内（ターン開始時）で呼ぶこと。CLI では起動時に
    一度呼べば、以降のターンも同一スレッド＝同一コンテキストで参照できる（従来動作と同一）。
    """
    _state_board_var.set(sb)


def set_tool_result_max_chars(n: int):
    """engine.node_plan が呼ぶ。コンテキスト使用率から逆算した1件あたりの文字上限を設定。

    ContextVar なので、並列ツール実行（copy_context 伝播先）でも同一ターンの値が見える一方、
    別セッションのターンとは干渉しない。
    """
    _dynamic_max_chars_var.set(n)


def __getattr__(name):
    """PEP 562: `registry._state_board` / `registry._dynamic_max_chars` を現在の実行
    コンテキストの値へ解決する。既存の全参照箇所（engine/subagent/tools）を無改修のまま
    コンテキスト別（＝セッション別）にするための仕掛け。"""
    if name == "_state_board":
        return _state_board_var.get()
    if name == "_dynamic_max_chars":
        return _dynamic_max_chars_var.get()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
