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


def register_tool(name: str, description: str, schema: dict, category: str = "core", prompt_desc: str = None):
    """ツール登録デコレータ。

    Args:
        name: ツール名（LLMが指定する識別子）
        description: ツールの説明文（LLMのFunction Callingスキーマ用）
        schema: 引数スキーマ（OpenAI Function Calling形式）
        category: "core" または "extended"（拡張ツールは inspect_tool で詳細取得）
        prompt_desc: プロンプトに表示する1行サマリー（Noneならdescriptionを使用）
    """

    def decorator(func):
        TOOL_REGISTRY[name] = {
            "func": func,
            "description": description,
            "schema": schema,
            "category": category,
            "prompt_desc": prompt_desc or description,
        }
        return func

    return decorator


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
