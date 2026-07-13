"""pixie_core — AnythingWithPixie(AWP) の ReAct エンジンを外部アプリから埋め込むための公開 API。

目的（PixieProject Phase 2 / pixie-core 切り出しの第一歩）:
    これまで組み込み側（CodeWithPixie など）は `engine` / `main` / `registry` / `config` を
    個別に import して AppContext を手組みしていた。それは AWP の内部実装への散在依存であり、
    AWP を更新すると組み込み側が静かに壊れる（監査 Fable の Major 指摘）。

    本モジュールは AWP 内部への依存を**この1枚に集約した安定な公開境界**である。組み込み側は
    `import pixie_core` だけを行い、内部モジュールには直接触れない。将来 engine 等を物理的に
    別パッケージへ移動しても、この API シグネチャを保てば組み込み側は無改修で済む。

    UI 非依存: 出力は output_fn コールバック、承認は interactive_fn コールバックで外部注入する
    （print/input には一切依存しない）。stdout の再設定やスレッド化・SSE 変換は組み込み側の責務。

公開 API:
    API_VERSION                         — 互換性チェック用の文字列。
    CancelTurn                          — 協調キャンセル用例外（output_fn / interactive_fn から送出）。
    READONLY_TOOLS / DESTRUCTIVE_TOOLS  — ツール分類（承認要否の判定に使う）。
    create_engine(server, workspace)    — Engine を構築（AppContext/AgentState/ツール登録/cwd/状態注入）。
    class Engine                        — .run_turn(user_text, *, output_fn, interactive_fn)。

注意: このモジュールは AWP の `src` をパスに含めた状態で import すること（AWP と同じフラット
import 前提: `from engine import ...`）。組み込み側は sys.path に AWP/src を前置してから
`import pixie_core` する。
"""
from __future__ import annotations

import os

# --- AWP 内部（この境界の内側でのみ import する） ---
from engine import run_graph, build_system_text
from main import AppContext
from state import AgentState
from registry import set_state_board, TOOL_REGISTRY
from llm_client import LMStudioBackend
from config import DESTRUCTIVE_TOOLS, READONLY_TOOLS
import paths

# ツール登録の副作用（@register_tool）。import するだけで TOOL_REGISTRY が満たされる。
import tools as _tools          # noqa: F401
import code_tool as _code_tool  # noqa: F401

#: 公開 API のバージョン。組み込み側は起動時にこれを検証して不整合を早期検知できる。
API_VERSION = "1.0"

__all__ = [
    "API_VERSION", "CancelTurn", "READONLY_TOOLS", "DESTRUCTIVE_TOOLS",
    "create_engine", "Engine", "tool_count",
]


class CancelTurn(Exception):
    """協調キャンセル。組み込み側の output_fn / interactive_fn から送出するとターンを打ち切る。

    run_graph は同期ループなので、外部からの中断は「コールバック内で例外を送出して脱出する」
    のが唯一安全な方法（スレッドの強制終了はできない）。承認コールバックで空承認を返す経路と
    併用すると、LLM 生成中／承認待ちのどちらでも確実に止められる。
    """


def tool_count() -> int:
    """登録済みツール数（疎通スモークにも使える）。"""
    return len(TOOL_REGISTRY)


class Engine:
    """1セッション分の埋め込みエンジン。AppContext と AgentState を保持し run_graph を回す。

    UI 非依存。マルチセッションは Phase 2 の後続課題（registry の state_board が現状プロセス
    グローバルなため、同一プロセスでの複数 Engine 並行実行は非対応）。
    """

    def __init__(self, context: AppContext, state: AgentState):
        self.context = context
        self.state = state

    @property
    def model_name(self) -> str:
        return getattr(self.context, "llm_model_name", "") or ""

    @property
    def tool_count(self) -> int:
        return len(TOOL_REGISTRY)

    def run_turn(self, user_text: str, *, output_fn, interactive_fn=None,
                 show_thinking: bool = False) -> str:
        """1ユーザーターンを実行して最終回答テキストを返す。

        ターンシーケンス（監査 F1: 忘れるとカウンタ持ち越しで2ターン目以降が壊れる）:
            reset_for_new_turn() -> chat_history.add(user) -> run_graph()

        Args:
            output_fn: run_graph の出力コールバック output_fn(text, end=, flush=)。
                       CancelTurn を送出すると即座に中断できる。
            interactive_fn: ツール実行直前の承認コールバック
                            (tool_calls, content) -> (approved_calls, user_override)。
                            None なら完全自律。
            show_thinking: True で思考ブロックもストリームする（既定 False = 本文のみ）。
        """
        self.state.reset_for_new_turn()
        self.state.chat_history.add("user", user_text)
        return run_graph(
            context=self.context,
            state=self.state,
            show_thinking=show_thinking,
            system_msg_builder=build_system_text,
            interactive_fn=interactive_fn,
            output_fn=output_fn,
        )


def create_engine(server: dict, workspace: str) -> Engine:
    """埋め込み用 Engine を構築する。

    行うこと:
        - cwd を workspace に固定し、AWP の paths を初期化（cwd 依存ツール/永続状態の作業対象）。
        - AppContext を実クラスで生成し、LM Studio バックエンドを接続。
        - AgentState を生成し、registry にプロセスグローバル注入（現状の単一セッション前提）。

    Args:
        server: {"base_url", "api_key"?, "model"?} 形式（AWP の config.json servers[] と同形式）。
        workspace: エージェントの作業対象＝cwd＝サンドボックスのルート（絶対パス推奨）。
    """
    os.chdir(str(workspace))
    paths.set_project_root(os.getcwd())

    ctx = AppContext()
    ctx.llm = LMStudioBackend(
        server["base_url"],
        server.get("api_key", "lm-studio"),
        server.get("model", "local-model"),
    )
    # サンプリングプロファイルはモデル名の部分一致で選ばれる（空だと常に default）。
    ctx.llm_model_name = server.get("model", "") or ""

    state = AgentState()
    set_state_board(state.state_board)  # プロセスグローバル注入（Phase 2 後続でセッション別化予定）

    return Engine(ctx, state)
