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
# 注: AppContext は CLI 層 main.py にあり（パッケージ外）、CLI スタックを巻き込むため
# トップレベルでは import せず create_engine() 内で遅延 import する。
from engine import run_graph, build_system_text
from state import AgentState
from registry import set_state_board, TOOL_REGISTRY
from llm_client import LMStudioBackend
from config import DESTRUCTIVE_TOOLS, READONLY_TOOLS
import paths

# ツール登録の副作用（@register_tool）。import するだけで TOOL_REGISTRY が満たされる。
import tools as _tools          # noqa: F401
import code_tool as _code_tool  # noqa: F401

#: 公開 API のバージョン。組み込み側は起動時にこれを検証して不整合を早期検知できる。
#: 1.1: registry の state_board / dynamic_max_chars を ContextVar 化し、1プロセスで複数の
#:      Engine を別スレッドで並行実行できるようにした（マルチセッション対応）。API 追加のみで
#:      1.0 と後方互換（Engine.run_turn がターン開始時に自セッションの state_board を束縛する）。
API_VERSION = "1.1"

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

    def _bind_context(self) -> None:
        """このセッションの state_board を現在の実行コンテキストへ束縛する。

        マルチセッションの要: run_turn を実行するスレッド内で呼ぶことで、registry の
        ContextVar がこのセッション専用の値になる（並列ツール実行にも copy_context で伝播）。
        別セッションのターンが別スレッドで走っていても互いに干渉しない。
        """
        set_state_board(self.state.state_board)

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
        self._bind_context()  # マルチセッション: 自セッションの state_board をこのスレッドに束縛
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
        - AgentState を生成（state_board の束縛は各 run_turn 内で行う＝マルチセッション対応）。

    Args:
        server: {"base_url", "api_key"?, "model"?} 形式（AWP の config.json servers[] と同形式）。
        workspace: エージェントの作業対象＝cwd＝サンドボックスのルート（絶対パス推奨）。

    注意（cwd 共有）: os.chdir はプロセス全体に効くため、1プロセスで複数 Engine を作る場合は
    全 Engine が同一 workspace を共有する（作業対象別のセッションは Phase 2 の cwd 抽象化待ち）。
    state_board のメモリ内分離は ContextVar で保証されるが、cwd 依存の永続ファイル
    (.pixie_notes/state_board.json) は共有されるため、複数セッション同時運用では
    最後の保存が勝つ点に注意（メモリ内の推論状態は正しく分離される）。
    """
    os.chdir(str(workspace))
    paths.set_project_root(os.getcwd())

    from main import AppContext  # 遅延 import: CLI 層(main)を必要時までパッケージに巻き込まない
    ctx = AppContext()
    ctx.llm = LMStudioBackend(
        server["base_url"],
        server.get("api_key", "lm-studio"),
        server.get("model", "local-model"),
    )
    # サンプリングプロファイルはモデル名の部分一致で選ばれる（空だと常に default）。
    ctx.llm_model_name = server.get("model", "") or ""

    return Engine(ctx, AgentState())
