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

from pathlib import Path

# --- AWP 内部（この境界の内側でのみ import する） ---
# 注: AppContext は CLI 層 main.py にあり（パッケージ外）、CLI スタックを巻き込むため
# トップレベルでは import せず create_engine() 内で遅延 import する。
from engine import run_graph, build_system_text
from state import AgentState
from registry import set_state_board, TOOL_REGISTRY, register_tool
from llm_client import LMStudioBackend
from config import DESTRUCTIVE_TOOLS, READONLY_TOOLS
import paths
from paths import get_workspace  # 現セッションの workspace（外部ツールのパス解決用に再エクスポート）

# ツール登録の副作用（@register_tool）。import するだけで TOOL_REGISTRY が満たされる。
import tools as _tools          # noqa: F401
import code_tool as _code_tool  # noqa: F401

#: 公開 API のバージョン。組み込み側は起動時にこれを検証して不整合を早期検知できる。
#: 1.1: registry の state_board / dynamic_max_chars を ContextVar 化（マルチセッション）。
#: 1.2: workspace も ContextVar 化し os.chdir を廃止。セッションごとに別フォルダを扱える。
#: 1.3: register_tool を公開。組み込み側が外部ツール（例: ask_copilot）を pack 付きで登録し、
#:      context.active_packs で on/off できるようにした。いずれも API 追加のみで後方互換。
API_VERSION = "1.3"

#: 外部ツール登録用のデコレータ（registry.register_tool の再エクスポート）。
#: 組み込み側は `@pixie_core.register_tool(name=..., pack="...")` で TOOL_REGISTRY に追加できる。
#: pack を付けると、そのセッションの context.active_packs に pack 名が含まれる時だけ LLM に提示される。

__all__ = [
    "API_VERSION", "CancelTurn", "READONLY_TOOLS", "DESTRUCTIVE_TOOLS",
    "create_engine", "Engine", "tool_count", "register_tool", "get_workspace",
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

    UI 非依存。1プロセスで複数 Engine を別スレッドで並行実行できる（state_board と
    workspace はターンごとに ContextVar 束縛され、セッション間で分離される）。
    """

    def __init__(self, context: AppContext, state: AgentState, workspace: str | None = None):
        self.context = context
        self.state = state
        self.workspace = workspace  # このセッションの作業対象フォルダ（絶対パス）

    def _bind_context(self) -> None:
        """このセッションの state_board と workspace を現在の実行コンテキストへ束縛する。

        マルチセッションの要: run_turn を実行するスレッド内で呼ぶことで、registry の
        state_board ContextVar と paths の workspace ContextVar がこのセッション専用の値になる
        （並列ツール実行にも copy_context で伝播）。別セッションのターンが別スレッドで走っていても
        互いに干渉しない。
        """
        set_state_board(self.state.state_board)
        if self.workspace:
            paths.bind_workspace(self.workspace)

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
    """埋め込み用 Engine を構築する（セッション別 workspace 対応・os.chdir しない）。

    行うこと:
        - AppContext を実クラスで生成し、LM Studio バックエンドを接続。
        - workspace を ContextVar に一時束縛した状態で AgentState を生成する。StateBoard/lessons/
          trajectory は構築時に永続パス（<workspace>/.pixie_notes/...）をキャプチャするため、
          この順序が重要。生成後に束縛は元に戻す（create_engine を呼んだスレッドに残さない）。

    Args:
        server: {"base_url", "api_key"?, "model"?} 形式（AWP の config.json servers[] と同形式）。
        workspace: エージェントの作業対象＝サンドボックスのルート（絶対パス推奨）。

    マルチセッション: os.chdir（プロセス全体）に依存しないため、1プロセスで別々の workspace を
    持つ複数 Engine を並行実行できる。実行時のファイル解決・永続化は run_turn がターンごとに
    workspace ContextVar を束縛し、engine のディスパッチ正規化＋paths.get_project_root() が担う。
    （プロセス cwd 自体は変更しないので、開いている workspace フォルダを OS 上で削除・移動もできる。）
    """
    from main import AppContext  # 遅延 import: CLI 層(main)を必要時までパッケージに巻き込まない

    ws = str(Path(workspace).resolve())
    ctx = AppContext()
    ctx.llm = LMStudioBackend(
        server["base_url"],
        server.get("api_key", "lm-studio"),
        server.get("model", "local-model"),
    )
    # サンプリングプロファイルはモデル名の部分一致で選ばれる（空だと常に default）。
    ctx.llm_model_name = server.get("model", "") or ""

    # StateBoard 等が構築時に <ws>/.pixie_notes/... を捕まえるよう、束縛してから生成→復元。
    token = paths.bind_workspace(ws)
    try:
        state = AgentState()
    finally:
        paths.reset_workspace(token)

    return Engine(ctx, state, workspace=ws)
