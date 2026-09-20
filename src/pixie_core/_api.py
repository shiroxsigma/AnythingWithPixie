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
    class Engine                        — run_turn / workspace snapshot / Workset 構築。

注意: このモジュールは AWP の `src` をパスに含めた状態で import すること（AWP と同じフラット
import 前提: `from engine import ...`）。組み込み側は sys.path に AWP/src を前置してから
`import pixie_core` する。
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
from dataclasses import dataclass, field
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
from . import changeset as _changeset
from . import workset as _workset
from .turn_control import TurnControl, TurnLimits, TurnStopped, active_control

_WORKSET_IGNORE_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".pixie_notes",
})
_WORKSET_TEXT_SUFFIXES = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".rst", ".txt", ".json",
    ".yaml", ".yml", ".toml",
})
_WORKSET_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})
_WORKSET_TASK_STOPWORDS = frozenset({
    "この", "ファイル", "コード", "実装", "修正", "変更", "追加", "確認", "して",
    "する", "ください", "the", "and", "with", "file", "code", "change", "update",
})


def _workset_task_terms(task: str) -> set[str]:
    """ファイル名照合に使える識別子・日本語名詞らしき語だけを小さく取り出す。"""
    terms = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}|[ぁ-んァ-ヶ一-龠]{2,}", task.lower())
    return {term.strip("._-") for term in terms
            if term.strip("._-") and term not in _WORKSET_TASK_STOPWORDS}


def _related_workset_paths(root: Path, task: str, pinned_paths: list[str],
                           buffers=None, scan_limit: int = 20_000):
    """ASTとMarkdown索引から関連候補を返す（LLM呼び出しなし）。"""
    return _workset.analyze(root, task, pinned_paths, buffers=buffers, scan_limit=scan_limit)

#: 公開 API のバージョン。組み込み側は起動時にこれを検証して不整合を早期検知できる。
#: 1.1: registry の state_board / dynamic_max_chars を ContextVar 化（マルチセッション）。
#: 1.2: workspace も ContextVar 化し os.chdir を廃止。セッションごとに別フォルダを扱える。
#: 1.3: register_tool を公開。組み込み側が外部ツール（例: ask_copilot）を pack 付きで登録し、
#:      context.active_packs で on/off できるようにした。いずれも API 追加のみで後方互換。
#: 1.4: create_engine に tool_set / system_suffix を追加（固定ツールプロファイルと静的
#:      システムプロンプト追記。例: NoteWithPixie の read 専用モード + 編集プロトコル指示）。
#:      Engine.load_history を追加（外部永続化履歴からの文脈シード）。API 追加のみで後方互換。
#: 1.5: 思考許容時間の実行時変更。set_think_budget/get_think_budget（deep モードの <think>
#:      上限秒。プロセス全体）と Engine.set_stream_timeout（LLM ストリームの打ち切り秒。
#:      セッション単位）。組み込み側の設定画面から変えられるようにするため。API 追加のみ。
#: 1.6: 会話履歴の外科的編集。Engine.history_size / history_tail / history_drop /
#:      history_replace。組み込み側が「回答が不要だった往復だけを文脈から消す」「会話を
#:      要約して次セッションへ引き継ぐ」を実現するため（CWP のコンテキスト節約機能）。
#:      API 追加のみで後方互換。
#: 1.7: Engine.set_workspace_snapshot。エディタの未保存バッファをセッション別に保持し、
#:      read_file がディスクより優先して読むための埋め込み API。
#: 1.8: Engine.build_workset。ピン留め・選択対象を全文なしの構造化参照へ変換する API。
#: 1.9: ChangeSet の preview / validate / apply / revert。複数ファイルをjournal付きで扱う。
#: 1.10: Markdown節操作と、要件・用語・リンク・MermaidのChangeSet整合性検査。
#: 1.11: AgentProfile / ContextPolicy / EngineEvent / turn metrics の公開境界。
API_VERSION = "1.12"

#: 外部ツール登録用のデコレータ（registry.register_tool の再エクスポート）。
#: 組み込み側は `@pixie_core.register_tool(name=..., pack="...")` で TOOL_REGISTRY に追加できる。
#: pack を付けると、そのセッションの context.active_packs に pack 名が含まれる時だけ LLM に提示される。

__all__ = [
    "API_VERSION", "TurnControl", "TurnLimits", "TurnStopped", "CancelTurn", "READONLY_TOOLS", "DESTRUCTIVE_TOOLS",
    "create_engine", "Engine", "tool_count", "register_tool", "get_workspace",
    "set_think_budget", "get_think_budget", "AgentProfile", "ContextPolicy", "EngineEvent",
]


@dataclass(frozen=True)
class ContextPolicy:
    """埋め込み先が安全に指定できるモデルコンテキストと通信上限。"""

    context_length: int | None = None
    overall_timeout: float | None = None
    read_idle_timeout: float | None = None

    def __post_init__(self):
        for name in ("context_length", "overall_timeout", "read_idle_timeout"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{name} は正数で指定してください: {value!r}")

    @classmethod
    def from_value(cls, value) -> "ContextPolicy":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("context policy は ContextPolicy または辞書で指定してください")

    def as_dict(self) -> dict:
        return {name: value for name in (
            "context_length", "overall_timeout", "read_idle_timeout")
                if (value := getattr(self, name)) is not None}


@dataclass(frozen=True)
class AgentProfile:
    """ツール・静的指示・pack・実行上限をまとめたセッション構築プロファイル。"""

    name: str = "code"
    tool_set: frozenset[str] | None = None
    system_suffix: str = ""
    active_packs: frozenset[str] = field(default_factory=frozenset)
    context_policy: ContextPolicy | None = None

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("profile name は空にできません")
        if self.tool_set is not None and not isinstance(self.tool_set, frozenset):
            object.__setattr__(self, "tool_set", frozenset(self.tool_set))
        if not isinstance(self.active_packs, frozenset):
            object.__setattr__(self, "active_packs", frozenset(self.active_packs))
        if self.context_policy is not None and not isinstance(self.context_policy, ContextPolicy):
            object.__setattr__(self, "context_policy", ContextPolicy.from_value(self.context_policy))


@dataclass(frozen=True)
class EngineEvent:
    """UI非依存のターンイベント。dataはイベント型固有の追加情報。"""

    type: str
    text: str = ""
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        value = {"type": self.type}
        if self.text:
            value["text"] = self.text
        value.update(copy.deepcopy(self.data))
        return value


class CancelTurn(Exception):
    """協調キャンセル。組み込み側の output_fn / interactive_fn から送出するとターンを打ち切る。

    run_graph は同期ループなので、外部からの中断は「コールバック内で例外を送出して脱出する」
    のが唯一安全な方法（スレッドの強制終了はできない）。承認コールバックで空承認を返す経路と
    併用すると、LLM 生成中／承認待ちのどちらでも確実に止められる。
    """


def tool_count() -> int:
    """登録済みツール数（疎通スモークにも使える）。"""
    return len(TOOL_REGISTRY)


def set_think_budget(seconds) -> int:
    """deep 思考モードの <think> 最大継続秒数を実行中に変更する（API 1.5・プロセス全体）。

    engine は起動時に config.DEEP_THINK_BUDGET_SEC を自モジュールへ束縛して使うため、
    config 側を書き換えても反映されない。ここで engine のモジュール変数を差し替える
    （run_graph は呼び出しのたびに参照するので、次ターンから効く）。

    スコープはプロセス全体（セッション別ではない）。埋め込み側の設定画面が「思考許容時間」
    として1つの値を持つ運用を想定している。実行中ターンには影響しない（ループ内で毎回
    比較されるため、厳密には走行中でも次の比較から効くが、依存しないこと）。

    Returns: 適用された秒数。5 未満や数値でない値は ValueError。
    """
    import engine as _engine

    try:
        v = int(seconds)
    except (TypeError, ValueError):
        raise ValueError(f"思考許容時間は秒数で指定してください: {seconds!r}")
    if v < 5:
        raise ValueError(f"思考許容時間が短すぎます（5秒以上）: {v}")
    _engine.DEEP_THINK_BUDGET_SEC = v
    return v


def get_think_budget() -> int:
    """現在の deep 思考の <think> 上限秒（API 1.5）。"""
    import engine as _engine

    return int(_engine.DEEP_THINK_BUDGET_SEC)


def _make_system_builder(suffix: str):
    """build_system_text に静的 suffix を追記する system_msg_builder を返す（API 1.4）。

    suffix はセッション内で不変であること。「静的指示は system、動的文脈は直近ユーザー
    メッセージ末尾」という設計の system 側に載せるためのフックであり、ターン毎に変わる
    内容を入れると prefix cache が全壊する。空文字なら build_system_text をそのまま返す。
    """
    if not suffix:
        return build_system_text

    def builder(context, state_board=None, **kw):
        return build_system_text(context, state_board, **kw) + "\n\n" + suffix

    return builder


class Engine:
    """1セッション分の埋め込みエンジン。AppContext と AgentState を保持し run_graph を回す。

    UI 非依存。1プロセスで複数 Engine を別スレッドで並行実行できる（state_board と
    workspace はターンごとに ContextVar 束縛され、セッション間で分離される）。
    """

    def __init__(self, context: object, state: AgentState, workspace: str | None = None,
                 system_suffix: str = ""):
        self.context = context
        self.state = state
        self.workspace = workspace  # このセッションの作業対象フォルダ（絶対パス）
        self._workspace_buffers: dict[str, dict] = {}
        self._working_file_context: dict | None = None
        # [API 1.4] 静的システムプロンプト追記。構築時に一度だけビルダーへ変換して保持する
        # （セッション内不変の契約を型で表す。ターン毎の再構築はしない）。
        self._system_builder = _make_system_builder(system_suffix)

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
        paths.bind_workspace_buffers(self._workspace_buffers)
        paths.bind_working_file_context(self._working_file_context)

    def set_working_files(self, files: list[str] | None) -> None:
        """編集対象を版付きWorksetへ登録し、次ターンから最新版だけを注入する。

        内容は会話履歴へ追加されない。AWP自身または外部プロセスが編集するとrevisionと
        hashが更新され、古い全文はLLMコンテキストへ残らない。None/空配列で解除する。
        """
        if files is None:
            files = []
        if not isinstance(files, list) or any(not isinstance(path, str) for path in files):
            raise TypeError("working files は文字列の配列である必要があります")
        if not files:
            self._working_file_context = None
            return
        self._working_file_context = paths.create_working_file_context(
            self.workspace or Path.cwd(), files
        )

    def set_workspace_snapshot(self, snapshot: dict | None) -> None:
        """エディタの現在状態を次ターン以降のファイル読み取りへ反映する（API 1.7）。

        ``snapshot`` は ``{"buffers": [{"path", "content", "base_hash"?}]}``。
        workspace 外のパス、文字列でない内容、不正な項目は拒否する。登録したパスは
        ``read_file`` でディスクより優先されるため、UI の未保存編集を古いディスク版で
        上書きせずに調査できる。空または None で前ターンのスナップショットを消去する。

        呼び出しはターン境界で行うこと。同一 Engine の実行中に変更してはならない。
        """
        raw_buffers = [] if not snapshot else snapshot.get("buffers", [])
        if not isinstance(raw_buffers, list):
            raise TypeError("workspace snapshot の buffers は配列である必要があります")
        root = Path(self.workspace or Path.cwd()).resolve()
        normalized: dict[str, dict] = {}
        for item in raw_buffers:
            if not isinstance(item, dict):
                raise TypeError("workspace buffer はオブジェクトである必要があります")
            path, content = item.get("path"), item.get("content")
            if not isinstance(path, str) or not path.strip() or not isinstance(content, str):
                raise TypeError("workspace buffer には文字列の path と content が必要です")
            # API入力ではWindows形式/ POSIX形式のどちらも同じpath identityへ寄せる。
            normalized_path = path.replace("\\", os.sep).replace("/", os.sep)
            target = Path(normalized_path)
            if not target.is_absolute():
                target = root / target
            target = target.resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"workspace 外の buffer は登録できません: {path}") from exc
            supplied_base_hash = item.get("base_hash")
            if not isinstance(supplied_base_hash, str):
                try:
                    disk_bytes = target.read_bytes()
                except FileNotFoundError:
                    disk_bytes = b""
                supplied_base_hash = hashlib.sha256(disk_bytes).hexdigest()
            normalized[str(target)] = {
                "content": content,
                "base_hash": supplied_base_hash,
            }
        self._workspace_buffers = normalized

    def build_workset(self, request: dict | None) -> dict:
        """ピン留め・選択対象から、本文を含まない最小 Workset を構築する（API 1.8）。

        ``request`` の公開入力は ``task``、``pinned_paths``、``selected_path``、
        ``max_items``、``include_related``。内容は直前の :meth:`set_workspace_snapshot` に同じパスがあれば
        未保存バッファを、無ければディスクを参照する。返す item は path、役割、行数、
        文字数、SHA-256 と buffer フラグだけで、巨大な全文をプロンプトへ複製しない。

        読めない・存在しない・workspace 外・上限超過の候補は黙って捨てず、``omitted``
        に理由を残す。呼び出しは set_workspace_snapshot と同じくターン境界で行うこと。
        """
        if request is None:
            request = {}
        if not isinstance(request, dict):
            raise TypeError("workset request はオブジェクトである必要があります")
        task = request.get("task", "")
        if not isinstance(task, str):
            raise TypeError("workset の task は文字列である必要があります")
        pinned = request.get("pinned_paths", [])
        if not isinstance(pinned, list) or any(not isinstance(p, str) for p in pinned):
            raise TypeError("workset の pinned_paths は文字列の配列である必要があります")
        selected = request.get("selected_path")
        if selected is not None and not isinstance(selected, str):
            raise TypeError("workset の selected_path は文字列である必要があります")
        try:
            max_items = max(1, min(100, int(request.get("max_items", 24))))
        except (TypeError, ValueError) as exc:
            raise TypeError("workset の max_items は整数である必要があります") from exc

        root = Path(self.workspace or Path.cwd()).resolve()
        explicit = ([selected] if selected else []) + pinned
        candidates: list[tuple[str, str, float, str]] = []
        for raw_path in explicit:
            role = "target" if selected and raw_path == selected else "pinned"
            candidates.append((raw_path, role, 1.0 if role == "target" else 0.95, "explicit"))
        scan_truncated = False
        metadata: dict[str, dict] = {}
        index_stats: dict[str, int] = {}
        if request.get("include_related", True):
            related, scan_truncated, metadata, index_stats = _related_workset_paths(
                root, task, explicit, self._workspace_buffers
            )
            candidates.extend(related)
        items: list[dict] = []
        omitted: list[dict] = []
        if scan_truncated:
            omitted.append({"path": "*", "reason": "scan_limit"})
        seen: set[str] = set()
        auto_added = 0
        for raw_path, role, score, reason in candidates:
            if not raw_path or not raw_path.strip():
                continue
            target = Path(raw_path)
            if not target.is_absolute():
                target = root / target
            try:
                target = target.resolve()
                rel = target.relative_to(root).as_posix()
            except (OSError, ValueError):
                omitted.append({"path": raw_path, "reason": "outside_workspace"})
                continue
            key = str(target)
            if key in seen:
                continue
            seen.add(key)
            if len(items) >= max_items:
                omitted.append({"path": rel, "reason": "max_items"})
                continue

            buffered = self._workspace_buffers.get(key)
            try:
                if buffered is not None:
                    content = buffered["content"]
                else:
                    content = target.read_text(encoding="utf-8")
            except FileNotFoundError:
                omitted.append({"path": rel, "reason": "not_found"})
                continue
            except (OSError, UnicodeError):
                omitted.append({"path": rel, "reason": "unreadable_text"})
                continue

            item = {
                "path": rel,
                "role": role,
                "score": score,
                "reason": reason,
                "lines": len(content.splitlines()),
                "chars": len(content),
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "buffer": buffered is not None,
            }
            item.update(metadata.get(rel, {}))
            items.append(item)
            if reason != "explicit":
                auto_added += 1
        return {
            "schema_version": "1",
            "task": task,
            "items": items,
            "omitted": omitted,
            "stats": {"items": len(items), "omitted": len(omitted), "auto_added": auto_added,
                      "scan_truncated": scan_truncated, **index_stats},
        }

    def preview_changeset(self, changeset: dict) -> dict:
        """複数ファイル変更の適用後内容を計算する。ファイルは変更しない（API 1.9）。"""
        return _changeset.preview(self.workspace or str(Path.cwd()), changeset,
                                  self._workspace_buffers)

    def validate_changeset(self, changeset: dict) -> dict:
        """ChangeSet の操作・path・base hash競合を検証する（API 1.9）。"""
        return _changeset.validate(self.workspace or str(Path.cwd()), changeset,
                                   self._workspace_buffers)

    def apply_changeset(self, changeset: dict) -> dict:
        """検証済みChangeSetをstageし、journalを残して一括適用する（API 1.9）。"""
        result = _changeset.apply(self.workspace or str(Path.cwd()), changeset,
                                  self._workspace_buffers)
        if result.get("applied"):
            # An approval hook can apply edits here instead of dispatching the
            # proposed write tool. Preserve that mutation in the action sequence
            # so a read before and after the edit is not a consecutive-read loop.
            actions = getattr(self.state, "executed_actions", None)
            if isinstance(actions, list):
                actions.append(f"apply_changeset:{result['id']}")
                self.state.loop_warn_count = 0
            pending = getattr(self.state, "pending_applied_changesets", None)
            if isinstance(pending, list):
                pending.append({
                    "id": result["id"],
                    "paths": [item["path"] for item in result.get("changes", [])],
                    "mutation_count": sum(
                        len(item.get("operations", []))
                        for item in changeset.get("changes", [])
                    ),
                })
            for item in result.get("changes", []):
                target = str((Path(self.workspace or Path.cwd()) / item["path"]).resolve())
                buffer_key = paths._lookup_path_key(self._workspace_buffers, Path(target))
                if item.get("delete"):
                    if buffer_key is not None:
                        self._workspace_buffers[buffer_key].update({
                            "deleted": True,
                            "base_hash": hashlib.sha256(b"").hexdigest(),
                        })
                elif buffer_key is not None:
                    disk_hash = hashlib.sha256(Path(target).read_bytes()).hexdigest()
                    self._workspace_buffers[buffer_key].update({
                        "content": item["after"],
                        "base_hash": disk_hash,
                    })
        return result

    def revert_changeset(self, change_id: str, *, force: bool = False) -> dict:
        """永続journalから、作成・変更・削除を適用前へ戻す（API 1.9）。"""
        result = _changeset.revert(self.workspace or str(Path.cwd()), change_id, force=force)
        if result.get("reverted"):
            # diskへ復元した後に古い仮想内容を優先しないよう、復元対象bufferを破棄する。
            root = Path(self.workspace or Path.cwd()).resolve()
            for rel in result.get("restored", []):
                self._workspace_buffers.pop(str((root / rel).resolve()), None)
        return result

    @property
    def model_name(self) -> str:
        return getattr(self.context, "llm_model_name", "") or ""

    @property
    def tool_count(self) -> int:
        return len(TOOL_REGISTRY)

    def run_turn(self, user_text: str, *, output_fn, interactive_fn=None,
                 show_thinking: bool = False, control=None) -> str:
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
        if control is None:
            llm = self.context.llm
            control = TurnControl(TurnLimits(
                think_seconds=get_think_budget(),
                stream_timeout=getattr(llm, "overall_timeout", 180.0),
                read_idle_timeout=getattr(llm, "read_idle_timeout", 30.0),
            ))
        token = active_control.set(control)
        try:
            control.check()
            self._bind_context()  # マルチセッション: 自セッションの state_board をこのスレッドに束縛
            self.state.reset_for_new_turn()
            self.state.chat_history.add("user", user_text)
            return run_graph(
                context=self.context,
                state=self.state,
                show_thinking=show_thinking,
                system_msg_builder=self._system_builder,
                interactive_fn=interactive_fn,
                output_fn=output_fn,
            )
        finally:
            active_control.reset(token)


    def run_turn_events(self, user_text: str, *, event_fn, interactive_fn=None,
                        show_thinking: bool = False, control=None) -> str:
        """run_turnを型付きEngineEventで駆動する（API 1.11）。

        engine内部の既存出力は``output``イベントへ可逆に格納する。埋め込み側は必要なら
        UI固有のtoken/status分類を行える。開始・完了・例外とturn metricsは文字列解析不要。
        """
        event_fn(EngineEvent("turn_started"))

        def output(text, end="", flush=False):
            event_fn(EngineEvent("output", str(text or ""), {
                "end": end, "flush": bool(flush)}))

        try:
            result = self.run_turn(
                user_text,
                output_fn=output,
                interactive_fn=interactive_fn,
                show_thinking=show_thinking,
                **({"control": control} if control is not None else {}),
            )
        except (CancelTurn, TurnStopped):
            raise
        except Exception as exc:
            event_fn(EngineEvent("turn_error", f"{type(exc).__name__}: {exc}", {
                "metrics": self.get_turn_metrics()}))
            raise
        event_fn(EngineEvent("turn_completed", str(result or ""), {
            "metrics": self.get_turn_metrics()}))
        return result

    def get_turn_metrics(self) -> dict:
        """直近ターンの計測・終了理由・受け入れ条件を安全なコピーで返す（API 1.11）。"""
        return {
            "llm_calls": copy.deepcopy(getattr(self.state, "llm_call_metrics", [])),
            "tool_calls": int(getattr(self.state, "tool_call_count", 0)),
            "exit_reason": str(getattr(self.state, "exit_reason", "") or ""),
            "acceptance_conditions": copy.deepcopy(
                getattr(self.state, "acceptance_conditions", [])),
            "acceptance_retries": int(getattr(self.state, "acceptance_retry_count", 0)),
        }

    def set_context_policy(self, policy) -> dict:
        """モデル窓長とストリーム上限を公開API経由で設定する（API 1.11）。"""
        value = ContextPolicy.from_value(policy)
        llm = getattr(self.context, "llm", None)
        if llm is None:
            return value.as_dict()
        if value.context_length is not None:
            setter = getattr(llm, "set_context_length", None)
            if callable(setter):
                setter(int(value.context_length))
            elif hasattr(llm, "_n_ctx"):
                llm._n_ctx = int(value.context_length)
            else:
                raise RuntimeError("このLLM backendはcontext length変更に対応していません")
        if value.overall_timeout is not None:
            llm.overall_timeout = float(value.overall_timeout)
        if value.read_idle_timeout is not None:
            llm.read_idle_timeout = float(value.read_idle_timeout)
        return value.as_dict()

    def set_profile(self, profile) -> AgentProfile:
        """Replace the session profile at a turn boundary (API 1.11)."""
        if not isinstance(profile, AgentProfile):
            if isinstance(profile, dict):
                profile = AgentProfile(**profile)
            else:
                raise TypeError("profile は AgentProfile または辞書で指定してください")
        self.context.fixed_tool_set = profile.tool_set
        self.context.active_packs = set(profile.active_packs)
        # CLI の /code と同じガードレール設定を埋め込み利用でも再現する。
        # これが無いと AgentProfile(name="code") でも短い正解を不完全と誤判定し、
        # 不要な再調査や best-of 呼び出しを重ねる。
        self.context.code_mode = profile.name.strip().lower() == "code"
        self._system_suffix = profile.system_suffix
        self._system_builder = _make_system_builder(profile.system_suffix)
        if profile.context_policy is not None:
            self.set_context_policy(profile.context_policy)
        self.profile = profile
        return profile

    def set_stream_timeout(self, overall_timeout: float, read_idle_timeout: float | None = None) -> None:
        """このセッションの LLM ストリーム打ち切り秒を変更する（API 1.5）。

        overall_timeout: チャンクが届き続けていても、この秒数を超えたら生成を打ち切る
                         （既定 180）。思考許容時間をこれより長くすると、思考の途中で
                         こちらに引っかかるため、埋め込み側で併せて引き上げること。
        read_idle_timeout: 完全無応答の検知に使うソケット受信タイムアウト（既定 30）。
                         None なら据え置き。
        """
        llm = getattr(self.context, "llm", None)
        if llm is None:
            return
        llm.overall_timeout = float(overall_timeout)
        if read_idle_timeout is not None:
            llm.read_idle_timeout = float(read_idle_timeout)

    def load_history(self, messages: list[dict]) -> None:
        """外部で永続化された会話履歴でこのセッションの ChatHistory をシードする（API 1.4）。

        用途: サーバ再起動後、組み込みアプリ側のサイドカー（例: NWP の .pixie_chat.json）の
        直近履歴から LLM 文脈を復元する。セッション新規作成直後に一度だけ呼ぶこと。
        role は "user"/"assistant" のみ取り込み、それ以外と空 content は無視する。

        注意: ここで _bind_context() は呼ばない。ChatHistory への追加は純粋なメモリ操作で
        workspace/state_board 束縛を必要とせず、呼び出しスレッド（アプリのイベントループ等）に
        ContextVar 束縛を漏らすと他セッションのパス解決を汚染するため（束縛は run_turn が
        worker スレッド内で行う契約）。
        """
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                self.state.chat_history.add(role, content)
        # add() は自動トリムしないため、max_messages を超えた古い側をここで落とす。
        self.state.chat_history.trim()

    # --- [API 1.6] 会話履歴の外科的編集（組み込み側のコンテキスト節約機能） -----------
    # 使い方（想定シーケンス）:
    #     n = engine.history_size()
    #     engine.run_turn(...)
    #     handles = engine.history_tail(n)   # このターンが積んだメッセージ群
    #     ...後で不要と分かったら...
    #     engine.history_drop(handles)       # その往復だけを文脈から外す
    # index ではなく**オブジェクトの同一性**で指すのが要点。run_graph の自動トリム
    # (check_and_trim_context) が古い側を先頭から落とすため、index は後からずれる。

    def history_size(self) -> int:
        """現在の履歴メッセージ数（system を含まない）。ターン境界の記録に使う。"""
        return len(self.state.chat_history.messages)

    def history_tail(self, start: int) -> list:
        """start 以降のメッセージを新しいリストに入れて返す（API 1.6）。

        要素は履歴が保持している dict **そのもの**（コピーではない）。呼び出し側は
        これを「後でその往復を指すためのハンドル」として保持する用途で使い、
        中身を書き換えないこと（書き換えは LLM へ送る履歴を直接汚す）。
        """
        return list(self.state.chat_history.messages[max(0, int(start)):])

    def history_drop(self, handles) -> int:
        """history_tail が返したメッセージを履歴から取り除く（API 1.6）。除去件数を返す。

        同一性 (`is`) で照合するので、既に自動トリムで落ちているものは黙って無視される
        （＝「消したはずのものが消えていない」ことは起きない）。除去後は
        assistant(tool_calls) と tool 応答の対が壊れていないか繕う — 対が崩れた履歴は
        OpenAI 互換 API が 400 を返すため、次のターンが丸ごと失敗する。
        """
        targets = [h for h in handles if isinstance(h, dict)]
        if not targets:
            return 0
        msgs = self.state.chat_history.messages
        kept = [m for m in msgs if not any(m is t for t in targets)]
        removed = len(msgs) - len(kept)
        if removed:
            self.state.chat_history.messages = _repair_tool_pairs(kept)
        return removed

    def history_replace(self, messages: list[dict]) -> None:
        """履歴を丸ごと差し替える（API 1.6）。用途: 会話の要約による引き継ぎ（/compact）。

        取り込み条件は load_history と同じ（user/assistant の非空 content のみ）。
        ツール呼び出しの痕跡は残らないので、対の繕いは不要。
        """
        self.state.chat_history.messages = []
        self.load_history(messages)


def _repair_tool_pairs(messages: list) -> list:
    """assistant(tool_calls) ↔ tool 応答の対応が崩れた履歴を繕う（history_drop の後始末）。

    1. 呼び出し元の assistant が居ない tool メッセージを落とす（孤児の実行結果）。
    2. tool 応答が続かない assistant(tool_calls) から tool_calls を外す。本文が無ければ
       そのメッセージ自体を落とす（中身が呼び出しだけだったため）。
    """
    out: list = []
    for m in messages:
        if m.get("role") == "tool":
            prev = out[-1] if out else None
            ok = prev is not None and (
                prev.get("role") == "tool"
                or (prev.get("role") == "assistant" and prev.get("tool_calls")))
            if not ok:
                continue
        out.append(m)

    kept: list = []
    for i, m in enumerate(out):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            answered = i + 1 < len(out) and out[i + 1].get("role") == "tool"
            if not answered:
                if not str(m.get("content") or "").strip():
                    continue
                m = {k: v for k, v in m.items() if k != "tool_calls"}
        kept.append(m)
    return kept


def create_engine(server: dict, workspace: str, *,
                  tool_set=None, system_suffix: str = "", profile: AgentProfile | None = None) -> Engine:
    """埋め込み用 Engine を構築する（セッション別 workspace 対応・os.chdir しない）。

    行うこと:
        - AppContext を実クラスで生成し、LM Studio バックエンドを接続。
        - workspace を ContextVar に一時束縛した状態で AgentState を生成する。StateBoard/lessons/
          trajectory は構築時に永続パス（<workspace>/.pixie_notes/...）をキャプチャするため、
          この順序が重要。生成後に束縛は元に戻す（create_engine を呼んだスレッドに残さない）。

    Args:
        server: {"base_url", "api_key"?, "model"?} 形式（AWP の config.json servers[] と同形式）。
        workspace: エージェントの作業対象＝サンドボックスのルート（絶対パス推奨）。
        tool_set: [API 1.4] LLM に提示するツール名の固定集合（iterable）。指定すると pack や
                  コア集合に関係なく「この集合のみ」が提示される（例: NWP の read 専用プロファイル）。
                  None（既定）は従来通り。変更はターン境界でのみ行うこと（prefix cache 保護）。
        system_suffix: [API 1.4] システムプロンプト末尾に追記する静的テキスト
                  （例: NWP の search/replace 編集プロトコル指示）。セッション内不変であること。

    マルチセッション: os.chdir（プロセス全体）に依存しないため、1プロセスで別々の workspace を
    持つ複数 Engine を並行実行できる。実行時のファイル解決・永続化は run_turn がターンごとに
    workspace ContextVar を束縛し、engine のディスパッチ正規化＋paths.get_project_root() が担う。
    （プロセス cwd 自体は変更しないので、開いている workspace フォルダを OS 上で削除・移動もできる。）
    """
    from main import AppContext  # 遅延 import: CLI 層(main)を必要時までパッケージに巻き込まない

    if profile is not None and not isinstance(profile, AgentProfile):
        if isinstance(profile, dict):
            profile = AgentProfile(**profile)
        else:
            raise TypeError("profile は AgentProfile または辞書で指定してください")
    if profile is not None:
        if tool_set is None:
            tool_set = profile.tool_set
        if not system_suffix:
            system_suffix = profile.system_suffix

    ws = str(Path(workspace).resolve())
    ctx = AppContext()
    ctx.llm = LMStudioBackend(
        server["base_url"],
        server.get("api_key", "lm-studio"),
        server.get("model", "local-model"),
    )
    # サンプリングプロファイルはモデル名の部分一致で選ばれる（空だと常に default）。
    ctx.llm_model_name = server.get("model", "") or ""
    # [LFM専用] CLI のサーバー切替(main.py の /api 処理)と同じ判定。これが無いと埋め込み経路では
    # tool_choice の丸め・role="tool" 送信などの LFM 専用処理が一切発火しない。
    ctx.is_lfm25 = "lfm" in ctx.llm_model_name.lower()
    ctx.supports_tool_role = ctx.is_lfm25
    # [API 1.4] 固定ツールプロファイル。engine の node_plan が最優先で参照する。
    if tool_set:
        ctx.fixed_tool_set = frozenset(tool_set)
    if profile is not None:
        ctx.active_packs = set(profile.active_packs)

    # StateBoard 等が構築時に <ws>/.pixie_notes/... を捕まえるよう、束縛してから生成→復元。
    token = paths.bind_workspace(ws)
    try:
        state = AgentState()
    finally:
        paths.reset_workspace(token)

    engine = Engine(ctx, state, workspace=ws, system_suffix=system_suffix)
    initial_profile = profile or AgentProfile(
        name="custom" if tool_set is not None or system_suffix else "code",
        tool_set=frozenset(tool_set) if tool_set is not None else None,
        system_suffix=system_suffix,
    )
    engine.set_profile(initial_profile)
    return engine
