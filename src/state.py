"""
AnythingPixie — 状態管理モジュール

AgentStateBoard（統合ステートボード）、ChatHistory（スライディングウィンドウ）、
AgentState（ReActループ用状態）、およびプロンプト注入を統合管理する。

依存: config.py（なし）, 標準ライブラリのみ
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from paths import get_data_path

# =====================================================
# AgentStateBoard — 統合ステートボード
# =====================================================

class AgentStateBoard:
    """統合ステートボード — 履歴ではなく「状態」を管理する。

    設計ルール:
    - 上書きのみ（追記禁止）。常に最新の1ブロックとして書き直す
    - 生データ禁止（ツール出力をそのまま保存しない、結論のみ）
    - GC: 完了タスク5件上限、解決済みエラー自動削除
    - インメモリ + JSON永続化（.pixie_notes/state_board.json）
    """

    DEFAULT_PATH = ""  # _resolve_default_path() で動的に設定
    MAX_COMPLETED_TASKS = 5
    MAX_KNOWLEDGE_ITEMS = 15
    MAX_ACTIVE_ERRORS = 5

    def __init__(self, file_path: str = None):
        self.goal: str = ""
        self.current_step: str = ""
        self.next_to_do: list[str] = []
        self.found_knowledge: dict[str, str] = {}
        self.completed_tasks: list[dict] = []
        self.active_errors: list[str] = []
        self.file_summaries: dict[str, str] = {}
        self.project_structure: str = ""  # /code-init で保存したプロジェクト全貌（view_tree+outline）

        # === 非同期タスク管理用フィールド ===
        self.waiting_for_async: str = ""      # "テスト実行中(PID:1234)"
        self.async_pid: int = None            # 実行中のプロセスID
        self.async_log_file: str = ""         # ログファイルパス
        self.async_timeout: int = 30          # ポーリング間隔（秒）

        self._created_at: float = time.time()
        self._updated_at: float = time.time()
        self._file_path = Path(file_path or get_data_path(".pixie_notes/state_board.json"))
        self._load()

    # ---- 目標の設定 ----

    def set_goal(self, goal: str):
        self.goal = goal.strip()
        self._updated_at = time.time()
        self._save()

    # ---- 状態の一括更新 ----

    def update(
        self,
        current_step: str = None,
        next_to_do: str = None,
        found_knowledge: str = None,
        errors: str = None,
    ):
        if current_step is not None:
            self.current_step = current_step.strip()
        if next_to_do is not None:
            self.next_to_do = [line.strip() for line in next_to_do.strip().splitlines() if line.strip()]
        if found_knowledge is not None:
            for line in found_knowledge.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    self.found_knowledge[key.strip()] = val.strip()
                else:
                    self.found_knowledge[line[:50]] = line
        if errors is not None:
            self.active_errors = [line.strip() for line in errors.strip().splitlines() if line.strip()]

        if len(self.found_knowledge) > self.MAX_KNOWLEDGE_ITEMS:
            sorted_keys = sorted(self.found_knowledge.keys())
            for key in sorted_keys[:len(self.found_knowledge) - self.MAX_KNOWLEDGE_ITEMS]:
                del self.found_knowledge[key]

        self._updated_at = time.time()
        self._save()

    # ---- タスクの完了 ----

    def complete_task(self, description: str, result: str = ""):
        self.completed_tasks.append({
            "description": description.strip(),
            "result": result.strip()[:200] if result else "",
            "completed_at": time.time(),
        })
        if len(self.completed_tasks) > self.MAX_COMPLETED_TASKS:
            self.completed_tasks = self.completed_tasks[-self.MAX_COMPLETED_TASKS:]
        self._updated_at = time.time()
        self._save()

    # ---- エラー管理 ----

    def add_error(self, error_str: str):
        self.active_errors.append(error_str.strip()[:200])
        if len(self.active_errors) > self.MAX_ACTIVE_ERRORS:
            self.active_errors = self.active_errors[-self.MAX_ACTIVE_ERRORS:]
        self._updated_at = time.time()
        self._save()

    def resolve_error(self, description: str):
        desc_lower = description.lower().strip()
        self.active_errors = [e for e in self.active_errors if desc_lower not in e.lower()]
        self._updated_at = time.time()
        self._save()

    # ---- ファイルサマリー ----

    def add_file_summary(self, file_path: str, summary: str):
        self.file_summaries[file_path] = summary.strip()[:500]
        self._updated_at = time.time()

    def search_file_summaries(self, query: str, max_results: int = 5) -> list[dict]:
        query_terms = query.lower().split()
        results = []
        for path, summary in self.file_summaries.items():
            score = 0.0
            file_name = Path(path).name.lower()
            for term in query_terms:
                if term in file_name:
                    score += 3.0
                count = summary.lower().count(term)
                if count > 0:
                    score += min(count * 0.5, 2.0)
            if score > 0:
                results.append({"file_path": path, "score": score, "summary": summary[:300]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    # ---- プロンプト注入用テキスト ----

    def to_injection_text(self, max_chars: int = 800) -> str:
        lines = ["【エージェント状態】"]

        if self.goal:
            lines.append(f"目標: {self.goal[:100]}")
        if self.current_step:
            lines.append(f"実行中: {self.current_step[:150]}")
        if self.next_to_do:
            lines.append("次のステップ:")
            for i, step in enumerate(self.next_to_do[:5], 1):
                lines.append(f"  {i}. {step[:80]}")
        if self.found_knowledge:
            lines.append("判明したこと:")
            for k, v in list(self.found_knowledge.items())[:5]:
                lines.append(f"  - {k}: {v[:60]}")
        if self.completed_tasks:
            lines.append("完了タスク:")
            for t in self.completed_tasks[-3:]:
                lines.append(f"  x {t['description'][:60]}")
        if self.active_errors:
            lines.append("未解決のエラー:")
            for e in self.active_errors[-3:]:
                lines.append(f"  ! {e[:80]}")

        if self.project_structure:
            ps = self.project_structure
            budget = 1500
            note = ""
            if len(ps) > budget:
                ps = ps[:budget]
                note = " ... (省略: 詳細は query_whiteboard/grep で)"
            lines.append(f"プロジェクト構造:{note}")
            lines.append(ps)

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n  ... (省略)"
        return text

    # ---- クエリ機能 ----

    def query(self, query_str: str) -> str:
        query_lower = query_str.lower()
        keywords = query_lower.replace("\u3000", " ").split()
        results = []

        for key, val in self.found_knowledge.items():
            score = sum(1 for kw in keywords if kw in key.lower() or kw in val.lower())
            if score > 0:
                results.append(f"[事実] {key}: {val}")

        file_results = self.search_file_summaries(query_str, max_results=3)
        for fr in file_results:
            results.append(f"[ファイル] {fr['file_path']}: {fr['summary'][:100]}...")

        for t in self.completed_tasks:
            score = sum(1 for kw in keywords if kw in t["description"].lower())
            if score > 0:
                results.append(f"[完了] {t['description']}")

        if not results:
            return f"クエリ '{query_str}' に一致する情報が見つかりませんでした。"
        return "\n".join(results[:10])

    # ---- 永続化 ----

    def _save(self):
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "goal": self.goal,
                "current_step": self.current_step,
                "next_to_do": self.next_to_do,
                "found_knowledge": self.found_knowledge,
                "completed_tasks": self.completed_tasks,
                "active_errors": self.active_errors,
                "file_summaries": self.file_summaries,
                "project_structure": self.project_structure,
                "waiting_for_async": self.waiting_for_async,
                "async_pid": self.async_pid,
                "async_log_file": self.async_log_file,
                "async_timeout": self.async_timeout,
                "created_at": self._created_at,
                "updated_at": self._updated_at,
            }
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[StateBoard] 保存エラー: {e}")

    def _load(self):
        if not self._file_path.exists():
            return
        try:
            with open(self._file_path, encoding="utf-8") as f:
                data = json.load(f)
            self.goal = data.get("goal", "")
            self.current_step = data.get("current_step", "")
            self.next_to_do = data.get("next_to_do", [])
            self.found_knowledge = data.get("found_knowledge", {})
            self.completed_tasks = data.get("completed_tasks", [])
            self.active_errors = data.get("active_errors", [])
            self.file_summaries = data.get("file_summaries", {})
            self.project_structure = data.get("project_structure", "")
            self.waiting_for_async = data.get("waiting_for_async", "")
            self.async_pid = data.get("async_pid", None)
            self.async_log_file = data.get("async_log_file", "")
            self.async_timeout = data.get("async_timeout", 30)
            self._created_at = data.get("created_at", time.time())
            self._updated_at = data.get("updated_at", time.time())
        except Exception as e:
            print(f"[StateBoard] ロードエラー: {e}")

    def clear(self):
        self.current_step = ""
        self.next_to_do = []
        self.found_knowledge = {}
        self.completed_tasks = []
        self.active_errors = []
        self.file_summaries = {}
        self.project_structure = ""
        self.waiting_for_async = ""
        self.async_pid = None
        self.async_log_file = ""
        self.async_timeout = 30
        self._updated_at = time.time()
        self._save()

    def is_empty(self) -> bool:
        return (
            not self.goal
            and not self.current_step
            and not self.next_to_do
            and not self.found_knowledge
            and not self.completed_tasks
            and not self.active_errors
            and not self.project_structure
        )


# =====================================================
# ChatHistory
# =====================================================

@dataclass
class ChatHistory:
    """スライディングウィンドウ型の会話履歴。"""
    messages: list = field(default_factory=list)
    max_messages: int = 20

    def add(self, role: str, content=None, tool_call_id: str = None, tool_calls: list = None) -> None:
        msg = {"role": role}
        if content is not None:
            msg["content"] = content
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def get_messages(self, system_msg: dict = None) -> list:
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(self.messages)
        return result

    def trim(self, keep_recent: int = None) -> list:
        keep = keep_recent or self.max_messages
        if len(self.messages) <= keep:
            return []

        cut_index = len(self.messages) - keep

        # assistant+toolペアが分割されないよう安全境界を調整
        if 0 < cut_index < len(self.messages):
            msg = self.messages[cut_index]
            if msg.get("role") == "tool":
                cut_index -= 1
            elif (msg.get("role") == "assistant"
                  and msg.get("tool_calls")
                  and cut_index + 1 < len(self.messages)
                  and self.messages[cut_index + 1].get("role") == "tool"):
                while (cut_index + 1 < len(self.messages)
                       and self.messages[cut_index + 1].get("role") == "tool"):
                    cut_index += 1

        if cut_index <= 0:
            return []

        popped = self.messages[:cut_index]
        self.messages = self.messages[cut_index:]
        return popped

    def clear(self):
        self.messages = []


# =====================================================
# AgentState — ReActループ用状態オブジェクト
# =====================================================

@dataclass
class AgentState:
    """エージェントの全体状態を管理する統合オブジェクト。"""
    state_board: AgentStateBoard = field(default_factory=AgentStateBoard)
    chat_history: ChatHistory = field(default_factory=ChatHistory)
    phase: str = "IDLE"
    tool_call_count: int = 0
    executed_actions: list = field(default_factory=list)
    loop_warn_count: int = 0
    last_response: str = ""
    exit_reason: str = ""
    max_tool_calls: int = 100
    no_tool_count: int = 0
    continuation_count: int = 0
    accumulated_content: str = ""  # length継続時の累積出力（重複結合済み）
    recent_contents: list = field(default_factory=list)
    step_count: int = 0  # ツール実行の通し番号（デバッグ表示用）
    guardrail_cooldown: int = 0  # ガードレール発火後のクールダウン（反復イテレーション数）
    thinking_notes: list = field(default_factory=list)  # 直近ターンの<think>末尾抽出（deep思考の引き継ぎ用）
    _was_deep: bool = False  # ヒステリシス: 一度deepに入ったらshallowに戻さない
    force_tool_choice: str | None = None  # 次回 node_plan 呼び出しでのみ tool_choice を上書き（例: "required"）。使用後は消費されnode_plan側でNoneに戻る。
    failure_signals: list = field(default_factory=list)  # ターン中の失敗信号（fast gate検出・ガードレール発火・異常exit_reasonの要約文字列）。教訓ストアのreflectionトリガー判定に使う（lessons.py）。

    def reset_for_new_turn(self):
        self.tool_call_count = 0
        self.executed_actions = []
        self.loop_warn_count = 0
        self.last_response = ""
        self.exit_reason = ""
        self.phase = "ROUTING"
        self.no_tool_count = 0
        self.continuation_count = 0
        self.accumulated_content = ""
        self.recent_contents = []
        self.step_count = 0
        self.guardrail_cooldown = 0
        self.thinking_notes = []
        self._was_deep = False
        self.force_tool_choice = None
        self.failure_signals = []


# =====================================================
# プロンプトインジェクター
# =====================================================

def build_system_prompt(base_prompt: str) -> str:
    """システムプロンプトを組み立てる（prefix cache 安定化版）。

    旧実装は state_board / ホワイトボード要約などの動的コンテキストを
    base_prompt の前（Middle層）に置く「サンドイッチ構造」だったが、
    system メッセージの内容が毎ターン変わるため llama.cpp の prefix cache
    （KVキャッシュ再利用）が全壊していた。

    system メッセージは base_prompt のみを含む静的な内容とし、
    動的コンテキスト（state_board・ホワイトボード・JITヒント・budget_hint・
    deep_hint・定期リマインダー等）は engine.py 側で「動的 suffix」として
    直近のユーザーメッセージ末尾に追記する方式に変更した
    （Lost in the Middle対策としての recency bias は、末尾配置でも同様に効く）。
    """
    return base_prompt
