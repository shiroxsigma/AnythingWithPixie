"""
AnythingPixie — 教訓ストア（経験メモリ・自己進化機構）

セッションをまたいで失敗経験を「教訓」として蓄積し、次回以降の関連タスクで
engine.py の _build_dynamic_suffix() から動的suffixとして自動注入する。

設計ルール（state.py の AgentStateBoard に準拠）:
- 永続化: .pixie_notes/lessons.json（paths.get_project_data_path 経由）
- GC: 上限件数超過時は「hit_count が低く古い」ものから削除
- 重複統合: 教訓文の単語 Jaccard 類似度が閾値以上なら新規追加せず既存の hit_count を+1

依存: paths のみ（標準ライブラリ + プロジェクト最下層モジュール）。
config / LLM 呼び出しには一切依存しない — reflection（LLM呼出）は engine.py 側の責務。
"""

import json
import re
import time
import uuid
from pathlib import Path

from paths import get_project_data_path

#: 教訓文の重複統合判定に使う Jaccard 類似度の閾値。
JACCARD_DUP_THRESHOLD: float = 0.6

#: 教訓文の保存時の最大文字数（1文・100字程度目安。安全弁として少し余裕を持たせる）。
LESSON_TEXT_MAX_CHARS: int = 160

#: 教訓レコード1件あたりの trigger_keywords 保持上限。
MAX_KEYWORDS_PER_LESSON: int = 10


def _tokenize(text: str) -> set[str]:
    """軽量トークン化: 英数字トークン + 日本語文字bi-gram。

    tools.py の score_tools / state.py の search_file_summaries と同様の
    キーワードマッチ方式（LLM不使用・決定的）。
    """
    text = (text or "").lower()
    tokens = set(re.findall(r"[\w]+", text))
    jp_chars = re.findall(r"[一-鿿぀-ゟ゠-ヿ]", text)
    for i in range(len(jp_chars) - 1):
        tokens.add(jp_chars[i] + jp_chars[i + 1])
    return tokens


def _jaccard(text_a: str, text_b: str) -> float:
    """2つのテキスト間の単語 Jaccard 類似度。"""
    ta, tb = _tokenize(text_a), _tokenize(text_b)
    if not ta or not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


class LessonStore:
    """教訓（過去の失敗経験からの学び）を永続化・検索するストア。

    レコード形式: {id, lesson, trigger_keywords, source, created_at, hit_count, last_used_at}
    """

    def __init__(self, file_path: str = None, max_items: int = 50):
        self.max_items = max_items
        self._file_path = Path(file_path or get_project_data_path(".pixie_notes/lessons.json"))
        self.lessons: list[dict] = []
        self._load()

    # ---- 追加（重複統合つき） ----

    def add(self, lesson: str, trigger_keywords: list[str] = None, source: str = "") -> dict | None:
        """教訓を追加する。既存教訓と類似度 >= JACCARD_DUP_THRESHOLD なら統合する。

        Returns:
            追加または統合された教訓レコード。lesson が空なら None。
        """
        lesson = (lesson or "").strip()
        if not lesson:
            return None
        if len(lesson) > LESSON_TEXT_MAX_CHARS:
            lesson = lesson[:LESSON_TEXT_MAX_CHARS]

        keywords = [k.strip() for k in (trigger_keywords or []) if k and str(k).strip()]
        keywords = keywords[:MAX_KEYWORDS_PER_LESSON]

        # 重複判定: 既存教訓文との Jaccard 類似度
        for existing in self.lessons:
            if _jaccard(lesson, existing.get("lesson", "")) >= JACCARD_DUP_THRESHOLD:
                existing["hit_count"] = existing.get("hit_count", 0) + 1
                existing["last_used_at"] = time.time()
                if keywords:
                    merged = list(dict.fromkeys(existing.get("trigger_keywords", []) + keywords))
                    existing["trigger_keywords"] = merged[:MAX_KEYWORDS_PER_LESSON]
                self._save()
                return existing

        record = {
            "id": uuid.uuid4().hex[:12],
            "lesson": lesson,
            "trigger_keywords": keywords,
            "source": source or "",
            "created_at": time.time(),
            "hit_count": 0,
            "last_used_at": None,
        }
        self.lessons.append(record)
        self._gc()
        self._save()
        return record

    # ---- 検索（キーワードスコアリング） ----

    def recall(self, query_text: str, max_results: int = 3) -> list[dict]:
        """query_text に関連度の高い教訓を上位 max_results 件返す。

        スコアリング: trigger_keywords が query に部分文字列一致(+3) / トークン一致(+1個)、
        教訓文自体のトークン一致(+0.5個)。ヒットした教訓の hit_count/last_used_at を更新する。
        """
        if not query_text or not self.lessons:
            return []
        q_tokens = _tokenize(query_text)
        query_lower = query_text.lower()

        scored = []
        for rec in self.lessons:
            score = 0.0
            for kw in rec.get("trigger_keywords", []):
                kw_l = (kw or "").lower().strip()
                if not kw_l:
                    continue
                if kw_l in query_lower:
                    score += 3.0
                score += len(_tokenize(kw) & q_tokens) * 1.0
            score += len(_tokenize(rec.get("lesson", "")) & q_tokens) * 0.5
            if score > 0:
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [rec for _, rec in scored[:max_results]]

        if results:
            now = time.time()
            for rec in results:
                rec["hit_count"] = rec.get("hit_count", 0) + 1
                rec["last_used_at"] = now
            self._save()
        return results

    # ---- プロンプト注入用テキスト ----

    def to_injection_text(self, query_text: str, max_chars: int = 600, max_results: int = 3) -> str:
        """recall 結果を「【過去の教訓】\\n- ...」形式に整形する。空なら "" を返す。"""
        results = self.recall(query_text, max_results=max_results)
        if not results:
            return ""

        header = "【過去の教訓】"
        lines = [header]
        total = len(header)
        for rec in results:
            line = f"- {rec.get('lesson', '')}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    # ---- GC ----

    def _gc(self):
        if len(self.lessons) <= self.max_items:
            return
        # hit_count が低く古い(last_used_at/created_at が小さい)ものから削除
        ordered = sorted(
            self.lessons,
            key=lambda r: (r.get("hit_count", 0), r.get("last_used_at") or r.get("created_at", 0)),
        )
        excess = len(self.lessons) - self.max_items
        to_remove_ids = {r["id"] for r in ordered[:excess]}
        self.lessons = [r for r in self.lessons if r["id"] not in to_remove_ids]

    # ---- 永続化 ----

    def _save(self):
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump({"lessons": self.lessons}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[LessonStore] 保存エラー: {e}")

    def _load(self):
        if not self._file_path.exists():
            return
        try:
            with open(self._file_path, encoding="utf-8") as f:
                data = json.load(f)
            self.lessons = data.get("lessons", [])
        except Exception as e:
            print(f"[LessonStore] ロードエラー: {e}")
            self.lessons = []


# =====================================================
# グローバルシングルトン（毎ターンの再ロードを避ける）
# =====================================================

_singleton: LessonStore | None = None


def get_lesson_store() -> LessonStore:
    """LessonStore のプロセス内シングルトンを返す（初回のみディスクからロード）。"""
    global _singleton
    if _singleton is None:
        try:
            from config import LESSONS_MAX_ITEMS
            max_items = LESSONS_MAX_ITEMS
        except Exception:
            max_items = 50
        _singleton = LessonStore(max_items=max_items)
    return _singleton


def reset_lesson_store() -> None:
    """シングルトンを破棄する（主にテスト用）。次回 get_lesson_store() で再ロードされる。"""
    global _singleton
    _singleton = None
