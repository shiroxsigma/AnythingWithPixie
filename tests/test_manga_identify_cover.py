"""manga_identify_cover のインターセプト処理（subagent._execute_manga_identify_cover）の単体テスト。

詳細設計 docs/design/toolpacks.md §3.6（P3: 表紙 Vision 委譲）。LLM は一切起動せず、
mock オブジェクト（create_chat_completion を差し替えたダミー）で以下を検証する:

- Vision 経路が全く無い（context.use_vision=False かつ delegate_vision=False）→ Error
- mock LLM が正常な JSON を返す → パース・整形された結果
- mock LLM が壊れた JSON / null 混在を返す → クラッシュせず適切なフォールバック
- パス検証（存在しない/画像でないパスの拒否）がインターセプト経路でも効くこと
- 経路優先順位: delegate が Vision 対応なら delegate 優先、メインへのフォールバック

`tools.resize_and_encode_image` は実ファイル読込を伴うため、isolated_manga と同様に
tmp_path 上にダミー画像を書いて実際に読ませる（PNG バイト列は base64 化できれば十分で
Pillow の有無どちらでも動く）。
"""

import json
from types import SimpleNamespace

import pytest

import toolpacks.manga as manga_mod
from subagent import _execute_manga_identify_cover

TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415478da63f8ffff3f0005fe02fe331295140000000049454e44ae"
    "426082"
)


@pytest.fixture
def isolated_manga(tmp_path, monkeypatch):
    def _patched(rel_path: str) -> str:
        return str(tmp_path / rel_path)

    # manga の生成物はプロジェクトルート基準（get_project_data_path）で解決される。
    monkeypatch.setattr(manga_mod, "get_project_data_path", _patched)
    monkeypatch.setattr(manga_mod, "get_data_path", _patched, raising=False)
    return tmp_path


@pytest.fixture
def cover_path(isolated_manga):
    covers_dir = isolated_manga / ".pixie_notes" / "manga_covers"
    covers_dir.mkdir(parents=True, exist_ok=True)
    p = covers_dir / "test_cover.jpg"
    p.write_bytes(TINY_PNG)
    # インターセプト関数には manga_scan が返す相対パス文字列の形で渡す
    return ".pixie_notes/manga_covers/test_cover.jpg"


class _FakeLLM:
    """create_chat_completion を差し替え可能な最小の LLM スタブ。

    contents にリストを渡すと呼び出し順に返す値を切り替えられる（response_format
    ありの初回呼び出しが空応答を返し、フォールバックの2回目呼び出しで正常応答を
    返す、という実環境確認で判明したシナリオを再現するため）。
    """

    def __init__(self, content: str = None, contents: list = None, raise_exc: Exception = None):
        self._contents = contents if contents is not None else [content]
        self._raise_exc = raise_exc
        self.calls = 0
        self.seen_response_formats = []

    def create_chat_completion(self, messages, **kwargs):
        self.seen_response_formats.append(kwargs.get("response_format"))
        idx = min(self.calls, len(self._contents) - 1)
        content = self._contents[idx]
        self.calls += 1
        if self._raise_exc:
            raise self._raise_exc
        return {"choices": [{"message": {"content": content}}]}


def _noop_output(*args, **kwargs):
    pass


def _make_context(llm=None, delegate_llm=None, use_vision=False, delegate_vision=False):
    return SimpleNamespace(
        llm=llm, delegate_llm=delegate_llm, use_vision=use_vision, delegate_vision=delegate_vision
    )


# =====================================================
# Vision 経路なし
# =====================================================

def test_no_vision_path_returns_error(cover_path):
    context = _make_context(llm=_FakeLLM(), use_vision=False, delegate_llm=None, delegate_vision=False)
    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    assert out.startswith("Error: Vision モデルが利用できません")


def test_delegate_present_but_not_vision_and_main_not_vision_returns_error(cover_path):
    """delegate_llm はあるが delegate_vision=False、メインも use_vision=False なら Error。"""
    context = _make_context(
        llm=_FakeLLM(), use_vision=False,
        delegate_llm=_FakeLLM(), delegate_vision=False,
    )
    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    assert out.startswith("Error: Vision モデルが利用できません")
    assert context.llm.calls == 0
    assert context.delegate_llm.calls == 0


# =====================================================
# 正常系: JSON パース・整形
# =====================================================

def test_main_vision_success_returns_parsed_json(cover_path):
    payload = json.dumps({"title": "冒険の書", "author": "山田太郎", "volume": "1", "confidence": "high"}, ensure_ascii=False)
    context = _make_context(llm=_FakeLLM(content=payload), use_vision=True, delegate_llm=None, delegate_vision=False)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data == {"title": "冒険の書", "author": "山田太郎", "volume": "1", "confidence": "high"}
    assert context.llm.calls == 1


def test_null_fields_are_preserved_as_none(cover_path):
    payload = json.dumps({"title": None, "author": None, "volume": None, "confidence": "low"})
    context = _make_context(llm=_FakeLLM(content=payload), use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data == {"title": None, "author": None, "volume": None, "confidence": "low"}


def test_empty_string_fields_are_normalized_to_none(cover_path):
    """title/author/volume が空文字列の場合も null 相当として扱う（`or None`）。"""
    payload = json.dumps({"title": "", "author": "", "volume": "", "confidence": "medium"})
    context = _make_context(llm=_FakeLLM(content=payload), use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data["title"] is None
    assert data["author"] is None
    assert data["volume"] is None
    assert data["confidence"] == "medium"


def test_invalid_confidence_value_falls_back_to_low(cover_path):
    payload = json.dumps({"title": "タイトル", "author": None, "volume": None, "confidence": "very_high"})
    context = _make_context(llm=_FakeLLM(content=payload), use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data["confidence"] == "low"


# =====================================================
# 壊れた応答: クラッシュせず Error を返す
# =====================================================

def test_broken_json_returns_error_without_crashing(cover_path):
    context = _make_context(llm=_FakeLLM(content="申し訳ありませんが画像を認識できませんでした。"), use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    assert out.startswith("Error:")
    assert "JSON解析に失敗" in out


def test_json_embedded_in_extra_text_is_recovered(cover_path):
    """前後に余計な文章がついたJSON応答も {..} の抽出で救済できる。"""
    payload = (
        "はい、分析結果は以下の通りです。\n"
        + json.dumps({"title": "回復テスト", "author": None, "volume": "3", "confidence": "medium"})
        + "\n以上です。"
    )
    context = _make_context(llm=_FakeLLM(content=payload), use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data["title"] == "回復テスト"
    assert data["volume"] == "3"


def test_llm_exception_returns_error_without_crashing(cover_path):
    context = _make_context(llm=_FakeLLM(raise_exc=RuntimeError("接続失敗")), use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    assert out.startswith("Error:")
    assert "接続失敗" in out


# =====================================================
# 実環境確認で判明: response_format(json_schema) が vision 入力と組み合わさると
# 一部バックエンドで空応答（finish_reason=stop・content=""）を返すことがある。
# その場合 response_format なしで1回だけ再試行するフォールバックを検証する。
# =====================================================

def test_empty_response_with_schema_falls_back_to_freeform_retry(cover_path):
    """1回目(response_format付き)が空応答 → 2回目(response_format無し)で正常JSONを取得。"""
    payload = json.dumps({"title": "再試行成功", "author": None, "volume": None, "confidence": "medium"})
    llm = _FakeLLM(contents=["", payload])
    context = _make_context(llm=llm, use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data["title"] == "再試行成功"
    assert llm.calls == 2
    # 1回目は response_format 指定あり、2回目は指定なし（None）であること
    assert llm.seen_response_formats[0] is not None
    assert llm.seen_response_formats[1] is None


def test_non_empty_schema_response_does_not_trigger_retry(cover_path):
    """1回目が(nullだらけでも)空でなければ、フォールバック再試行は行わない。"""
    payload = json.dumps({"title": None, "author": None, "volume": None, "confidence": "low"})
    llm = _FakeLLM(content=payload)
    context = _make_context(llm=llm, use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data == {"title": None, "author": None, "volume": None, "confidence": "low"}
    assert llm.calls == 1


def test_both_attempts_empty_returns_error(cover_path):
    """response_format 有り・無し 両方とも空応答ならクラッシュせずErrorを返す。"""
    llm = _FakeLLM(contents=["", ""])
    context = _make_context(llm=llm, use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    assert out.startswith("Error:")
    assert llm.calls == 2


# =====================================================
# パス検証（インターセプト経路でも効くこと）
# =====================================================

def test_missing_cover_path_returns_error_before_calling_llm(isolated_manga):
    context = _make_context(llm=_FakeLLM(), use_vision=True)
    missing = str(isolated_manga / ".pixie_notes" / "manga_covers" / "nope.jpg")

    out = _execute_manga_identify_cover(context, {"cover_path": missing}, _noop_output)
    assert out.startswith("Error:")
    assert "存在しません" in out
    assert context.llm.calls == 0


def test_non_image_path_rejected_before_calling_llm(isolated_manga):
    txt_path = isolated_manga / "readme.txt"
    txt_path.write_text("hello", encoding="utf-8")
    context = _make_context(llm=_FakeLLM(), use_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": str(txt_path)}, _noop_output)
    assert out.startswith("Error:")
    assert "認識できません" in out
    assert context.llm.calls == 0


# =====================================================
# 経路優先順位: delegate（Vision対応）優先 → メインへのフォールバック
# =====================================================

def test_delegate_vision_preferred_over_main(cover_path):
    """delegate_llm がVision対応なら、メインもVision対応でも delegate を使う。"""
    payload = json.dumps({"title": "委譲鯖タイトル", "author": None, "volume": None, "confidence": "high"})
    delegate = _FakeLLM(content=payload)
    main = _FakeLLM(content=json.dumps({"title": "メイン鯖タイトル", "author": None, "volume": None, "confidence": "high"}))
    context = _make_context(llm=main, use_vision=True, delegate_llm=delegate, delegate_vision=True)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data["title"] == "委譲鯖タイトル"
    assert delegate.calls == 1
    assert main.calls == 0


def test_falls_back_to_main_when_delegate_not_vision(cover_path):
    """delegate_llm はあるが Vision非対応なら、メインがVision対応であればメインを使う。"""
    payload = json.dumps({"title": "メイン鯖タイトル", "author": None, "volume": None, "confidence": "high"})
    delegate = _FakeLLM(content="dummy")
    main = _FakeLLM(content=payload)
    context = _make_context(llm=main, use_vision=True, delegate_llm=delegate, delegate_vision=False)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data["title"] == "メイン鯖タイトル"
    assert delegate.calls == 0
    assert main.calls == 1


def test_falls_back_to_main_when_no_delegate_configured(cover_path):
    """delegate_llm が None（未設定）ならメインがVision対応であればメインを使う。"""
    payload = json.dumps({"title": "単独メイン", "author": None, "volume": None, "confidence": "high"})
    main = _FakeLLM(content=payload)
    context = _make_context(llm=main, use_vision=True, delegate_llm=None, delegate_vision=False)

    out = _execute_manga_identify_cover(context, {"cover_path": cover_path}, _noop_output)
    data = json.loads(out)
    assert data["title"] == "単独メイン"
    assert main.calls == 1
