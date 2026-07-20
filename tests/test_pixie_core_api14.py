"""pixie_core API 1.4 の追加分（tool_set / system_suffix / load_history）の安全網。

API 1.4 は NoteWithPixie のエンジン置換（Phase 1）のための追加:
- create_engine(tool_set=...) → context.fixed_tool_set（node_plan が最優先で参照する固定プロファイル）
- create_engine(system_suffix=...) → build_system_text 出力末尾への静的追記
- Engine.load_history(messages) → 外部サイドカー履歴からの ChatHistory シード

いずれも「追加のみ・既定値では従来経路と完全一致」が後方互換の契約。
"""
import pytest

import pixie_core
from pixie_core import _api


_SERVER = {"base_url": "http://localhost:1/v1", "model": "test-model"}


def _engine(tmp_path, **kw):
    return pixie_core.create_engine(_SERVER, str(tmp_path), **kw)


def test_api_version_consistent_between_init_and_api():
    """__init__.py（即値・遅延ロード用）と _api.py の API_VERSION が一致していること。

    二重定義は遅延ロード設計上の意図的なものだが、片方だけ上げると組み込み側の
    バージョン検証が偽陰性/偽陽性になる（NWP は 1.4 以上を必須とする）。"""
    assert pixie_core.API_VERSION == _api.API_VERSION == "1.5"


# --- tool_set（固定ツールプロファイル） ---

def test_create_engine_default_has_no_fixed_tool_set(tmp_path):
    eng = _engine(tmp_path)
    assert getattr(eng.context, "fixed_tool_set", None) is None, (
        "tool_set 未指定で fixed_tool_set が設定されている（後方互換破壊: "
        "node_plan の従来分岐に入らなくなる）"
    )


def test_create_engine_tool_set_becomes_frozenset(tmp_path):
    names = {"read_file", "grep_search"}
    eng = _engine(tmp_path, tool_set=names)
    fixed = eng.context.fixed_tool_set
    assert isinstance(fixed, frozenset) and fixed == frozenset(names)


def test_fixed_tool_set_names_resolve_in_registry(tmp_path):
    """固定プロファイルの名前が registry から OpenAI tools 配列に引けること。

    node_plan の fixed_tools 分岐は registry_to_openai_tools(sorted(available_tools))
    で名前引きするため、これが通れば提示集合の強制が機能する。
    """
    from tools import registry_to_openai_tools
    names = {"read_file", "grep_search"}
    tools = registry_to_openai_tools(sorted(names))
    # registry_to_openai_tools は登録順を保つため、集合一致で検証する
    assert {t["function"]["name"] for t in tools} == names


# --- system_suffix（静的システムプロンプト追記） ---

def test_make_system_builder_empty_is_passthrough():
    assert _api._make_system_builder("") is _api.build_system_text


def test_make_system_builder_appends_suffix(monkeypatch):
    monkeypatch.setattr(_api, "build_system_text",
                        lambda context, state_board=None, **kw: "BASE")
    builder = _api._make_system_builder("SUFFIX-MARKER")
    out = builder(context=object(), state_board=None)
    assert out.startswith("BASE") and out.endswith("SUFFIX-MARKER")


def test_engine_uses_suffix_builder(tmp_path):
    eng = _engine(tmp_path, system_suffix="SUFFIX-MARKER")
    assert eng._system_builder is not _api.build_system_text
    eng2 = _engine(tmp_path)
    assert eng2._system_builder is _api.build_system_text


# --- load_history（外部履歴シード） ---

def test_load_history_seeds_chat_history(tmp_path):
    eng = _engine(tmp_path)
    eng.load_history([
        {"role": "user", "content": "こんにちは"},
        {"role": "assistant", "content": "どうも"},
        {"role": "tool", "content": "無視されるべき"},
        {"role": "user", "content": ""},  # 空 content も無視
    ])
    msgs = eng.state.chat_history.messages
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "こんにちは"), ("assistant", "どうも")]


def test_load_history_respects_trim(tmp_path):
    eng = _engine(tmp_path)
    many = []
    for i in range(30):
        many.append({"role": "user", "content": f"u{i}"})
        many.append({"role": "assistant", "content": f"a{i}"})
    eng.load_history(many)
    limit = eng.state.chat_history.max_messages
    assert len(eng.state.chat_history.messages) <= limit
    # 直近側が残る（先頭からの単純切り捨てではない）
    assert eng.state.chat_history.messages[-1]["content"] == "a29"
