"""pixie_core API 1.5（思考許容時間の実行時変更）の安全網。

- set_think_budget/get_think_budget: deep モードの <think> 上限秒を実行中に変更する。
  engine が config から自モジュールへ束縛した DEEP_THINK_BUDGET_SEC を差し替えるのが要点
  （config 側を書き換えても engine は見ないため、そこを取り違えると静かに無効化される）。
- Engine.set_stream_timeout: セッションの LLM ストリーム打ち切り秒（思考を長くするなら
  こちらも併せて延ばす必要がある）。
"""
import pytest

import pixie_core
from pixie_core import _api

_SERVER = {"base_url": "http://localhost:1/v1", "model": "test-model"}


@pytest.fixture(autouse=True)
def restore_budget():
    """モジュール変数を書き換えるテストなので、既定値に戻してから次へ渡す。"""
    import engine as _engine
    original = _engine.DEEP_THINK_BUDGET_SEC
    yield
    _engine.DEEP_THINK_BUDGET_SEC = original


def test_set_think_budget_rebinds_engine_global():
    import engine as _engine
    assert _api.set_think_budget(240) == 240
    assert _engine.DEEP_THINK_BUDGET_SEC == 240
    assert _api.get_think_budget() == 240


def test_set_think_budget_accepts_numeric_string():
    assert _api.set_think_budget("120") == 120


@pytest.mark.parametrize("bad", [4, 0, -1, "abc", None])
def test_set_think_budget_rejects_invalid(bad):
    import engine as _engine
    before = _engine.DEEP_THINK_BUDGET_SEC
    with pytest.raises(ValueError):
        _api.set_think_budget(bad)
    assert _engine.DEEP_THINK_BUDGET_SEC == before  # 失敗時は据え置き


def test_public_api_exposes_think_budget():
    """遅延ロードの facade（__init__）経由でも引けること（_PUBLIC 追加漏れの検出）。"""
    assert pixie_core.set_think_budget is _api.set_think_budget
    assert pixie_core.get_think_budget is _api.get_think_budget


def test_engine_set_stream_timeout(tmp_path):
    eng = pixie_core.create_engine(_SERVER, str(tmp_path))
    eng.set_stream_timeout(600)
    assert eng.context.llm.overall_timeout == 600.0
    idle_before = eng.context.llm.read_idle_timeout
    eng.set_stream_timeout(300, read_idle_timeout=45)
    assert eng.context.llm.overall_timeout == 300.0
    assert eng.context.llm.read_idle_timeout == 45.0 != idle_before


def test_engine_set_stream_timeout_without_llm(tmp_path):
    """llm 未接続の Engine でも例外にしない（設定適用のループが壊れない）。"""
    eng = pixie_core.create_engine(_SERVER, str(tmp_path))
    eng.context.llm = None
    eng.set_stream_timeout(600)  # 例外が出ないこと
