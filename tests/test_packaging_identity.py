"""pixie_core 物理パッケージ化の安全網。

フラット構成でも パッケージ構成でも成立する不変条件を固定する。特に:
- get_app_root() が AWP ルート（pyproject.toml のある所）を指すこと
  → paths.py を pixie_core/ へ移動したときに __file__ 逆算がズレる事故を検出する。
- registry がステートフルな単一モジュールであること（TOOL_REGISTRY / ContextVar / __getattr__）。
- `import <flat>` と `import pixie_core.<flat>` がパッケージ化後に同一オブジェクトであること
  （エイリアスシムが同一性を保つ＝monkeypatch("engine.X") 等が実体に当たる前提）。
- engine 先 / pixie_core 先 のどちらの import 順序でも成功すること（__init__ 循環の検出）。
"""
import importlib
import os
import subprocess
import sys

import pytest

_AWP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_AWP_ROOT, "src")

_CORE_MODULES = ["engine", "engine_helpers", "state", "registry", "tools", "code_tool",
                 "code_index", "llm_client", "subagent", "shadow_verify", "lessons",
                 "trajectory", "config", "paths"]


def test_get_app_root_points_at_awp_root():
    import paths
    root = paths.get_app_root()
    assert os.path.isfile(os.path.join(root, "pyproject.toml")), (
        f"get_app_root()={root!r} が AWP ルート（pyproject.toml のある所）を指していない。"
        " paths.py の __file__ 逆算が移動でズレた可能性。"
    )


def test_registry_is_stateful_singleton():
    import registry
    registry.set_state_board("SB_MARKER")
    assert registry._state_board == "SB_MARKER"          # PEP562 __getattr__ 経由
    reg2 = importlib.import_module("registry")
    assert reg2 is registry
    assert reg2.TOOL_REGISTRY is registry.TOOL_REGISTRY   # 単一 dict
    registry.set_state_board(None)                        # 後始末


def test_flat_and_package_identity_when_packaged():
    import pixie_core
    if not hasattr(pixie_core, "__path__"):
        pytest.skip("pixie_core はまだパッケージではない（移動前）")
    for name in _CORE_MODULES:
        flat = importlib.import_module(name)
        pkg = importlib.import_module(f"pixie_core.{name}")
        assert flat is pkg, f"{name}: フラットと pixie_core.{name} が別オブジェクト（同一性破壊）"


def _run(code: str):
    env = {**os.environ, "PYTHONPATH": _SRC, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_import_order_engine_first():
    r = _run("import engine; import pixie_core; print(pixie_core.tool_count())")
    assert r.returncode == 0, f"engine 先 import が失敗:\n{r.stderr}"


def test_import_order_pixie_core_first():
    r = _run("import pixie_core; import engine; print(pixie_core.tool_count())")
    assert r.returncode == 0, f"pixie_core 先 import が失敗:\n{r.stderr}"


def test_public_api_lazy_smoke():
    r = _run("import pixie_core; assert pixie_core.API_VERSION.startswith('1.'); "
             "assert pixie_core.tool_count() > 0; "
             "assert pixie_core.CancelTurn is not None; "
             "assert len(pixie_core.DESTRUCTIVE_TOOLS) > 0")
    assert r.returncode == 0, f"公開API の疎通に失敗:\n{r.stderr}"
