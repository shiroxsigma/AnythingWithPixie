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
import shutil
import subprocess
import sys
from pathlib import Path

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


def test_core_runs_without_flat_modules_or_source_tree(tmp_path):
    """配布される pixie_core だけで作業領域を読み書きできる。"""
    package_parent = tmp_path / "installed"
    shutil.copytree(Path(_SRC) / "pixie_core", package_parent / "pixie_core",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    code = """
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
workspace = Path(sys.argv[2])
package_dir = Path(sys.argv[1]) / "pixie_core"
before = {p.relative_to(package_dir) for p in package_dir.rglob("*") if p.is_file()}
import pixie_core
from pixie_core import paths, tools

assert Path(pixie_core.__file__).parent.samefile(package_dir)
assert pixie_core.API_VERSION.startswith("1.")
assert pixie_core.tool_count() > 0
assert Path(paths.get_app_root()).samefile(package_dir)
assert not (Path(paths.get_app_root()) / ".pixie_notes").exists()

engine = pixie_core.create_engine(
    {"base_url": "http://127.0.0.1:1234/v1", "model": "smoke"}, str(workspace)
)
assert Path(engine.workspace).samefile(workspace)
token = paths.bind_workspace(str(workspace))
try:
    assert Path(paths.get_project_data_path(".pixie_notes")).parent.samefile(workspace)
    target = workspace / "smoke.txt"
    result = tools.write_file(str(target), "isolated package write\\n")
    assert not result.startswith("Error:"), result
    assert target.read_text(encoding="utf-8") == "isolated package write\\n"
    assert "isolated package write" in tools.read_file(str(target))
    tools.platform.system = lambda: "Windows"
    tools.get_bundled_path = lambda name: str(workspace / "missing-rg.exe")
    tools.shutil.which = lambda name: None
    search = tools.grep_search("isolated package write", path=str(workspace), context_lines=0)
    assert "total_matches=1" in search and "Search summary (authoritative)" in search
finally:
    paths.reset_workspace(token)

assert not (Path(paths.get_app_root()) / ".pixie_notes").exists()
assert {p.relative_to(package_dir) for p in package_dir.rglob("*") if p.is_file()} == before
flat = {"main", "config", "paths", "registry", "tools", "engine", "state",
        "llm_client", "code_tool", "code_index", "lfm_tooluse"}
assert flat.isdisjoint(sys.modules), flat & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(package_parent), str(workspace)],
        cwd=tmp_path, capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stderr
