"""code_index.py のテスト（pure ast・モデル不要）。"""

import os
from pathlib import Path

from code_index import build_index, find_dead_symbols, get_symbol_range, summarize


def _make_fixture(root: Path) -> None:
    (root / "mod_a.py").write_text(
        "from mod_b import helper\n"
        "\n"
        "@register_tool(name='x')\n"
        "def live_tool():\n"
        "    return helper()\n"
        "\n"
        "def dead_func():\n"
        "    return 42\n",
        encoding="utf-8",
    )
    (root / "mod_b.py").write_text(
        "def helper():\n"
        "    return 1\n"
        "\n"
        "class C:\n"
        "    def method(self):\n"
        "        return self.other()\n"
        "    def other(self):\n"
        "        return 0\n",
        encoding="utf-8",
    )
    (root / "entry.py").write_text(
        "from mod_a import live_tool\n"
        "\n"
        "def main():\n"
        "    live_tool()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (root / "broken.py").write_text("def (\n", encoding="utf-8")


def test_build_index_symbol_ranges(tmp_path):
    _make_fixture(tmp_path)
    index = build_index(str(tmp_path))
    syms = {s["name"]: s for s in index["files"]["mod_a.py"]["symbols"]}
    assert syms["live_tool"]["lineno"] == 4
    assert syms["live_tool"]["end_lineno"] is not None
    assert syms["live_tool"]["end_lineno"] >= 5
    assert "register_tool" in syms["live_tool"]["decorators"]


def test_parse_error_isolated(tmp_path):
    _make_fixture(tmp_path)
    index = build_index(str(tmp_path))
    assert index["files"]["broken.py"]["parse_error"] is not None
    assert index["stats"]["n_parse_errors"] == 1
    assert len(index["files"]["mod_a.py"]["symbols"]) > 0


def test_dead_code_detection(tmp_path):
    _make_fixture(tmp_path)
    index = build_index(str(tmp_path))
    candidates = find_dead_symbols(index, include_dynamic_string_check=False)
    names = {c["qualname"] for c in candidates}
    assert "dead_func" in names
    assert "live_tool" not in names   # @register_tool シード
    assert "main" not in names         # __main__ シード
    assert "helper" not in names       # live_tool から到達


def test_summarize_compact(tmp_path):
    _make_fixture(tmp_path)
    index = build_index(str(tmp_path))
    text = summarize(index)
    assert isinstance(text, str)
    assert len(text) < 12000
    assert "モジュール" in text


def test_cache_invalidation_on_content_change(tmp_path):
    cache = str(tmp_path / "code_index.json")
    _make_fixture(tmp_path)
    i1 = build_index(str(tmp_path), cache_path=cache)
    assert i1["stats"]["reparsed"] == 4  # 初回は4ファイル全解析
    # 内容変更 → 当該ファイルのみ再解析
    p = tmp_path / "mod_a.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    i2 = build_index(str(tmp_path), cache_path=cache, force=False)
    assert i2["stats"]["reparsed"] >= 1
    assert i2["stats"]["reused_from_cache"] >= 3
    # mtime のみ変更（内容同じ）→ md5 同じなので再利用
    os.utime(tmp_path / "mod_b.py", None)
    i3 = build_index(str(tmp_path), cache_path=cache, force=False)
    assert i3["stats"]["reparsed"] == 0


def test_cache_version_mismatch_cold_rebuild(tmp_path):
    cache = str(tmp_path / "code_index.json")
    Path(cache).write_text('{"schema_version":"0","root":"x","files":{}}', encoding="utf-8")
    _make_fixture(tmp_path)
    index = build_index(str(tmp_path), cache_path=cache, force=False)
    assert index["stats"]["reused_from_cache"] == 0
    assert index["stats"]["reparsed"] == 4


def test_get_symbol_range_and_unknown(tmp_path):
    _make_fixture(tmp_path)
    index = build_index(str(tmp_path))
    rng = get_symbol_range(index, str(tmp_path / "mod_b.py"), "method")
    assert rng is not None
    assert rng[0] >= 1
    assert get_symbol_range(index, str(tmp_path / "mod_b.py"), "nonexistent") is None


def test_call_graph_cross_file_edge(tmp_path):
    _make_fixture(tmp_path)
    index = build_index(str(tmp_path))
    callees = index["call_graph"].get("entry.py::main", [])
    assert "live_tool" in callees
