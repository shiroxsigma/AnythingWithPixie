"""用途別ツールパックの汎用登録・可視性を検証する。"""

import pytest

from registry import TOOL_REGISTRY, get_active_tool_names, register_tool
from toolpacks import AVAILABLE_PACKS, load_pack


def test_register_tool_without_pack_is_core_tool():
    name = "_test_dummy_core_tool"
    try:
        register_tool(name=name, description="d",
                      schema={"type": "object", "properties": {}, "required": []})(lambda: "ok")
        assert TOOL_REGISTRY[name]["pack"] is None
        assert name in get_active_tool_names(set())
    finally:
        TOOL_REGISTRY.pop(name, None)


def test_pack_tool_is_visible_only_when_active():
    name = "_test_dummy_pack_tool"
    try:
        register_tool(name=name, description="d",
                      schema={"type": "object", "properties": {}, "required": []},
                      pack="test_pack")(lambda: "ok")
        assert name not in get_active_tool_names(set())
        assert name in get_active_tool_names({"test_pack"})
    finally:
        TOOL_REGISTRY.pop(name, None)


def test_unknown_pack_is_rejected():
    with pytest.raises(ValueError, match="未知のツールパック"):
        load_pack("not_registered")


def test_no_builtin_pack_is_currently_registered():
    assert AVAILABLE_PACKS == frozenset()
