import registry


def test_experimental_changeset_is_not_in_default_active_tools():
    assert "apply_search_replace_changeset" in registry.TOOL_REGISTRY
    assert "apply_search_replace_changeset" not in registry.get_active_tool_names(set())
