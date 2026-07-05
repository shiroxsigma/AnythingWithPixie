"""generate_behavior_prompt のゴールデンテスト — 出力バイト完全不変を保証。

リファクタ（デッドコード削除・セクション分割）前後で、生成されるシステム
プロンプトが1バイトも変わらないことを検証する。fixture は現状コードで生成済み。
"""

import hashlib
from pathlib import Path

import pytest

from config import MANGA_TOOL_SET
from tools import generate_behavior_prompt

GOLDEN_DIR = Path(__file__).parent / "golden" / "behavior_prompt"

# 9ツールセット × {shallow, deep, code} = 27ケース。
# 各セットが generate_behavior_prompt の S2-S6 分岐を網羅する。
# code モードは _CODE_MODE_POLICY（/code 専門ワークフロー）を固定。
CASES = [
    ("01_all_none", None),
    ("02_empty_set", set()),
    ("03_read_file", {"read_file"}),
    ("04_outline_read", {"get_code_outline", "read_file"}),
    ("05_research_code_paths", {"research_code_paths"}),
    ("06_update_state_read", {"update_state", "read_file"}),
    ("07_edit_full", {"search_and_replace", "write_file", "replace_lines", "run_command", "read_file"}),
    ("08_doc_stack", {"gather_project_info", "view_tree", "get_code_outline", "grep_search", "analyze_file", "read_file", "write_file"}),
    ("09_write_sections", {"write_sections", "append_to_file", "write_file"}),
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@pytest.mark.parametrize("mode", ["shallow", "deep", "code"])
@pytest.mark.parametrize("name,tools", CASES)
def test_behavior_prompt_golden(name, tools, mode):
    golden = GOLDEN_DIR / f"{name}__{mode}.txt"
    assert golden.exists(), f"fixture missing: {golden}"
    expected = golden.read_text(encoding="utf-8")
    actual = generate_behavior_prompt(available_tools=tools, thinking_mode=mode, mode=mode)
    assert actual == expected, (
        f"output drifted for {name}__{mode}: "
        f"expected sha={_sha(expected)} actual sha={_sha(actual)}"
    )


# mode="manga" 用ゴールデンケース（詳細設計 §4: 全ケース×mangaは不要・代表2ケースで十分）。
# 01_all_none は「全ツール言及可」の分岐、10_manga_toolset は実際の /manga 固定ツールセット
# (MANGA_TOOL_SET) での _MANGA_MODE_POLICY + 各セクションの組み立てを検証する。
MANGA_CASES = [
    ("01_all_none", None),
    ("10_manga_toolset", MANGA_TOOL_SET),
]


@pytest.mark.parametrize("name,tools", MANGA_CASES)
def test_behavior_prompt_golden_manga(name, tools):
    golden = GOLDEN_DIR / f"{name}__manga.txt"
    assert golden.exists(), f"fixture missing: {golden}"
    expected = golden.read_text(encoding="utf-8")
    actual = generate_behavior_prompt(available_tools=tools, thinking_mode="shallow", mode="manga")
    assert actual == expected, (
        f"output drifted for {name}__manga: "
        f"expected sha={_sha(expected)} actual sha={_sha(actual)}"
    )


@pytest.mark.parametrize("name,tools", CASES)
def test_behavior_prompt_deep_equals_shallow(name, tools):
    """system プロンプトの基本方針は thinking_mode に依存せず常に共通固定である（prefix cache 保護）。

    以前は thinking_mode=="deep" で _BASIC_POLICY_DEEP に切り替わっていたが、thinking_mode は
    ターン間で振動しうる（_was_deep は reset_for_new_turn() でリセットされる）ため、system
    メッセージ自体が変わると llama.cpp の prefix cache が全壊するリスクがあった。
    現在は thinking_mode="shallow"/"deep" で generate_behavior_prompt の出力が完全一致する
    （deep 固有の追加指示は動的 suffix 側 _build_dynamic_suffix の deep_hint に統合済み）。
    この不変条件が将来のリファクタで崩れていないことを検証する。
    """
    shallow_prompt = generate_behavior_prompt(available_tools=tools, thinking_mode="shallow", mode="shallow")
    deep_prompt = generate_behavior_prompt(available_tools=tools, thinking_mode="deep", mode="deep")
    assert deep_prompt == shallow_prompt, (
        f"{name}: thinking_mode='deep' の出力が 'shallow' と一致しない"
        "（基本方針セクションが再び thinking_mode で切り替わっていないか確認してください）"
    )
