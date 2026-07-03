"""generate_behavior_prompt のゴールデンテスト — 出力バイト完全不変を保証。

リファクタ（デッドコード削除・セクション分割）前後で、生成されるシステム
プロンプトが1バイトも変わらないことを検証する。fixture は現状コードで生成済み。
"""

import hashlib
from pathlib import Path

import pytest

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
