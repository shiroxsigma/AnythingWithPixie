# -*- coding: utf-8 -*-
"""実装後の検証: ツール定義トークン削減 + golden ずれの原因確認"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, "src")

# [1] 実装後のツール定義トークン（registry_to_openai_tools に clean_schema 統合済み）
import tools, code_tool
from tools import registry_to_openai_tools
s = json.dumps(registry_to_openai_tools(), ensure_ascii=False)
cur = len(s) // 3
print(f"[1] 実装後 全ツール(clean_schema適用): {len(s)} chars ≈ {cur:,} tok")
print(f"    A現状(4,799 tok) から {4799 - cur:,} tok 削減 ({100*(4799-cur)//4799}%)")
print(f"    ※ E案目標(3,650 tok) との差: {cur - 3650:+,} tok")

# [2] golden ずれの確認（generate_behavior_prompt は本変更で未編集）
from tools import generate_behavior_prompt
from pathlib import Path
print("\n[2] behavior_prompt golden テスト差分確認:")
print("    (generate_behavior_prompt は本変更で未編集 → ずれは既存)")
for name, tools in [("01_all_none", None), ("02_empty_set", set()), ("07_edit_full", {"search_and_replace","write_file","replace_lines","run_command","read_file"})]:
    g = Path(f"tests/golden/behavior_prompt/{name}__shallow.txt")
    if not g.exists():
        print(f"  {name}: golden ファイル不在")
        continue
    exp = g.read_text(encoding="utf-8")
    act = generate_behavior_prompt(available_tools=tools, thinking_mode="shallow", mode="shallow")
    status = "✓ 一致" if exp == act else "✗ 不一致"
    print(f"  {name}__shallow: {status}  (exp={len(exp)} chars, act={len(act)} chars)")
    if exp != act:
        for i, (a, b) in enumerate(zip(exp, act)):
            if a != b:
                lo = max(0, i - 20)
                print(f"      first diff @ {i}: exp=...{exp[lo:i+15]!r}...")
                print(f"                          act=...{act[lo:i+15]!r}...")
                break
        if len(exp) != len(act):
            print(f"      長さ差: {len(act) - len(exp):+d} chars")
