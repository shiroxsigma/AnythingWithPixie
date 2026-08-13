# -*- coding: utf-8 -*-
"""ツール定義ブロック トークン比較: 4パターン"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, "src")

import tools
import code_tool
from registry import TOOL_REGISTRY


def minimal_schema(schema):
    """required 引数の型情報のみ残す最小 schema"""
    if not isinstance(schema, dict):
        return schema
    props = schema.get("properties", {})
    req = schema.get("required", [])
    mini_props = {}
    for k in req:
        if k in props and isinstance(props[k], dict):
            t = props[k].get("type")
            mini_props[k] = {"type": t} if t else {}
    return {"type": "object", "properties": mini_props, "required": req}


def clean_schema(schema):
    """引数の description のみ削除。type/enum/default/required は保持（精度リスク低）"""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    props = schema.get("properties", {})
    clean = {}
    for k, v in props.items():
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "description"}
        else:
            clean[k] = v
    out["properties"] = clean
    return out


def make(mode):
    """mode: 'full' / 'core_full' / 'all_min' / 'lean'"""
    out = []
    for name, e in TOOL_REGISTRY.items():
        is_ext = e.get("category") == "extended"
        if mode == "full":                                    # 現状: 全フル
            desc, sch = e["description"], e["schema"]
        elif mode == "core_full":                            # core=フル, ext=prompt_desc+最小schema
            if is_ext:
                desc, sch = e.get("prompt_desc") or e["description"], minimal_schema(e["schema"])
            else:
                desc, sch = e["description"], e["schema"]
        elif mode == "all_min":                              # 全ツール最小schema, desc維持
            desc, sch = e["description"], minimal_schema(e["schema"])
        elif mode == "lean":                                 # 全ツール最小schema + desc=prompt_desc
            desc, sch = e.get("prompt_desc") or e["description"], minimal_schema(e["schema"])
        elif mode == "clean":                                # 全ツール: 引数desc削除のみ(enum/default保持)
            desc, sch = e["description"], clean_schema(e["schema"])
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": sch}})
    return out


def tok(s):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(s)), "tiktoken"
    except Exception:
        return len(s) // 3, "chars/3"


n_core = sum(1 for e in TOOL_REGISTRY.values() if e.get("category") == "core")
n_ext = sum(1 for e in TOOL_REGISTRY.values() if e.get("category") == "extended")
base = tok(json.dumps(make("full"), ensure_ascii=False))[0]
method = tok("")[1]

print("=" * 60)
print(" ツール定義ブロック トークン比較")
print("=" * 60)
print(f"計測法: {method}   ツール総数: {len(TOOL_REGISTRY)} (core {n_core} / ext {n_ext})\n")
print(f"{'パターン':<34}{'tok':>8}{'削減':>8}{'削減率':>8}")
print("-" * 60)

labels = [
    ("full",      "A 現状(全フル)"),
    ("core_full", "B coreフル + ext軽量"),
    ("clean",     "E 全ツール: 引数desc削除(精度保持)"),
    ("all_min",   "C 全ツール最小schema(desc維持)"),
    ("lean",      "D 全最小schema + desc=prompt_desc"),
]
for mode, label in labels:
    s = json.dumps(make(mode), ensure_ascii=False)
    t, _ = tok(s)
    cut = base - t
    pct = 100 * cut // base if base else 0
    print(f"{label:<34}{t:>8,}{cut:>8,}{pct:>7}%")

print()
print("--- core と extended の schema サイズ（削減余地の所在）---")
core_s = sum(len(json.dumps(e["schema"], ensure_ascii=False)) for e in TOOL_REGISTRY.values() if e.get("category") == "core")
ext_s = sum(len(json.dumps(e["schema"], ensure_ascii=False)) for e in TOOL_REGISTRY.values() if e.get("category") == "extended")
core_min = sum(len(json.dumps(minimal_schema(e["schema"]), ensure_ascii=False)) for e in TOOL_REGISTRY.values() if e.get("category") == "core")
ext_min = sum(len(json.dumps(minimal_schema(e["schema"]), ensure_ascii=False)) for e in TOOL_REGISTRY.values() if e.get("category") == "extended")
print(f"core({n_core})   schema: {core_s:>5} chars → 最小化で {core_min:>5} chars  (削減 {core_s - core_min})")
print(f"ext({n_ext})  schema: {ext_s:>5} chars → 最小化で {ext_min:>5} chars  (削減 {ext_s - ext_min})")
