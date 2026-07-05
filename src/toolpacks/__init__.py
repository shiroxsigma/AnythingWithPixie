"""ツールパックのロードエントリポイント。

用途特化のツール群（manga / web 等）は、この機構を通じてのみ TOOL_REGISTRY に
登録される。パックモジュールは import 時に `@register_tool(..., pack=...)` を
実行するだけで、有効化（active_packs への追加）とは独立している
（詳細設計 `docs/design/toolpacks.md` §1.3 参照）。

- ロード（import）: load_pack(name) が importlib で `toolpacks.<name>` を import する。
  import 済みなら no-op（冪等）。ロードしただけでは有効化されない
  （registry.get_active_tool_names の active_packs に含まれて初めて available_tools に載る）。
- 有効化: config.json の "toolpacks" キー、または CLI の `/pack <name>` コマンド
  （main.py 側で context.active_packs に追加する）。

依存: なし（importlib のみ・標準ライブラリ）。
"""

import importlib

#: 実装済みのパック名（load_pack が受け付ける名前の一覧）。
AVAILABLE_PACKS: frozenset[str] = frozenset({"manga"})

#: ロード済みパック名（import 済みで再ロードを no-op にするための集合）。
_loaded: set[str] = set()


def load_pack(name: str) -> None:
    """パック名から `toolpacks.<name>` モジュールを import し、ツール登録を発火させる。

    2回目以降の呼び出しは _loaded による早期リターンで no-op（import 自体も
    Python の sys.modules キャッシュにより冪等だが、明示的に管理する）。

    Args:
        name: パック名（例: "manga"）。AVAILABLE_PACKS にない名前は ValueError。
    """
    if name in _loaded:
        return
    if name not in AVAILABLE_PACKS:
        raise ValueError(f"未知のツールパック: '{name}'（利用可能: {sorted(AVAILABLE_PACKS)}）")
    importlib.import_module(f"toolpacks.{name}")
    _loaded.add(name)


def is_loaded(name: str) -> bool:
    """パックが既にロード（import）済みかを返す。"""
    return name in _loaded
