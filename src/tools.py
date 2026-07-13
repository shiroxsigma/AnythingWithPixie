"""互換シム（物理パッケージ化）: フラット名 import を pixie_core パッケージの実体へ委譲する。

engine 等のコアは src/pixie_core/ に物理移動した。AWP の CLI・テストは従来どおり
`import <name>` / `from <name> import ...` を使うが、本シムが sys.modules を実体
(pixie_core.<name>) にエイリアスすることでモジュール同一性を保つ
（TOOL_REGISTRY / ContextVar / monkeypatch("<name>.X") が実体に当たる）。
"""
import importlib as _il
import sys as _sys

_sys.modules[__name__] = _il.import_module("pixie_core." + __name__)
