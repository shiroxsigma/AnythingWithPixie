"""pixie_core — AnythingWithPixie(AWP) の ReAct エンジンを外部アプリから埋め込むための公開パッケージ。

物理構成（Phase 2 / #1 物理パッケージ化）:
    engine 等のコア14モジュールは本パッケージ配下（`pixie_core/*.py`）に物理的に置かれる。
    AWP の CLI(main.py) と 388 テストは従来どおりフラット名（`import engine` 等）で参照するが、
    `src/<name>.py` に置いた **sys.modules エイリアスシム**が `pixie_core.<name>` の実体へ委譲する
    （モジュール同一性を保つため、monkeypatch("engine.X") 等のテストも実体に当たる）。
    別プロジェクト CodeWithPixie は `import pixie_core` だけで駆動する（内部モジュール非依存）。

公開 API（`pixie_core.<name>`）:
    API_VERSION                         — 互換性チェック用（import 副作用ゼロで参照可）。
    CancelTurn                          — 協調キャンセル用例外。
    READONLY_TOOLS / DESTRUCTIVE_TOOLS  — ツール分類。
    create_engine(server, workspace)    — Engine を構築。
    Engine                              — .run_turn(user_text, *, output_fn, interactive_fn)。
    tool_count()                        — 登録ツール数（疎通スモーク）。

遅延ロード（重要）:
    `__init__` は API_VERSION 以外を即時 import しない。公開 API への初回アクセス時に `_api`
    サブモジュール（engine 等を読み込む本体）を遅延ロードする。これにより
    「シムが import pixie_core.engine → __init__ 先行実行」の局面で facade を巻き込む
    循環 import（partially initialized module）を構造的に回避する。
"""

#: 公開 API のバージョン（import 副作用ゼロで参照できるよう即値で置く）。
#: 1.1 マルチセッション／1.2 セッション別 workspace／1.3 register_tool 公開（外部ツール登録）。
API_VERSION = "1.3"

_PUBLIC = frozenset({
    "CancelTurn", "create_engine", "Engine", "tool_count",
    "READONLY_TOOLS", "DESTRUCTIVE_TOOLS", "register_tool", "get_workspace",
})


def __getattr__(name):  # PEP 562: 公開 API を初回アクセス時に _api から遅延解決する
    if name in _PUBLIC:
        from . import _api
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _PUBLIC)
