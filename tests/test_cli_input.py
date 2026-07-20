"""cli_input（prompt_toolkit optional / input() フォールバック）のテスト。

CI（prompt_toolkit 未導入）でもフォールバック系は green になること。
リッチ系（ChatInputSession）は prompt_toolkit 導入時のみ実行。
"""

import builtins

import pytest

from cli_input import SLASH_COMMANDS, _FallbackChatInput, _has_prompt_toolkit

# =====================================================
# 補完リストの網羅性: SLASH_COMMANDS ⊇ main.py 実装コマンド
# =====================================================

def test_slash_commands_covers_implemented():
    """main.py に現れる全スラッシュコマンドが補完リストに含まれるか。

    補完リストと実装（run_cli_chat の判定部）の乖離を検出するのが目的。
    """
    import re
    from pathlib import Path

    main_path = Path(__file__).resolve().parent.parent / "src" / "main.py"
    main_src = main_path.read_text(encoding="utf-8")
    # '/xxx' 形式の文字列リテラルを抽出（判定部の == '/think' や startswith('/code') 等）。
    found = set(re.findall(r"['\"](/[a-z][a-z0-9_-]*)['\"]", main_src))
    missing = found - set(SLASH_COMMANDS)
    assert not missing, f"実装にあるが補完リスト(SLASH_COMMANDS)に無い: {sorted(missing)}"


def test_slash_commands_nonempty_and_well_formed():
    """コマンドは全て '/' で始まり、quit/exit はコマンドではない（EXIT_WORDS 側）。"""
    from cli_input import EXIT_WORDS

    assert SLASH_COMMANDS, "SLASH_COMMANDS が空"
    assert all(c.startswith("/") for c in SLASH_COMMANDS)
    assert "quit" in EXIT_WORDS and "exit" in EXIT_WORDS
    assert not any(c in SLASH_COMMANDS for c in EXIT_WORDS)


# =====================================================
# Fallback: input() + """ ヒアドキュメント（main.py 旧ロジック移植の回帰テスト）
# =====================================================

def _stub_input(monkeypatch, answers):
    """builtins.input を、answers を順に返す stub に差し替え。"""
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))


def test_fallback_single_line(monkeypatch):
    _stub_input(monkeypatch, ["hello world"])
    assert _FallbackChatInput().get_chat_input("You: ", multiline=True) == "hello world"


def test_fallback_inline_heredoc(monkeypatch):
    """1行で完結する \"\"\"内容\"\"\" は中身だけになる。"""
    _stub_input(monkeypatch, ['"""one liner"""'])
    assert _FallbackChatInput().get_chat_input("You: ", multiline=True) == "one liner"


def test_fallback_multiline_heredoc(monkeypatch):
    """\"\"\" で始まる入力は閉じ \"\"\" まで複数行を集めて改行結合する。"""
    _stub_input(monkeypatch, ['"""line1', "line2", 'line3"""'])
    result = _FallbackChatInput().get_chat_input("You: ", multiline=True)
    assert result == "line1\nline2\nline3"


def test_fallback_multiline_disabled(monkeypatch):
    """multiline=False なら ヒアドキュメント処理をせず1行をそのまま返す。"""
    _stub_input(monkeypatch, ['"""not heredoc'])
    result = _FallbackChatInput().get_chat_input("You: ", multiline=False)
    assert result == '"""not heredoc'


def test_fallback_simulates_no_prompt_toolkit(monkeypatch):
    """_has_prompt_toolkit=False なら create_chat_input_session は _FallbackChatInput を返す。"""
    import cli_input

    monkeypatch.setattr(cli_input, "_has_prompt_toolkit", False)
    sess = cli_input.create_chat_input_session()
    assert isinstance(sess, _FallbackChatInput)


def test_fallback_eof_propagates(monkeypatch):
    """input() の EOFError はそのまま送出され、呼出側で quit 扱いできる。"""
    def raise_eof(*a, **k):
        raise EOFError()

    monkeypatch.setattr(builtins, "input", raise_eof)
    with pytest.raises(EOFError):
        _FallbackChatInput().get_chat_input("You: ")


# =====================================================
# Rich: ChatInputSession（prompt_toolkit 導入時のみ）
# =====================================================

def test_rich_session_when_available():
    """prompt_toolkit 導入時は ChatInputSession が返る（未導入時は skip）。"""
    if not _has_prompt_toolkit:
        pytest.skip("prompt_toolkit not installed")

    from cli_input import ChatInputSession, create_chat_input_session

    sess = create_chat_input_session()
    assert isinstance(sess, ChatInputSession)


def test_factory_consistency():
    """create_chat_input_session の戻り型が _has_prompt_toolkit と整合する。"""
    import cli_input

    sess = cli_input.create_chat_input_session()
    if _has_prompt_toolkit:
        from cli_input import ChatInputSession

        assert isinstance(sess, ChatInputSession)
    else:
        assert isinstance(sess, _FallbackChatInput)
