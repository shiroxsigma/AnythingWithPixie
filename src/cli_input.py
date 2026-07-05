"""
AnythingPixie — CLI 入力モジュール

prompt_toolkit が利用可能なら Claude Code 風のリッチ入力
（Enter=送信、Ctrl+J/Alt+Enter/Shift+Enter/\\+Enter=改行、スラッシュコマンド補完、
入力履歴）を提供し、未導入時は input() + ''' ヒアドキュメント にフォールバックする。

境界は create_chat_input_session() が返すオブジェクトの get_chat_input() に集約し、
main.py 側はどちらが動いているか意識しない（duck-typing で同一インターフェース）。

依存: なし（prompt_toolkit は optional）
"""

import html as _html
import importlib.util

# llm_client.py / main.py:1098 の capture 慣用句と同じ「optional 依存の存在判定」。
_has_prompt_toolkit = importlib.util.find_spec("prompt_toolkit") is not None


# スラッシュコマンドの補完候補（単一ソース）。
# main.py のコマンド判定部（run_cli_chat 内）と整合させること。
# test_slash_commands_covers_implemented で「補完リスト ⊇ 実装コマンド」を検証。
SLASH_COMMANDS = [
    "/think", "/deep", "/review", "/verify", "/review_loop", "/step",
    "/mem", "/debug", "/reset", "/context", "/recap",
    "/code-init", "/code", "/trace", "/api", "/delegate-api", "/poll_async",
    "/pack", "/manga",
]
# コマンドではないが補完候補に入れておくと便利。
EXIT_WORDS = ["quit", "exit"]


# =====================================================
# Fallback: input() + """ ヒアドキュメント（main.py 旧ロジック移植）
# =====================================================

class _FallbackChatInput:
    """prompt_toolkit 未導入時の入力。

    multiline=True かつ入力が ''' で始まる場合は、閉じ ''' まで複数行を集めて結合する
    （main.py:664-676 の旧ロジックそのまま）。
    """

    def get_chat_input(self, prompt: str = "You: ", *, multiline: bool = True) -> str:
        user_input = input(prompt)
        if multiline and user_input.startswith('"""'):
            lines = [user_input[3:]]  # remove opening """
            if lines[0].rstrip().endswith('"""') and len(lines[0].rstrip()) > 3:
                # 1行で完結: """内容"""
                user_input = lines[0].rstrip()[:-3]
            else:
                while True:
                    line = input("... ")
                    if line.rstrip().endswith('"""') and len(line.rstrip()) > 3:
                        lines.append(line.rstrip()[:-3])
                        break
                    lines.append(line)
                user_input = "\n".join(lines)
        return user_input


# =====================================================
# Rich: prompt_toolkit 版
# =====================================================

if _has_prompt_toolkit:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    def _make_key_bindings() -> KeyBindings:
        """Enter=送信、改行キーを返す。

        multiline=True の PromptSession に対し Enter を送信に割り当てる（Claude Code と同仕様）。
        改行は全ターミナルで確実な Ctrl+J / Esc→Enter / 末尾\\+Enter で担保する。

        ※ Shift+Enter は prompt_toolkit のキー名でバインドできない（"s-enter" は Invalid key）。
           Kitty Keyboard Protocol 対応ターミナル + 端末個別設定が必要なため標準ではバインドしない。
        """
        kb = KeyBindings()

        def insert_newline(event):
            event.current_buffer.insert_text("\n")

        # 改行: Ctrl+J と Esc→Enter は全ターミナルで確実。
        # （多くのターミナルで Alt+Enter は Esc + Enter と同じバイト列になるため、
        #   この escape+enter バインディングで捕捉できる）
        kb.add("c-j")(insert_newline)
        kb.add("escape", "enter")(insert_newline)    # Esc → Enter（2ストローク・Alt+Enter 相当）

        @kb.add("enter")
        def _enter(event):
            # 末尾が \ なら改行挿入（\ を残す: Claude Code の \+Enter と同じ）。
            # それ以外は送信。
            buf = event.current_buffer
            if buf.text.endswith("\\"):
                buf.insert_text("\n")
            else:
                buf.validate_and_handle()

        return kb

    class ChatInputSession:
        """prompt_toolkit 利用時の入力セッション。ループ外で1つ保持して使い回す。

        - FileHistory: 入力履歴を history_path に永続化（↑/↓ で呼び出し）。
        - WordCompleter: バッファ先頭でのみスラッシュコマンドを補完（sentence=True）。
        - mouse_support=False: ツール承認メニュー（msvcrt）との raw mode 衝突を避ける。
        """

        def __init__(self, history_path: str | None = None):
            self._history = FileHistory(history_path) if history_path else InMemoryHistory()
            self._completer = WordCompleter(
                SLASH_COMMANDS + EXIT_WORDS,
                ignore_case=True,
                match_middle=False,
                sentence=True,
            )
            self._bindings = _make_key_bindings()
            # PromptSession は Windows コンソールバッファ等の「実際のコンソール」を要求する。
            # テストやコンソール無し環境での構築失敗を避けるため、初回 get_chat_input
            # まで生成を遅延する（履歴/補完/キーバインドはここで準備済み）。
            self._session = None

        def _ensure_session(self):
            if self._session is None:
                self._session = PromptSession(
                    history=self._history,
                    completer=self._completer,
                    key_bindings=self._bindings,
                    multiline=True,        # Enter=送信をバインディングで上書きするため multiline 表示。
                    mouse_support=False,
                )
            return self._session

        def get_chat_input(self, prompt: str = "You: ", *, multiline: bool = True) -> str:
            # multiline 引数は互換性のため受けるが、Enter=送信モデルなので表示は常に multiline。
            # （末尾 \ や Ctrl+J 等で明示的に改行しない限り1行で送信される）
            formatted = self._build_prompt(prompt)
            return self._ensure_session().prompt(formatted)

        @staticmethod
        def _build_prompt(prompt: str):
            """[Planning]/[System] 等のブラケットプレフィックスをシアン着色したプロンプト。"""
            esc = _html.escape(prompt)
            for tag in ("[Planning]", "[System]"):
                esc = esc.replace(tag, f"<ansicyan>{tag}</ansicyan>")
            # 遅延 import でモジュールトップの import ブロックを汚さない
            from prompt_toolkit.formatted_text import HTML
            return HTML(esc)


# =====================================================
# Factory
# =====================================================

def create_chat_input_session(history_path: str | None = None):
    """prompt_toolkit があれば ChatInputSession、なければ _FallbackChatInput を返す。

    どちらも get_chat_input(prompt, *, multiline) を持つ（duck-typing）。
    """
    if _has_prompt_toolkit:
        return ChatInputSession(history_path=history_path)
    return _FallbackChatInput()
