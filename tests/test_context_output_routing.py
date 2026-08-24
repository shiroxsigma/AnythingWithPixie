"""コンテキスト圧縮の通知が埋め込み先callbackへ流れることを検証する。"""

import engine


def test_hard_trim_routes_notice_and_whiteboard_output_to_callback(monkeypatch):
    monkeypatch.setattr(engine, "estimate_tokens", lambda _llm, text: len(text))
    observed = {"whiteboard_callback": None}

    def _whiteboard(_llm, _messages, output_fn=None):
        observed["whiteboard_callback"] = output_fn

    monkeypatch.setattr(engine, "_update_whiteboard", _whiteboard)
    output = []

    def _output(text="", **_kwargs):
        output.append(text)

    messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": f"message-{index}-" + "x" * 40}
        for index in range(6)
    ]

    trimmed = engine.check_and_trim_context(
        object(), messages, max_context=80, output_fn=_output
    )

    assert len(trimmed) < 7
    assert "コンテキスト上限" in "".join(output)
    assert observed["whiteboard_callback"] is _output
