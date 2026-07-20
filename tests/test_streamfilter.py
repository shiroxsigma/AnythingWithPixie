"""StreamFilter (engine.py) のテスト — 思考ブロック除去フィルタ。"""

from engine import StreamFilter


def test_removes_think_block():
    sf = StreamFilter(remove_thinking=True)
    out = sf.process("<think>secret reasoning here</think>visible answer")
    assert "visible answer" in out
    assert "secret" not in out


def test_passthrough_when_disabled():
    sf = StreamFilter(remove_thinking=False)
    out = sf.process("<think>reasoning</think>answer")
    assert "reasoning" in out
    assert "answer" in out


def test_think_block_split_across_chunks():
    sf = StreamFilter(remove_thinking=True)
    out = sf.process("<think>rea") + sf.process("soning</think>answer")
    assert "answer" in out
    assert "reasoning" not in out


def test_capture_thinking():
    sf = StreamFilter(remove_thinking=True, capture_thinking=True)
    sf.process("<think>the captured thought</think>answer")
    assert "the captured thought" in sf.get_last_thought()


def test_gemma_channel_format_removed():
    sf = StreamFilter(remove_thinking=True)
    out = sf.process("<|channel>thoughtsome ideas<channel|>result")
    assert "result" in out
    assert "some ideas" not in out


def test_get_last_thought_empty_default():
    sf = StreamFilter(remove_thinking=True, capture_thinking=True)
    assert sf.get_last_thought() == ""
