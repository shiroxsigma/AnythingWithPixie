"""Transport failures must terminate turns without disguising partial work as success."""

import http.client
import io
import json
import urllib.error

import pytest

from pixie_core import llm_client
from pixie_core.turn_control import TurnStopped


class Response:
    def __init__(self, *, lines=(), body=b"", failure=None):
        self.lines = lines
        self.body = body
        self.failure = failure
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def __iter__(self):
        yield from self.lines
        if self.failure is not None:
            raise self.failure

    def read(self):
        if self.failure is not None:
            raise self.failure
        return self.body


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(llm_client.LMStudioBackend, "_fetch_n_ctx", lambda _: 32768)
    return llm_client.LMStudioBackend("http://localhost:8098/v1", model="bonsai2-27b")


def sse(payload):
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def assert_terminal_error(chunks, detail):
    failures = [chunk for chunk in chunks if chunk.get("__llm_error__")]
    assert len(failures) == 1
    assert failures[0] is chunks[-1]
    assert detail in failures[0]["__llm_error__"]
    assert failures[0]["choices"][0]["finish_reason"] == "error"
    assert sum(choice.get("finish_reason") == "error"
               for chunk in chunks for choice in chunk["choices"]) == 1


@pytest.mark.parametrize("failure", [
    TimeoutError("header timeout"),
    urllib.error.URLError("connection refused"),
    http.client.RemoteDisconnected("closed during headers"),
])
def test_header_failure_yields_one_error_without_retry(backend, monkeypatch, failure):
    calls = []

    def open_response(*args):
        calls.append(args)
        raise failure

    monkeypatch.setattr(llm_client, "_open_completion", open_response)
    assert_terminal_error(list(backend.create_chat_completion([])), str(failure))
    assert len(calls) == 1


@pytest.mark.parametrize("status", [400, 422, 500])
@pytest.mark.parametrize("failure", [TimeoutError("retry timeout"), urllib.error.URLError("offline")])
def test_retry_open_failure_is_terminal(backend, monkeypatch, status, failure):
    requests = []
    error_body = io.BytesIO(b"unsupported")

    def open_response(request, _timeout):
        requests.append(json.loads(request.data))
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, status, "error", {}, error_body)
        raise failure

    monkeypatch.setattr(llm_client, "_open_completion", open_response)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    chunks = list(backend.create_chat_completion([], thinking_budget_tokens=10))
    assert_terminal_error(chunks, str(failure))
    assert len(requests) == 2
    assert error_body.closed
    if status in {400, 422}:
        assert "thinking_budget_tokens" not in requests[1]
        assert backend._thinking_budget_supported is False
    else:
        assert requests[0] == requests[1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", [
    TimeoutError("body timeout"),
    ConnectionResetError("reset"),
    http.client.IncompleteRead(b"partial", 10),
])
def test_read_failure_closes_response_without_retry(backend, monkeypatch, stream, failure):
    partial = {"choices": [{"delta": {"content": "Partial answer"}}]}
    response = Response(lines=[sse(partial)], failure=failure)
    opens = []

    def open_response(*args):
        opens.append(args)
        return response

    monkeypatch.setattr(llm_client, "_open_completion", open_response)
    chunks = list(backend.create_chat_completion([], stream=stream))
    assert_terminal_error(chunks, type(failure).__name__)
    assert response.closed
    assert len(opens) == 1
    if stream:
        assert chunks[0] == partial


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("payload", [b"not json", b"[]", b"{}", b'{"choices":[null]}'])
def test_malformed_response_is_terminal(backend, monkeypatch, stream, payload):
    response = Response(body=payload, lines=[b"data: " + payload, b"data: [DONE]"])
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    assert_terminal_error(list(backend.create_chat_completion([], stream=stream)), "Error")
    assert response.closed


@pytest.mark.parametrize("delta", [
    None, [], {"content": {}}, {"reasoning_content": 123}, {"tool_calls": [None]},
    {"tool_calls": [{"function": {"name": "read_file"}}]},
    {"tool_calls": [{"index": [], "function": {}}]},
    {"tool_calls": [{"index": 0, "function": None}]},
    {"tool_calls": [{"index": 0, "function": {"arguments": {"path": "x"}}}]},
])
def test_malformed_stream_delta_is_rejected_before_reaching_engine(backend, monkeypatch, delta):
    response = Response(lines=[sse({"choices": [{"delta": delta}]}), b"data: [DONE]"])
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    assert_terminal_error(list(backend.create_chat_completion([])), "Invalid completion response")


@pytest.mark.parametrize("lines", [
    [sse({"error": {"message": "context exceeded"}})],
    [b"event: error\n", sse({"message": "context exceeded"})],
    [sse({"choices": [{"delta": {}, "finish_reason": "error"}]})],
])
def test_server_error_produces_one_terminal_marker(backend, monkeypatch, lines):
    response = Response(lines=lines)
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    assert_terminal_error(list(backend.create_chat_completion([])), "Server")
    assert response.closed


def test_eof_after_partial_output_is_terminal(backend, monkeypatch):
    response = Response(lines=[sse({"choices": [{"delta": {"content": "unfinished"}}]})])
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    assert_terminal_error(list(backend.create_chat_completion([])), "Connection closed before completion")


def test_overall_timeout_produces_one_terminal_marker(backend, monkeypatch):
    response = Response(lines=[sse({"choices": [{"delta": {"content": "late"}}]})])
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    clock = iter([0.0, backend.overall_timeout + 1])
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: next(clock))
    assert_terminal_error(list(backend.create_chat_completion([])), "overall timeout")
    assert response.closed


@pytest.mark.parametrize("phase", ["open", "read", "retry"])
def test_cancellation_propagates_without_conversion_or_extra_retry(backend, monkeypatch, phase):
    cancellation = TurnStopped("cancelled")
    response = Response(failure=cancellation)
    opens = []

    def open_response(*args):
        opens.append(args)
        if phase == "read":
            return response
        if phase == "retry" and len(opens) == 1:
            raise urllib.error.HTTPError("url", 400, "unsupported", {}, io.BytesIO())
        raise cancellation

    monkeypatch.setattr(llm_client, "_open_completion", open_response)
    with pytest.raises(TurnStopped) as caught:
        list(backend.create_chat_completion([], thinking_budget_tokens=10))
    assert caught.value is cancellation
    assert len(opens) == (2 if phase == "retry" else 1)
    if phase == "read":
        assert response.closed


def test_http_error_body_timeout_preserves_status_and_closes_body(backend, monkeypatch):
    class BrokenBody(io.BytesIO):
        def read(self, *_):
            raise TimeoutError("error body timeout")

    body = BrokenBody()

    def open_response(*_):
        raise urllib.error.HTTPError("url", 401, "unauthorized", {}, body)

    monkeypatch.setattr(llm_client, "_open_completion", open_response)
    chunks = list(backend.create_chat_completion([]))
    assert_terminal_error(chunks, "HTTP 401")
    assert "error body timeout" in chunks[0]["__llm_error__"]
    assert body.closed


def test_valid_stream_preserves_tool_calls_and_timings(backend, monkeypatch):
    partial = {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": '{"path":"x"}'}}
    ]}}]}
    final = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "timings": {"prompt_n": 10}}
    response = Response(lines=[sse(partial), sse(final), sse({"choices": [], "usage": {}}), b"data:[DONE]"])
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    assert list(backend.create_chat_completion([])) == [partial, final]
    assert backend.last_timings == final["timings"]
    assert response.closed


def test_nonstream_preserves_reasoning_and_timings(backend, monkeypatch):
    payload = {"choices": [{"message": {"role": "assistant", "content": "answer", "reasoning_content": "thinking"},
                            "finish_reason": "stop"}], "timings": {"prompt_n": 12}}
    response = Response(body=json.dumps(payload).encode())
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    chunks = list(backend.create_chat_completion([], stream=False))
    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "thinking"
    assert chunks[0]["choices"][0]["delta"]["content"] == "answer"
    assert backend.last_timings == payload["timings"]
    assert response.closed


def test_closing_generator_closes_stream(backend, monkeypatch):
    response = Response(lines=[sse({"choices": [{"delta": {"content": "partial"}}]})])
    monkeypatch.setattr(llm_client, "_open_completion", lambda *_: response)
    completion = backend.create_chat_completion([])
    next(completion)
    completion.close()
    assert response.closed
