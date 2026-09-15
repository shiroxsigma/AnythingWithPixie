"""Exercise actual sockets, including cancellation before response headers."""
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pixie_core.llm_client import LMStudioBackend
from pixie_core.turn_control import TurnControl, TurnLimits, TurnStopped, active_control


@pytest.fixture
def endpoint(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                if self.path.startswith("/headers"):
                    entered.set()
                    release.wait(5)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                if self.path.startswith("/body"):
                    entered.set()
                    release.wait(5)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(LMStudioBackend, "_fetch_n_ctx", lambda _: 32768)
    yield f"http://127.0.0.1:{server.server_port}", entered, release
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(url, control):
    token = active_control.set(control)
    try:
        return list(LMStudioBackend(url).create_chat_completion([]))
    finally:
        active_control.reset(token)


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_cancel_wakes_silent_http_request_and_next_request_works(endpoint, phase):
    url, entered, _ = endpoint
    control = TurnControl()
    with ThreadPoolExecutor() as pool:
        task = pool.submit(request, url + "/" + phase, control)
        assert entered.wait(2)
        control.cancel()
        with pytest.raises(TurnStopped, match="cancelled"):
            task.result(timeout=1.5)
    assert request(url, TurnControl())[0]["choices"][0]["delta"]["content"] == "ok"


@pytest.mark.parametrize("limits, reason", [
    (TurnLimits(timeout=0.2), "turn_timeout"),
    (TurnLimits(stream_timeout=0.2), "stream_timeout"),
])
def test_deadline_interrupts_header_wait(endpoint, limits, reason):
    url, _, _ = endpoint
    with ThreadPoolExecutor() as pool:
        task = pool.submit(request, url + "/headers", TurnControl(limits))
        with pytest.raises(TurnStopped, match=reason):
            task.result(timeout=1.5)


def test_multiple_model_calls_share_the_request_limit(endpoint):
    url, _, _ = endpoint
    control = TurnControl(TurnLimits(llm_calls=1))
    request(url, control)
    with pytest.raises(TurnStopped, match="llm_calls_limit"):
        request(url, control)
    assert control.snapshot()["llm_calls"] == 1


def test_parallel_tools_cannot_overrun_budget():
    control = TurnControl(TurnLimits(tool_calls=3))
    def charge():
        try:
            control.charge("tool_calls")
            return True
        except TurnStopped:
            return False
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(lambda _: charge(), range(20)))
    assert sum(results) == 3


def test_limit_configuration_is_immutable_and_validated():
    from dataclasses import FrozenInstanceError
    limits = TurnLimits(think_seconds=15)
    with pytest.raises(FrozenInstanceError):
        limits.think_seconds = 90
    for value in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            TurnLimits(timeout=value)


def test_cancellation_before_request_does_not_connect(endpoint):
    url, entered, _ = endpoint
    control = TurnControl()
    control.cancel()
    with pytest.raises(TurnStopped):
        request(url + "/headers", control)
    assert not entered.is_set()
