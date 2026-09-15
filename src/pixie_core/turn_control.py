"""Per-request cancellation and immutable execution limits, shared by nested turns."""
from __future__ import annotations

import contextvars
import http.client
import io
import socket
import threading
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass


class TurnStopped(BaseException):
    """Must pass through ordinary tool/API error handlers without retrying."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class TurnLimits:
    timeout: float = 600.0
    llm_calls: int = 32
    tool_calls: int = 100
    think_seconds: float = 90.0
    stream_timeout: float = 180.0
    read_idle_timeout: float = 30.0

    def __post_init__(self):
        import math
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Turn limits must be finite and positive")


class TurnControl:
    def __init__(self, limits: TurnLimits | None = None):
        self.limits = limits or TurnLimits()
        self.started = time.monotonic()
        self.deadline = self.started + self.limits.timeout
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self.counts = {"llm_calls": 0, "tool_calls": 0}

    def cancel(self):
        self.cancelled.set()

    def check(self):
        if self.cancelled.is_set():
            raise TurnStopped("cancelled")
        if time.monotonic() >= self.deadline:
            raise TurnStopped("turn_timeout")

    def charge(self, kind: str, amount: int = 1):
        with self._lock:
            self.check()
            if amount <= 0:
                raise ValueError("amount must be positive")
            if self.counts[kind] + amount > getattr(self.limits, kind):
                raise TurnStopped(kind + "_limit")
            self.counts[kind] += amount

    def wait(self, seconds: float):
        self.check()
        self.cancelled.wait(min(seconds, max(0, self.deadline - time.monotonic())))
        self.check()

    def snapshot(self):
        with self._lock:
            return {**self.counts, "elapsed_seconds": time.monotonic() - self.started}


active_control = contextvars.ContextVar("pixie_turn_control", default=None)


class _SocketReader(io.RawIOBase):
    def __init__(self, sock, owner):
        self.sock, self.owner = sock, owner

    def readable(self):
        return True

    def readinto(self, buffer):
        idle_deadline = time.monotonic() + self.owner.control.limits.read_idle_timeout
        while True:
            self.owner._check()
            try:
                return self.sock.recv_into(buffer)
            except TimeoutError:
                if time.monotonic() >= idle_deadline:
                    raise


class _Socket:
    def __init__(self, sock, owner):
        self.sock, self.owner = sock, owner
        sock.settimeout(0.1)

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def close(self):
        # HTTPConnection releases its reference on Connection: close before the
        # response body is consumed. ControlledResponse owns the real socket.
        pass

    def makefile(self, mode, *args, **kwargs):
        if mode != "rb":
            raise ValueError("Only binary HTTP response reads are supported")
        return io.BufferedReader(_SocketReader(self.sock, self.owner))

    def sendall(self, data):
        view = memoryview(data)
        idle_deadline = time.monotonic() + self.owner.control.limits.read_idle_timeout
        while view:
            self.owner._check()
            try:
                size = self.sock.send(view)
            except TimeoutError:
                if time.monotonic() >= idle_deadline:
                    raise
                continue
            if size == 0:
                raise ConnectionError("Connection closed while sending request")
            view = view[size:]
            idle_deadline = time.monotonic() + self.owner.control.limits.read_idle_timeout


class ControlledResponse:
    """Close the socket from a watcher to wake header/body reads immediately."""
    def __init__(self, request, control: TurnControl):
        self.control = control
        control.charge("llm_calls")
        parts = urllib.parse.urlsplit(request.full_url)
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        self.connection = cls(parts.hostname, parts.port, timeout=min(5.0, control.limits.read_idle_timeout, control.limits.timeout))
        self.finished = threading.Event()
        self.response = None
        self.sock = None
        self.failure = None
        self.deadline = min(control.deadline, time.monotonic() + control.limits.stream_timeout)
        self.watcher = threading.Thread(target=self._watch, daemon=True)
        self.watcher.start()
        try:
            self.connection.connect()
            self.sock = self.connection.sock
            self.connection.sock = _Socket(self.sock, self)
            self._check()
            self.connection.request(request.get_method(), urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, "")),
                                    body=request.data, headers=dict(request.header_items()))
            self.response = self.connection.getresponse()
            self._check()
            if self.response.status >= 400:
                status, reason, headers = self.response.status, self.response.reason, self.response.headers
                # The caller reads/closes this object just like urllib's HTTPError.
                raise urllib.error.HTTPError(request.full_url, status, reason, headers, self)
        except urllib.error.HTTPError:
            raise
        except BaseException:
            self.close()
            self._check()
            raise

    def _check(self):
        self.control.check()
        if time.monotonic() >= self.deadline:
            raise TurnStopped("stream_timeout")
        if self.failure:
            raise TurnStopped(self.failure)

    def _watch(self):
        while not self.finished.wait(0.05):
            if self.control.cancelled.is_set() or time.monotonic() >= self.deadline:
                self.failure = "cancelled" if self.control.cancelled.is_set() else (
                    "turn_timeout" if time.monotonic() >= self.control.deadline else "stream_timeout")
                sock = self.sock or self.connection.sock
                if sock:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return

    def read(self, *args):
        try:
            return self.response.read(*args)
        finally:
            self._check()

    def __iter__(self):
        while True:
            try:
                line = self.response.readline()
            finally:
                self._check()
            if not line:
                return
            yield line

    def close(self):
        self.finished.set()
        if self.response:
            self.response.close()
        self.connection.close()
        if self.sock:
            self.sock.close()
        if self.watcher is not threading.current_thread():
            self.watcher.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
