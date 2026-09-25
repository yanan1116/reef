"""Serve mode: a local sidecar that agents treat as their model endpoint.

Agents point their OpenAI-compatible ``base_url`` here and stay
reef-oblivious; the sidecar owns the bookkeeping scenarios otherwise
re-implement per example (see ``examples/openclawrl/personal/stamp_proxy.py``
for the hand-rolled version this subsumes):

* **Session stamping** — the harness declares the session boundary out of
  band; the sidecar stamps it onto forwarded calls matching a predicate.
* **Trajectory capture** — chat completions are recorded (request, response,
  reef receipt) into a bounded in-memory store, drainable over HTTP.
* **SSE in both directions** — upstream streams pass through chunk by chunk
  while an accumulator rebuilds the completion for capture; with
  ``force_non_stream`` the upstream call is buffered and the completion is
  re-synthesized as a stream for clients that asked for one.

    python -m reef_client.serve --listen 29100 --upstream http://127.0.0.1:29000

Control endpoints (never forwarded):

    POST   /_session {"id": "..."}   declare one global session boundary
                                     (legacy mode: bare paths ride it;
                                     sequential sessions only)
    POST   /_sessions {"id": "..."}  create an addressable session: the
                                     response ``url`` (a ``/s/{id}`` prefix)
                                     is the agent's base_url root, so
                                     concurrent sessions self-identify by
                                     path and never share state
    GET    /_sessions                list addressable session ids
    DELETE /_sessions/{id}           close a session
    GET    /_captures                snapshot captured turns as JSON
    GET    /_sessions/{id}/captures  snapshot one session's turns
    DELETE /_captures                clear the capture store

Everything else is forwarded to upstream. A ``/s/{id}/...`` path has the
prefix stripped before forwarding, and the stamped session is the routed id.
With no prefix the stamped session is the globally declared one, so agents
whose base_url is fixed (e.g. a static config file) keep working unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from reef_client.sse import SSEAccumulator, synthesize_sse_events

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",  # always recomputed for the forwarded body
}

RECEIPT_HEADER = "x-reef-agent-record-id"

SESSION_PREFIX = "/s/"
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._~-]+")


def has_tools(body: dict[str, Any] | None) -> bool:
    """Main-loop heuristic: agent-loop calls carry tool definitions; harness
    side tasks (title generation, compression summaries, ...) share the
    endpoint but must not register as turns."""
    return bool(body) and "tools" in body


@dataclass
class CapturedTurn:
    session_id: str
    path: str
    status: int
    request: dict[str, Any] | None
    response: dict[str, Any] | None
    receipt: str | None
    client_stream: bool
    duration_s: float
    at: float = field(default_factory=time.time)


class CaptureStore:
    """Thread-safe bounded store of captured turns."""

    def __init__(self, limit: int = 10000) -> None:
        self._turns: deque[CapturedTurn] = deque(maxlen=limit)
        self._lock = threading.Lock()

    def add(self, turn: CapturedTurn) -> None:
        with self._lock:
            self._turns.append(turn)

    def snapshot(self, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(turn) for turn in self._turns if session_id is None or turn.session_id == session_id]

    def clear(self) -> int:
        with self._lock:
            count = len(self._turns)
            self._turns.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)


class SessionRegistry:
    """Addressable sessions: each registered id owns a ``/s/{id}`` URL prefix."""

    def __init__(self) -> None:
        self._created: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str) -> None:
        with self._lock:
            self._created.setdefault(session_id, time.time())

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._created

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._created.pop(session_id, None) is not None

    def list(self) -> list[str]:
        with self._lock:
            return sorted(self._created)


@dataclass
class ServeConfig:
    upstream: str
    listen_host: str = "127.0.0.1"
    listen_port: int = 29100
    # Session stamping: when a session is declared, stamp this header onto
    # forwarded calls for which ``stamp_when(body, method)`` is true.
    session_header: str | None = None
    stamp_when: Callable[[dict[str, Any] | None, str], bool] = lambda body, method: method == "POST"
    # Static headers added to every forwarded call (client-sent values win):
    # e.g. reef's x-reef-scenario / authorization when fronting reef directly.
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    # Static headers that REPLACE client-sent values. OpenAI-compatible
    # clients always send their own ``Authorization: Bearer <api_key>``, so a
    # shim that owns the upstream's auth must override, never setdefault.
    override_headers: Mapping[str, str] = field(default_factory=dict)
    # Buffer upstream calls (strip stream/stream_options) and re-synthesize
    # SSE for streaming clients — required when the upstream needs the
    # complete response for capture (e.g. reef's training block).
    force_non_stream: bool = False
    capture_paths: tuple[str, ...] = ("/v1/chat/completions",)
    capture_limit: int = 10000
    timeout_s: float = 7200.0

    def __post_init__(self) -> None:
        if not self.upstream:
            raise ValueError("upstream must be non-empty")
        if urlsplit(self.upstream).scheme not in ("http", "https"):
            raise ValueError(f"upstream must be an http or https URL, got {self.upstream!r}")


def build_handler(config: ServeConfig, store: CaptureStore) -> type[BaseHTTPRequestHandler]:
    parsed = urlsplit(config.upstream)
    state: dict[str, str] = {"session_id": ""}
    registry = SessionRegistry()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        #: Set once a write to the agent fails; the relay then drains the
        #: upstream in silence so the turn is still captured.
        _client_gone = False
        #: Set once the SSE response head has been committed to the agent —
        #: from then on a failure travels as an SSE error event, not a status.
        _sse_head_sent = False

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        def _reply_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write_to_client(body)

        def _route(self) -> tuple[str, str | None]:
            """Split the request target into (forward_path, routed session id)."""
            split = urlsplit(self.path)
            path = split.path
            session_id = None
            if path.startswith(SESSION_PREFIX):
                session_id, _, remainder = path[len(SESSION_PREFIX) :].partition("/")
                path = "/" + remainder
            if split.query:
                path += f"?{split.query}"
            return path, session_id or None

        def _dispatch(self) -> None:
            forward_path, routed = self._route()
            if routed is not None and routed not in registry:
                self._send_error_json(404, f"unknown session {routed!r}; POST /_sessions first")
                return
            self._forward(forward_path, routed)

        def do_POST(self) -> None:
            if self.path == "/_session":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with lock:
                    state["session_id"] = str(body.get("id", ""))
                self._reply_json(200, {"ok": True, "id": state["session_id"]})
                return
            if self.path == "/_sessions":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                session_id = str(body.get("id", "")).strip()
                if not _SESSION_ID_RE.fullmatch(session_id):
                    self._reply_json(400, {"error": {"message": "id must match [A-Za-z0-9._~-]+"}})
                    return
                registry.create(session_id)
                host = self.headers.get("Host", f"{config.listen_host}:{config.listen_port}")
                path = f"{SESSION_PREFIX}{session_id}"
                self._reply_json(200, {"id": session_id, "path": path, "url": f"http://{host}{path}"})
                return
            self._dispatch()

        def do_GET(self) -> None:
            if self.path == "/_captures":
                self._reply_json(200, {"turns": store.snapshot()})
                return
            if self.path == "/_sessions":
                self._reply_json(200, {"sessions": registry.list()})
                return
            if self.path.startswith("/_sessions/") and self.path.endswith("/captures"):
                session_id = self.path[len("/_sessions/") : -len("/captures")]
                self._reply_json(200, {"turns": store.snapshot(session_id)})
                return
            self._dispatch()

        def do_DELETE(self) -> None:
            if self.path == "/_captures":
                self._reply_json(200, {"cleared": store.clear()})
                return
            if self.path.startswith("/_sessions/"):
                session_id = self.path[len("/_sessions/") :]
                if registry.delete(session_id):
                    self._reply_json(200, {"deleted": session_id})
                else:
                    self._reply_json(404, {"error": {"message": f"unknown session {session_id!r}"}})
                return
            self._send_error_json(404, "unknown control endpoint")

        def _send_error_json(self, status: int, message: str) -> None:
            with contextlib.suppress(Exception):
                self._reply_json(status, {"error": {"message": message}})

        def _write_to_client(self, data: bytes) -> bool:
            """Write to the agent, tolerating an agent that has gone away.

            A client that times out, is killed, or simply stops reading is an
            ordinary event for a sidecar, not a fault: the exchange upstream
            already happened and its capture is what the harness came for.
            Letting the write raise loses that capture and prints a traceback
            for something nobody can act on, so the disconnect is recorded and
            the relay is allowed to finish.
            """
            if self._client_gone:
                return False
            try:
                self.wfile.write(data)
                return True
            except (BrokenPipeError, ConnectionResetError):
                self._client_gone = True
                return False

        def _write_chunk(self, data: bytes) -> None:
            self._write_to_client(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _begin_sse(self) -> None:
            """Commit the SSE response head before reef has answered.

            Agents often sit behind a transparent egress proxy whose deadline
            for a response head is seconds (harbor's gost sidecar: 15s), while
            reef legitimately stalls for minutes — a weight publish pauses the
            engine mid-update. A streaming client asked for SSE, so the head
            carries no information worth waiting for; sending it immediately
            makes the head latency a property of this process, not of GPU
            load. The price: the receipt header cannot ride a streamed reply
            (it is not known yet) — it stays available in the capture.
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.flush()
            self._sse_head_sent = True

        def _stream_error(self, message: str, status: int = 502) -> None:
            """After the head is committed, an error is an event, not a status.

            An OpenAI-style client parses the error payload and raises — a far
            better failure than the cut socket it would otherwise see.
            """
            event = {"error": {"message": message, "type": "reef_client_serve", "code": status}}
            self._write_chunk(f"data: {json.dumps(event)}\n\n".encode())
            self._write_to_client(b"0\r\n\r\n")

        def _forward(self, forward_path: str, routed_session: str | None) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else None
            body_json: dict[str, Any] | None = None
            if raw:
                try:
                    body_json = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body_json = None

            client_stream = bool(body_json and body_json.get("stream") is True)
            include_usage = bool(body_json and (body_json.get("stream_options") or {}).get("include_usage"))

            forward_body = raw
            if body_json is not None and config.force_non_stream:
                stripped = {key: value for key, value in body_json.items() if key not in ("stream", "stream_options")}
                forward_body = json.dumps(stripped).encode()

            headers = {key: value for key, value in self.headers.items() if key.lower() not in _HOP_BY_HOP}
            with lock:
                declared = state["session_id"]
            session_id = routed_session if routed_session is not None else declared
            stamped = ""
            if session_id and config.session_header and config.stamp_when(body_json, self.command):
                headers.setdefault(config.session_header, session_id)
                stamped = session_id
            for name, value in config.extra_headers.items():
                headers.setdefault(name, value)
            if config.override_headers:
                headers = {
                    name: value
                    for name, value in headers.items()
                    if name.lower() not in {key.lower() for key in config.override_headers}
                }
                headers.update(config.override_headers)
            if forward_body is not None:
                headers["Content-Length"] = str(len(forward_body))

            # The head goes out before reef is even contacted: from here on
            # nothing between the agent and this process can time the response
            # head out, no matter how long the engine stalls.
            if client_stream:
                self._begin_sse()

            # The upstream's own scheme: an https upstream (the API platform) over a plain connection is
            # answered with a redirect and an empty body, which the agent sees as an empty completion.
            if parsed.scheme == "https":
                conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                    parsed.hostname, parsed.port, timeout=config.timeout_s
                )
            else:
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=config.timeout_s)
            started = time.monotonic()
            try:
                conn.request(self.command, forward_path, body=forward_body, headers=headers)
                response = conn.getresponse()
            except Exception as exc:
                if self._sse_head_sent:
                    self._stream_error(f"reef-client serve: {exc}")
                else:
                    self._send_error_json(502, f"reef-client serve: {exc}")
                conn.close()
                return

            try:
                self._relay(response, forward_path, body_json, stamped, client_stream, include_usage, started)
            finally:
                conn.close()

        def _relay(
            self,
            response: http.client.HTTPResponse,
            forward_path: str,
            request_json: dict[str, Any] | None,
            session_id: str,
            client_stream: bool,
            include_usage: bool,
            started: float,
        ) -> None:
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            receipt = response_headers.get(RECEIPT_HEADER)
            is_sse = "text/event-stream" in response_headers.get("content-type", "")
            capture = forward_path in config.capture_paths and request_json is not None

            if is_sse:
                accumulator = self._relay_stream(response)
                receipt = accumulator.agent_record_id or receipt
                if capture:
                    store.add(
                        CapturedTurn(
                            session_id,
                            forward_path,
                            response.status,
                            request_json,
                            accumulator.completion,
                            receipt,
                            client_stream,
                            time.monotonic() - started,
                        )
                    )
                self._write_to_client(b"0\r\n\r\n")
                return

            payload = response.read()
            duration = time.monotonic() - started
            completion: dict[str, Any] | None = None
            try:
                completion = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                completion = None
            if capture:
                store.add(
                    CapturedTurn(
                        session_id,
                        forward_path,
                        response.status,
                        request_json,
                        completion,
                        receipt,
                        client_stream,
                        duration,
                    )
                )

            if self._sse_head_sent:
                # The head is committed: an upstream failure becomes an error
                # event, a buffered completion is re-synthesized as SSE.
                if response.status >= 400 or completion is None:
                    detail = payload.decode("utf-8", "replace")[:2048] or "empty upstream response"
                    self._stream_error(detail, status=response.status)
                    return
                for event in synthesize_sse_events(
                    completion,
                    include_usage=include_usage,
                    agent_record_id=receipt,
                ):
                    self._write_chunk(event.encode())
                self._write_to_client(b"0\r\n\r\n")
                return

            if not client_stream or response.status >= 400:
                # Buffered relay; upstream errors keep their status verbatim.
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() in _HOP_BY_HOP:
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self._write_to_client(payload)
                return

            # The client asked for a stream but the upstream answered a
            # buffered completion (force_non_stream): re-synthesize SSE.
            if completion is None:
                self._send_error_json(502, "reef-client serve: upstream returned non-JSON for a streaming request")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for event in synthesize_sse_events(
                completion,
                include_usage=include_usage,
                agent_record_id=receipt,
            ):
                self._write_chunk(event.encode())
            self._write_to_client(b"0\r\n\r\n")

        def _relay_stream(self, response: http.client.HTTPResponse) -> SSEAccumulator:
            if not self._sse_head_sent:
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() in _HOP_BY_HOP:
                        continue
                    self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            accumulator = SSEAccumulator()
            while True:
                chunk = response.read(16384)
                if not chunk:
                    break
                accumulator.feed(chunk)
                self._write_chunk(chunk)
            accumulator.finish()
            return accumulator

    return Handler


def run(config: ServeConfig, store: CaptureStore | None = None) -> None:
    store = store or CaptureStore(config.capture_limit)
    server = ThreadingHTTPServer((config.listen_host, config.listen_port), build_handler(config, store))
    print(
        f"reef-client serve listening on {config.listen_host}:{config.listen_port} -> {config.upstream}",
        flush=True,
    )
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", type=int, default=29100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--upstream", default="http://127.0.0.1:29000")
    parser.add_argument("--session-header", default=None, help="header stamped with the declared session id")
    parser.add_argument(
        "--stamp-tools-only",
        action="store_true",
        help="stamp only main agent-loop calls (requests carrying tool definitions)",
    )
    parser.add_argument("--force-non-stream", action="store_true", help="buffer upstream; synthesize SSE for clients")
    parser.add_argument("--no-capture", action="store_true", help="disable trajectory capture")
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()

    stamp_when = None
    if args.stamp_tools_only:
        stamp_when = lambda body, method: method == "POST" and has_tools(body)  # noqa: E731
    config = ServeConfig(
        upstream=args.upstream,
        listen_host=args.host,
        listen_port=args.listen,
        session_header=args.session_header,
        **({"stamp_when": stamp_when} if stamp_when else {}),
        force_non_stream=args.force_non_stream,
        capture_paths=() if args.no_capture else ("/v1/chat/completions",),
        timeout_s=args.timeout,
    )
    run(config)


if __name__ == "__main__":
    main()
