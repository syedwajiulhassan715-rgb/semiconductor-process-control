"""Local web server for the replay monitor. Standard library only.

One :class:`~secom.monitor.ReplaySession` is shared by every request behind a
lock. Playback is paced by the page, which calls ``/api/step`` on a timer, so
the server holds no clock and every response is a deterministic function of
the requests before it.

Endpoints (all JSON)::

    GET  /api/design           fixed design: p0, p1, h, g-chart and watchlist limits
    GET  /api/state?since=K    state plus history points from index K
    POST /api/step   {"n": 5}             advance n runs
    POST /api/seek   {"run": 400}         replay from the start to run 400
    POST /api/reset  {}                   back to run 0, acknowledgements cleared
    POST /api/ack    {"alarm_id": 1, "note": "..."}

``POST`` responses carry the same body as ``GET /api/state``; pass ``since`` in
the request body to receive only new history points.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

from .monitor import AlarmError, ReplaySession, phase_spans

#: Upper bound on runs per /api/step call, so one request can't stall the lock.
MAX_STEP = 2000


def _clean(obj: Any) -> Any:
    """Replace NaN/inf with None so the output is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def _index_html() -> bytes:
    return resources.files("secom").joinpath("monitor_static/index.html").read_bytes()


class _ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _int(body: dict[str, Any], key: str, default: int | None = None) -> int:
    value = body.get(key, default)
    if value is None:
        raise _ApiError(HTTPStatus.BAD_REQUEST, f"missing {key!r}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise _ApiError(HTTPStatus.BAD_REQUEST, f"{key!r} must be an integer") from None


class MonitorServer(ThreadingHTTPServer):
    """HTTP server bound to one replay session."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        session: ReplaySession,
        *,
        speed: float = 20.0,
        quiet: bool = True,
    ) -> None:
        self.session = session
        self.speed = speed
        self.quiet = quiet
        self.lock = threading.Lock()
        super().__init__(address, _Handler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        host = str(host)
        return f"http://{'127.0.0.1' if host in ('', '0.0.0.0') else host}:{port}/"


class _Handler(BaseHTTPRequestHandler):
    server: MonitorServer
    server_version = "secom-monitor"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if not self.server.quiet:
            super().log_message(format, *args)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(_clean(payload), allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json")

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "body is not valid JSON") from None
        if not isinstance(data, dict):
            raise _ApiError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        return data

    def _dispatch(self, routes: dict[str, Callable[[], Any]]) -> None:
        path = urlparse(self.path).path
        handler = routes.get(path)
        try:
            if handler is None:
                raise _ApiError(HTTPStatus.NOT_FOUND, f"no route {path}")
            result = handler()
        except _ApiError as exc:
            self._json({"error": str(exc)}, exc.status)
            return
        except AlarmError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        if isinstance(result, bytes):
            self._send(HTTPStatus.OK, result, "text/html; charset=utf-8")
        else:
            self._json(result)

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        session, lock = self.server.session, self.server.lock

        def state() -> dict[str, Any]:
            since = _int({"since": query.get("since", ["0"])[0]}, "since")
            with lock:
                return session.snapshot(since)

        def design() -> dict[str, Any]:
            d = session.design.as_dict()
            d["n_runs"] = len(session.runs)
            d["phases"] = phase_spans(session.runs)
            d["default_speed"] = self.server.speed
            return d

        self._dispatch(
            {
                "/": _index_html,
                "/index.html": _index_html,
                "/api/design": design,
                "/api/state": state,
            }
        )

    def do_POST(self) -> None:
        session, lock = self.server.session, self.server.lock

        def with_body(action: Callable[[dict[str, Any]], None]) -> Callable[[], Any]:
            def run() -> dict[str, Any]:
                body = self._body()
                since = _int(body, "since", 0)
                with lock:
                    action(body)
                    return session.snapshot(since)

            return run

        def step(body: dict[str, Any]) -> None:
            n = _int(body, "n", 1)
            if not 0 <= n <= MAX_STEP:
                raise _ApiError(HTTPStatus.BAD_REQUEST, f"'n' must be in [0, {MAX_STEP}]")
            session.step(n)

        def seek(body: dict[str, Any]) -> None:
            session.seek(_int(body, "run"))

        def reset(body: dict[str, Any]) -> None:
            session.reset()

        def ack(body: dict[str, Any]) -> None:
            note = body.get("note", "")
            if not isinstance(note, str) or len(note) > 500:
                raise _ApiError(
                    HTTPStatus.BAD_REQUEST, "'note' must be a string of at most 500 chars"
                )
            session.acknowledge(_int(body, "alarm_id"), note.strip())

        self._dispatch(
            {
                "/api/step": with_body(step),
                "/api/seek": with_body(seek),
                "/api/reset": with_body(reset),
                "/api/ack": with_body(ack),
            }
        )


def serve(
    session: ReplaySession,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    speed: float = 20.0,
    quiet: bool = True,
) -> MonitorServer:
    """Create a server; call ``serve_forever()`` on it (or run it in a thread)."""
    return MonitorServer((host, port), session, speed=speed, quiet=quiet)
