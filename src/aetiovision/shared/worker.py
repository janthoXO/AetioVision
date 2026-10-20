"""Model worker processes: torch models run outside the MCP / CLI process (local HTTP, stdlib only).

Server: `serve({"op": handler}, port)`, handler(dict) -> dict. Client: `call(url, op, payload, spawn)`
auto-starts the worker with the `spawn` argv when nothing listens on a localhost url.
numpy arrays travel as {"__npy__": base64 .npy}.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import numpy as np


def _enc(o):
    if isinstance(o, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, o, allow_pickle=False)
        return {"__npy__": base64.b64encode(buf.getvalue()).decode()}
    raise TypeError(type(o))


def _dec(d: dict):
    if "__npy__" in d:
        return np.load(io.BytesIO(base64.b64decode(d["__npy__"])), allow_pickle=False)
    return d


def dumps(o) -> bytes:
    return json.dumps(o, default=_enc).encode()


def loads(b: bytes):
    return json.loads(b, object_hook=_dec)


def serve(handlers: dict[str, Callable[[dict], dict]], port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send(200, {"ok": True, "ops": sorted(handlers)})

        def do_POST(self):
            op = self.path.strip("/")
            if op not in handlers:
                return self._send(404, {"error": f"unknown op {op}"})
            try:
                body = loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, handlers[op](body))
            except Exception as e:  # reported to the client, worker keeps running
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def _send(self, status, obj):
            data = dumps(obj)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    host = os.environ.get("AETIO_WORKER_HOST", "127.0.0.1")  # 0.0.0.0 inside a container
    print(f"worker listening on {host}:{port}, ops {sorted(handlers)}", flush=True)
    HTTPServer((host, port), H).serve_forever()  # ponytail: one request at a time, models aren't thread-safe anyway


class WorkerError(RuntimeError): ...


def call(url: str, op: str, payload: dict, spawn: list[str] | None = None, timeout: float = 900) -> dict:
    try:
        return _post(url, op, payload, timeout)
    except urllib.error.URLError as e:
        if not (spawn and isinstance(e.reason, ConnectionRefusedError) and urlparse(url).hostname in ("127.0.0.1", "localhost")):
            raise WorkerError(f"worker {url} unreachable: {e.reason}") from e
    _start(url, spawn)
    return _post(url, op, payload, timeout)


def _post(url: str, op: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"{url.rstrip('/')}/{op}", dumps(payload), {"Content-Type": "application/json"})
    try:
        return loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        raise WorkerError(f"worker {url}/{op}: {loads(e.read()).get('error', e)}") from e


def _start(url: str, spawn: list[str], wait: float = 1800) -> None:
    log = Path(tempfile.gettempdir()) / f"aetio-worker-{urlparse(url).port}.log"
    proc = subprocess.Popen(spawn, stdout=log.open("ab"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    deadline = time.time() + wait  # first start may download model weights
    while time.time() < deadline:
        if proc.poll() is not None:
            raise WorkerError(f"worker exited with {proc.returncode}, see {log}")
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise WorkerError(f"worker did not come up within {wait}s, see {log}")
