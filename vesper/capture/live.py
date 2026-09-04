"""Live frame server: the drone-camera side channel for in-progress runs.

RunCapture publishes every frame here (when VESPER_LIVE_PORT is set); a tiny
threaded HTTP server serves, per stream:

    GET  /streams           JSON: {"run": <id>, "streams": ["overview", "fpv"]}
    GET  /<stream>.jpg      latest frame, single JPEG
    GET  /<stream>.mjpeg    multipart/x-mixed-replace stream (~10 fps)

The warm session (scripts/warm_session.py) uses the same server for two more
things, so Operations gets an AO map and instant tasking:

    GET  /state             JSON snapshot the sim loop publishes each step
                            (drone positions, targets, found/reached, t)
    POST /command           enqueue a command for the sim loop to apply on its
                            own thread (reset / deploy) -- body is JSON

Pure stdlib + PIL; runs inside the sim process on the GPU box. The browser UI
points <img> tags at the streams and polls /state for the map.
"""
import io
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

_BOUNDARY = "vesperframe"


class LiveFrameServer:
    def __init__(self, port: int, run_id: str = ""):
        self.port = port
        self.run_id = run_id
        self._jpeg: dict[str, bytes] = {}
        self._seq: dict[str, int] = {}
        self._state: dict = {}
        self._commands: "queue.Queue[dict]" = queue.Queue()
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._start()

    def publish(self, rgb: np.ndarray, stream: str) -> None:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=75)
        with self._lock:
            self._jpeg[stream] = buf.getvalue()
            self._seq[stream] = self._seq.get(stream, 0) + 1

    def set_state(self, state: dict) -> None:
        """Called by the sim loop each step with the world snapshot for /state."""
        with self._lock:
            self._state = state

    def drain_commands(self) -> list[dict]:
        """Called by the sim loop; returns queued commands to apply on its thread."""
        out = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                return out

    def _start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep sim stdout clean
                pass

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_POST(self):
                path = self.path.split("?")[0].strip("/")
                if path != "command":
                    self.send_response(404); self.end_headers(); return
                n = int(self.headers.get("Content-Length", 0))
                try:
                    cmd = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    self.send_response(400); self.end_headers(); return
                server._commands.put(cmd)
                self._json({"queued": cmd.get("kind", "?")})

            def do_GET(self):
                path = self.path.split("?")[0].strip("/")
                if path == "streams":
                    with server._lock:
                        self._json({"run": server.run_id, "streams": sorted(server._jpeg)})
                    return
                if path == "state":
                    with server._lock:
                        self._json(server._state)
                    return
                name, dot, kind = path.rpartition(".")[0], ".", path.rpartition(".")[2]
                if kind == "jpg":
                    with server._lock:
                        data = server._jpeg.get(name)
                    if not data:
                        self.send_response(404); self.end_headers(); return
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if kind == "mjpeg":
                    self.send_response(200)
                    self._cors()
                    self.send_header(
                        "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
                    self.end_headers()
                    last = -1
                    try:
                        while True:
                            with server._lock:
                                data = server._jpeg.get(name)
                                seq = server._seq.get(name, 0)
                            if data and seq != last:
                                last = seq
                                self.wfile.write(
                                    f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                    f"Content-Length: {len(data)}\r\n\r\n".encode())
                                self.wfile.write(data)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.1)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                self.send_response(404); self.end_headers()

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()

    def close(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
