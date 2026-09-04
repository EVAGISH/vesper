"""Live frame server: the drone-camera side channel for in-progress runs.

RunCapture publishes every frame here (when VESPER_LIVE_PORT is set); a tiny
threaded HTTP server serves, per stream:

    GET /streams            JSON: {"run": <id>, "streams": ["overview", "fpv"]}
    GET /<stream>.jpg       latest frame, single JPEG
    GET /<stream>.mjpeg     multipart/x-mixed-replace stream (~10 fps)

Pure stdlib + PIL; runs inside the sim process on the GPU box. The browser UI
points <img> tags at it — the operator sees what the drone sees, not a viewport.
"""
import io
import json
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
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._start()

    def publish(self, rgb: np.ndarray, stream: str) -> None:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=75)
        with self._lock:
            self._jpeg[stream] = buf.getvalue()
            self._seq[stream] = self._seq.get(stream, 0) + 1

    def _start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep sim stdout clean
                pass

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")

            def do_GET(self):
                path = self.path.split("?")[0].strip("/")
                if path == "streams":
                    with server._lock:
                        body = json.dumps(
                            {"run": server.run_id, "streams": sorted(server._jpeg)}
                        ).encode()
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
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
