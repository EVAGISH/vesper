"""Run capture: every sim run writes frames, an MP4, and a manifest to runs/<id>/.

capture_pull.sh rsyncs the directory from the droplet to the Mac.

The PNG sequence is scratch space for ffmpeg and is deleted once the MP4 exists.
It is not small: at compress_level=1 a 1280x720 frame is ~800 KB, so a 30 s
two-stream run leaves ~3 GB behind. Keeping them had grown runs/ to 68 GB --
170x the size of the 46 MP4s they encoded -- which went straight into every
droplet snapshot. Pass keep_frames=True if you genuinely need the stills.
"""
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


class RunCapture:
    def __init__(self, name: str, root: str | Path = "runs", keep_frames: bool = False):
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True)
        self.keep_frames = keep_frames
        self._n: dict[str, int] = {}
        self._meta: dict = {"name": name, "started": time.time()}
        self._pool = ThreadPoolExecutor(max_workers=4)     # PNG encoding off the sim thread
        self._pending = []

    def _stream_dir(self, stream: str) -> Path:
        return self.dir / ("frames" if stream == "overview" else f"frames_{stream}")

    def add_frame(self, rgba: np.ndarray, stream: str = "overview") -> None:
        d = self._stream_dir(stream)
        if stream not in self._n:
            d.mkdir(exist_ok=True)
            self._n[stream] = 0
        frame = np.array(np.asarray(rgba)[..., :3], dtype=np.uint8, copy=True)   # detach from the GPU buffer
        path = d / f"{self._n[stream]:06d}.png"
        self._pending.append(self._pool.submit(lambda f=frame, p=path: Image.fromarray(f).save(p, compress_level=1)))
        self._n[stream] += 1
        if len(self._pending) > 64:
            self._pending = [f for f in self._pending if not f.done()]

    def note(self, **kv) -> None:
        self._meta.update(kv)

    def finish(self, fps: int = 30) -> Path:
        for f in self._pending:
            f.result()
        self._pool.shutdown(wait=True)
        video = self.dir / "overview.mp4"
        for stream, count in self._n.items():
            if not count:
                continue
            out = self.dir / f"{stream}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-i", str(self._stream_dir(stream) / "%06d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                check=True,
            )
        if not self.keep_frames:
            for stream in self._n:
                shutil.rmtree(self._stream_dir(stream), ignore_errors=True)
        self._meta.update(frames=self._n, fps=fps, finished=time.time(),
                          kept_frames=self.keep_frames)
        (self.dir / "manifest.json").write_text(json.dumps(self._meta, indent=2))
        return video
