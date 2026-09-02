"""Run capture: every sim run writes frames, an MP4, and a manifest to runs/<id>/.

capture_pull.sh rsyncs the directory from the droplet to the Mac.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


class RunCapture:
    def __init__(self, name: str, root: str | Path = "runs"):
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True)
        self._n: dict[str, int] = {}
        self._meta: dict = {"name": name, "started": time.time()}

    def _stream_dir(self, stream: str) -> Path:
        return self.dir / ("frames" if stream == "overview" else f"frames_{stream}")

    def add_frame(self, rgba: np.ndarray, stream: str = "overview") -> None:
        d = self._stream_dir(stream)
        if stream not in self._n:
            d.mkdir(exist_ok=True)
            self._n[stream] = 0
        img = Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8))
        img.save(d / f"{self._n[stream]:06d}.png")
        self._n[stream] += 1

    def note(self, **kv) -> None:
        self._meta.update(kv)

    def finish(self, fps: int = 30) -> Path:
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
        self._meta.update(frames=self._n, fps=fps, finished=time.time())
        (self.dir / "manifest.json").write_text(json.dumps(self._meta, indent=2))
        return video
