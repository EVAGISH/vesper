"""Run capture: every sim run writes frames, an MP4, and a manifest to runs/<id>/.

If VESPER_RUNS_BUCKET is set (compose passes it through), finish() syncs the
run to s3://$VESPER_RUNS_BUCKET/runs/<id>/ so capture_pull.sh can fetch it.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


class RunCapture:
    def __init__(self, name: str, root: str | Path = "runs"):
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        self.dir = Path(root) / self.run_id
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True)
        self._n = 0
        self._meta: dict = {"name": name, "started": time.time()}

    def add_frame(self, rgba: np.ndarray) -> None:
        img = Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8))
        img.save(self.frames_dir / f"{self._n:06d}.png")
        self._n += 1

    def note(self, **kv) -> None:
        self._meta.update(kv)

    def finish(self, fps: int = 30) -> Path:
        video = self.dir / "overview.mp4"
        if self._n:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-i", str(self.frames_dir / "%06d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                check=True,
            )
        self._meta.update(frames=self._n, fps=fps, finished=time.time())
        (self.dir / "manifest.json").write_text(json.dumps(self._meta, indent=2))

        bucket = os.environ.get("VESPER_RUNS_BUCKET")
        if bucket:
            subprocess.run(
                ["aws", "s3", "sync", str(self.dir), f"s3://{bucket}/runs/{self.run_id}"],
                check=False,  # a failed sync shouldn't kill a finished run
            )
        return video
