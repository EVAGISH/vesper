"""Sightings from a real detector instead of from the simulator's own truth.

The search task decides "the drone can see that vehicle" in one of three ways,
and they differ only in who is allowed to answer:

  geometry    a cone, a range and a line-of-sight trace over the world map.
              Cheap, runs at thousands of environments, and knows the truth --
              it *is* the truth, softened by a dropout probability. This is the
              default, and the model every policy so far was trained against.
  pixels      the renderer's instance segmentation: a vehicle is seen when
              enough of its mask lands in the frame. Honest about occlusion and
              range, still the simulator answering.
  detector    this module. The rendered RGB goes to a trained RF-DETR, and a
              vehicle is seen when the detector puts a box on it. Nothing about
              the vehicle's existence reaches the belief except through the
              network's output -- misses, false negatives under canopy and the
              range at which a 30-pixel APC stops being findable are the
              detector's, not a config value.

Attribution. A detector returns boxes, not identities: it says "an APC is
*there*", never "that is vehicle 2". The task's belief is per target, so a box
has to be matched back to a slot. We project each vehicle's true world position
into the camera and accept the box that lands on it -- exactly the role instance
segmentation plays in pixel mode, and no more: the truth answers *which* target
a detection belongs to, never *whether* there was one. A vehicle the detector
misses stays unknown; a box on empty ground matches nothing and is dropped.

The projection is the same pinhole the env's TiledCamera is built from
(vesper.lab.frames.sensor_pose sets its pose, the task config its field of
view), so pixels here and pixels there are the same pixels.

Pure torch plus stdlib HTTP: this module imports on a Mac with no Isaac and no
CUDA, and the geometry is tested there. Only `RemoteDetector.detect` needs the
box.

Frame: world metres, x east, y north, z up. Camera: +x forward, +y left, +z up
(Isaac's "world" convention). Shapes: N envs, K targets, M boxes.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

import torch

from vesper.control.se3 import quat_to_rot
from vesper.lab.frames import sensor_pose


class DetectorError(RuntimeError):
    """The detector could not be reached or refused the batch."""


def focal_px(res: int, fov_half_deg: float) -> float:
    """Pinhole focal length in pixels for a square tile of `res` px."""
    return (res / 2.0) / math.tan(math.radians(fov_half_deg))


def project(drone_pos, quat, target_pos, cam_pitch_deg: float, fov_half_deg: float,
            res: int, offset=(0.0, 0.0, 0.0)):
    """Where each target lands in the body-fixed camera's frame.

    Returns (uv [N,K,2] pixels, visible [N,K] bool). u runs right, v runs down,
    the origin is the top-left corner. `visible` is only "in front of the lens
    and inside the tile" -- occlusion, range and whether anything can actually
    be made out there are the detector's business.
    """
    cam_pos, cam_quat = sensor_pose(drone_pos, quat, cam_pitch_deg, offset)
    R = quat_to_rot(cam_quat)                                   # camera -> world
    rel = target_pos - cam_pos.unsqueeze(1)                     # [N,K,3] world
    # world -> camera: R^T, applied to every target at once
    cam = torch.einsum("nij,nkj->nki", R.transpose(1, 2), rel)  # [N,K,3]
    depth = cam[..., 0]
    f = focal_px(res, fov_half_deg)
    safe = depth.clamp(min=1e-3)
    u = res / 2.0 - f * cam[..., 1] / safe                      # +y is left
    v = res / 2.0 - f * cam[..., 2] / safe                      # +z is up
    uv = torch.stack([u, v], dim=2)
    inside = (depth > 0.1) & (u >= 0) & (u < res) & (v >= 0) & (v < res)
    return uv, inside


def match_boxes(uv, inside, boxes, res: int, pad_px: float = 4.0):
    """Attribute detections to targets. Returns pseudo pixel counts [N,K] int64.

    `boxes` is a length-N list of [M_i,4] xyxy tensors in the tile's own pixel
    coordinates (M_i may be 0). A target claims the smallest box that contains
    its projected pixel, grown by `pad_px` so a box that is tight around a hull
    still catches a centroid sitting a pixel outside it. The value returned is
    that box's area, which keeps the task's `sight_px` threshold meaning what it
    says: a detection too small to be worth anything is still too small.

    Smallest-box-wins matters when two vehicles are close: the wide box that
    covers both would otherwise be handed to whichever slot is checked first.
    Two targets in one box both claim it -- that is a genuine ambiguity in the
    detection, and calling both seen is the reading that does not invent a miss.
    """
    n, k = uv.shape[0], uv.shape[1]
    out = torch.zeros(n, k, dtype=torch.long, device=uv.device)
    for i in range(n):
        b = boxes[i]
        if b is None or len(b) == 0:
            continue
        b = torch.as_tensor(b, dtype=torch.float32, device=uv.device).reshape(-1, 4)
        area = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))
        pu, pv = uv[i, :, 0].unsqueeze(1), uv[i, :, 1].unsqueeze(1)          # [K,1]
        hit = ((pu >= b[:, 0] - pad_px) & (pu <= b[:, 2] + pad_px) &
               (pv >= b[:, 1] - pad_px) & (pv <= b[:, 3] + pad_px))          # [K,M]
        hit = hit & inside[i].unsqueeze(1)
        if not hit.any():
            continue
        # smallest matching box per target; +inf parks the non-matches
        cost = torch.where(hit, area.unsqueeze(0).expand_as(hit),
                           torch.full_like(hit, float("inf"), dtype=area.dtype))
        best = cost.argmin(dim=1)
        got = hit.any(dim=1)
        out[i] = torch.where(got, area[best].round().long(), torch.zeros_like(out[i]))
    return out


class RemoteDetector:
    """Client for scripts/detect_server.py -- the RF-DETR sitting beside the sim.

    The detector cannot live in the Isaac container: rfdetr installs into the
    venv scripts/rfdetr_setup.sh builds under scratch/, and anything pip-installed
    into a running Isaac container dies with it. So it runs as its own process on
    the droplet and the sim posts frames to it over the loopback. The sim
    container uses host networking, so 127.0.0.1 is the same machine either way.

    Frames go over as a raw uint8 buffer with a JSON header line, which is the
    whole protocol: at 64 tiles of 384 px that is 28 MB a step over loopback and
    the encode cost of anything nicer is not worth paying.
    """

    def __init__(self, url: str, threshold: float = 0.5, timeout: float = 30.0,
                 retries: int = 2):
        self.url = url.rstrip("/")
        self.threshold = float(threshold)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.info: dict = {}

    # ------------------------------------------------------------------ health
    def health(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=8) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise DetectorError(f"detector at {self.url} is not answering: {e}") from e

    def wait_ready(self, seconds: float = 120.0) -> dict:
        """Block until the server reports a loaded model. Raises on timeout."""
        import time
        deadline = time.time() + seconds
        last = None
        while time.time() < deadline:
            try:
                h = self.health()
                if h.get("ready"):
                    self.info = h
                    return h
                last = h.get("status") or "loading"
            except DetectorError as e:
                last = str(e)
            time.sleep(2.0)
        raise DetectorError(f"detector at {self.url} never became ready ({last})")

    # ------------------------------------------------------------------ detect
    def detect(self, rgb):
        """rgb [N,H,W,3] uint8 tensor -> (boxes, scores), lists of length N.

        Boxes are xyxy in the submitted tile's own pixel coordinates.
        """
        arr = rgb.detach().to("cpu", torch.uint8)
        if arr.shape[-1] > 3:              # some annotators hand back RGBA
            arr = arr[..., :3]
        arr = arr.contiguous()
        n, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
        header = json.dumps({"n": n, "h": h, "w": w, "threshold": self.threshold})
        body = header.encode() + b"\n" + arr.numpy().tobytes()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.url}/detect", data=body,
                    headers={"Content-Type": "application/octet-stream"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                return out["boxes"], out["scores"]
            except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as e:
                last = e
        # Falling back to geometry here would quietly turn a detector run into a
        # geometry run and nobody would see it in the curve. Fail loudly instead.
        raise DetectorError(
            f"detector at {self.url} failed on a batch of {n} after "
            f"{self.retries + 1} attempts: {last}")


class DetectorSight:
    """Turns a rendered batch into the [N,K] the search task expects.

    Holds the camera intrinsics so the env only has to hand over poses and
    pixels, and keeps a running audit of the detector against the simulator's
    own segmentation when the env can supply it -- recall is the number that
    says whether a detector-in-the-loop run is learning from a sensor or from
    noise, and it is nearly free to compute.
    """

    def __init__(self, client: RemoteDetector, cam_pitch_deg: float, fov_half_deg: float,
                 res: int, offset=(0.0, 0.0, 0.0), pad_px: float = 4.0):
        self.client = client
        self.cam_pitch_deg = float(cam_pitch_deg)
        self.fov_half_deg = float(fov_half_deg)
        self.res = int(res)
        self.offset = tuple(offset)
        self.pad_px = float(pad_px)
        self.last_boxes: int = 0

    def seen_px(self, rgb, drone_pos, quat, target_pos):
        """[N,K] pseudo pixel counts: box area where the detector found a target."""
        boxes, _ = self.client.detect(rgb)
        self.last_boxes = sum(len(b) for b in boxes)
        uv, inside = project(drone_pos, quat, target_pos, self.cam_pitch_deg,
                             self.fov_half_deg, self.res, self.offset)
        return match_boxes(uv, inside, boxes, self.res, self.pad_px)
