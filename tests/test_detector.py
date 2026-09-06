"""The detector's geometry: where a target lands in the tile, and which box
gets to claim it. Pure torch, so this runs on a Mac with no Isaac and no GPU --
the network itself is exercised on the box, this is the part that decides
whether a correct detection is credited to the right vehicle.
"""
import math

import pytest
import torch

from vesper.lab.detector import DetectorSight, focal_px, match_boxes, project
from vesper.lab.frames import camera_axis

RES = 128
PITCH = 40.0
FOV = 55.0


def _identity(n=1):
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0                       # wxyz, nose along world +x
    return q


def test_target_on_the_camera_axis_lands_in_the_middle():
    quat = _identity()
    pos = torch.tensor([[10.0, -4.0, 60.0]])
    axis = camera_axis(quat, math.radians(PITCH))
    target = (pos + 90.0 * axis).unsqueeze(1)
    uv, inside = project(pos, quat, target, PITCH, FOV, RES)
    assert bool(inside[0, 0])
    assert uv[0, 0, 0] == pytest.approx(RES / 2, abs=0.5)
    assert uv[0, 0, 1] == pytest.approx(RES / 2, abs=0.5)


def test_the_lens_edge_is_the_configured_field_of_view():
    """A target exactly fov_half off the axis sits on the tile's edge -- the
    rendered TiledCamera is built from the same two numbers, so the projection
    and the render agree on what is in frame."""
    quat = _identity()
    pos = torch.zeros(1, 3)
    pos[0, 2] = 80.0
    f = focal_px(RES, FOV)
    assert f == pytest.approx((RES / 2) / math.tan(math.radians(FOV)))
    # 100 m down the axis, then sideways by exactly the half-angle
    axis = camera_axis(quat, math.radians(PITCH))[0]
    left = torch.tensor([0.0, 1.0, 0.0])                 # camera +y, nose along +x
    d = 100.0
    target = (pos[0] + d * axis + d * math.tan(math.radians(FOV)) * left).view(1, 1, 3)
    uv, _ = project(pos, quat, target, PITCH, FOV, RES)
    assert uv[0, 0, 0] == pytest.approx(0.0, abs=1.0)    # +y is left, so u -> 0
    # a hair further round and it has left the tile
    wider = (pos[0] + d * axis + d * math.tan(math.radians(FOV + 3)) * left).view(1, 1, 3)
    _, inside = project(pos, quat, wider, PITCH, FOV, RES)
    assert not bool(inside[0, 0])


def test_behind_and_off_tile_targets_are_not_in_frame():
    quat = _identity()
    pos = torch.zeros(1, 3)
    pos[0, 2] = 50.0
    behind = torch.tensor([[[-200.0, 0.0, 0.0]]])
    _, inside = project(pos, quat, behind, PITCH, FOV, RES)
    assert not bool(inside[0, 0])


def test_a_box_on_the_target_is_a_sighting_and_empty_ground_is_not():
    uv = torch.tensor([[[64.0, 64.0], [10.0, 10.0]]])          # two targets
    inside = torch.tensor([[True, True]])
    boxes = [torch.tensor([[54.0, 54.0, 74.0, 74.0]])]         # 20x20 on the first only
    got = match_boxes(uv, inside, boxes, RES)
    assert got[0, 0].item() == 400
    assert got[0, 1].item() == 0


def test_a_detection_the_network_did_not_make_is_a_miss():
    uv = torch.tensor([[[64.0, 64.0]]])
    inside = torch.tensor([[True]])
    assert match_boxes(uv, inside, [torch.zeros(0, 4)], RES)[0, 0].item() == 0
    assert match_boxes(uv, inside, [None], RES)[0, 0].item() == 0


def test_a_target_out_of_frame_cannot_be_claimed_by_a_box():
    """`inside` is the gate: a vehicle the camera is not pointing at must not
    inherit a detection that happens to sit at its projected pixel."""
    uv = torch.tensor([[[64.0, 64.0]]])
    boxes = [torch.tensor([[0.0, 0.0, 128.0, 128.0]])]
    assert match_boxes(uv, torch.tensor([[False]]), boxes, RES)[0, 0].item() == 0
    assert match_boxes(uv, torch.tensor([[True]]), boxes, RES)[0, 0].item() > 0


def test_the_tighter_box_wins_when_two_vehicles_are_close():
    """A wide box covering both hulls and a tight one on the second: the tight
    box is the better attribution for the target it lands on."""
    uv = torch.tensor([[[30.0, 30.0], [70.0, 70.0]]])
    inside = torch.tensor([[True, True]])
    wide = [0.0, 0.0, 100.0, 100.0]           # 10000 px
    tight = [60.0, 60.0, 80.0, 80.0]          # 400 px
    got = match_boxes(uv, inside, [torch.tensor([wide, tight])], RES)
    assert got[0, 0].item() == 10000          # only the wide box reaches it
    assert got[0, 1].item() == 400            # the tight one claims the second


def test_the_pad_catches_a_centroid_just_outside_a_tight_box():
    uv = torch.tensor([[[52.0, 64.0]]])
    inside = torch.tensor([[True]])
    boxes = [torch.tensor([[54.0, 54.0, 74.0, 74.0]])]
    assert match_boxes(uv, inside, boxes, RES, pad_px=0.0)[0, 0].item() == 0
    assert match_boxes(uv, inside, boxes, RES, pad_px=4.0)[0, 0].item() == 400


class _FakeClient:
    """Stands in for the box: answers with one box over the whole tile."""

    def __init__(self):
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        n = rgb.shape[0]
        return ([torch.tensor([[0.0, 0.0, float(RES), float(RES)]])] * n,
                [[0.9]] * n)


def test_sight_turns_a_rendered_batch_into_per_target_counts():
    sight = DetectorSight(_FakeClient(), PITCH, FOV, RES)
    quat = _identity(2)
    pos = torch.tensor([[0.0, 0.0, 60.0], [5.0, 5.0, 60.0]])
    axis = camera_axis(quat, math.radians(PITCH))
    on_axis = pos + 70.0 * axis
    behind = pos + torch.tensor([[-150.0, 0.0, 0.0]])
    targets = torch.stack([on_axis, behind], dim=1)              # [2,2,3]
    got = sight.seen_px(torch.zeros(2, RES, RES, 3, dtype=torch.uint8), pos, quat, targets)
    assert got.shape == (2, 2)
    assert (got[:, 0] > 0).all()          # the one in frame is seen
    assert (got[:, 1] == 0).all()         # the one behind the drone is not
    assert sight.last_boxes == 2


# --- the wire ---------------------------------------------------------------
# scripts/detect_server.py cannot run here (rfdetr lives in the venv on the GPU
# box), so the contract is pinned from this side: a stub speaking the documented
# protocol, driving the real client.

class _StubServer:
    """A detect server that puts one box over the middle of every frame."""

    def __init__(self, ready=True, fail=False):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        state = {"ready": ready, "status": "ready" if ready else "loading"}
        self.seen = []
        seen = self.seen

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._json(200, state)

            def do_POST(self):
                if fail:
                    return self._json(500, {"error": "boom"})
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                head, _, payload = raw.partition(b"\n")
                hdr = json.loads(head)
                seen.append((hdr, len(payload)))
                n, w = hdr["n"], hdr["w"]
                mid = w / 2
                self._json(200, {
                    "boxes": [[[mid - 8, mid - 8, mid + 8, mid + 8]] for _ in range(n)],
                    "scores": [[0.87] for _ in range(n)],
                })

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_the_client_round_trips_a_batch_of_frames():
    from vesper.lab.detector import RemoteDetector
    srv = _StubServer()
    try:
        c = RemoteDetector(srv.url, threshold=0.4)
        assert c.wait_ready(5)["ready"]
        rgb = torch.zeros(3, 32, 32, 3, dtype=torch.uint8)
        boxes, scores = c.detect(rgb)
        assert len(boxes) == 3 and len(scores) == 3
        hdr, nbytes = srv.seen[-1]
        assert hdr == {"n": 3, "h": 32, "w": 32, "threshold": 0.4}
        assert nbytes == 3 * 32 * 32 * 3           # raw uint8, no encoding
    finally:
        srv.close()


def test_a_broken_detector_stops_the_run_instead_of_quietly_becoming_geometry():
    """Falling back to the built-in sensor would turn a detector run into a
    geometry run with nothing in the curve to show it."""
    from vesper.lab.detector import DetectorError, RemoteDetector
    srv = _StubServer(fail=True)
    try:
        c = RemoteDetector(srv.url, retries=0)
        with pytest.raises(DetectorError):
            c.detect(torch.zeros(1, 8, 8, 3, dtype=torch.uint8))
    finally:
        srv.close()


def test_waiting_on_a_detector_that_never_loads_times_out():
    from vesper.lab.detector import DetectorError, RemoteDetector
    srv = _StubServer(ready=False)
    try:
        with pytest.raises(DetectorError):
            RemoteDetector(srv.url).wait_ready(3)
    finally:
        srv.close()
