"""Serve a world USD over the WebRTC livestream, so the streaming client shows a
scene instead of an empty stage.

    /isaac-sim/python.sh scripts/live_world.py [assets/cornell/cornell.usd]

Connect the Isaac Sim WebRTC Streaming Client to the box's public IP once
"serving" is printed. Ctrl-C (or docker stop) to end the session.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("world", nargs="?", default="assets/cornell/cornell.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.livestream = 2  # WebRTC
app = AppLauncher(args).app

import sys

import omni.usd  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

usd = str(Path(args.world).resolve())
omni.usd.get_context().open_stage(usd)
for _ in range(30):  # let the stage settle before framing the shot
    app.update()
# aerial view over the campus core (world coords; spawn is the origin)
set_camera_view(eye=[60.0, 60.0, 220.0], target=[-220.0, -482.0, 20.0])
print(f"[live_world] serving {usd} -- connect the WebRTC client now", flush=True)
while app.is_running():
    app.update()
