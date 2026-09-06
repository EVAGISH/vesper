#!/usr/bin/env bash
# Start Isaac Sim on the GPU box in WebRTC livestream mode.
# Connect from the Mac with the Isaac Sim WebRTC Streaming Client -> box public IP.
# (Script name differs across 5.x builds: try isaac-sim.streaming.sh, then runheadless.sh)
set -euo pipefail
cd "$(dirname "$0")/../docker"
docker compose run --rm sim bash -lc \
  '/isaac-sim/isaac-sim.streaming.sh 2>/dev/null || /isaac-sim/runheadless.sh'
