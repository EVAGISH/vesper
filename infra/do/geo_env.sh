#!/usr/bin/env bash
# Python env for the geo world build on the droplet (no Isaac, no GPU): usd-core +
# shapely/rasterio/etc in /root/geo-venv, plus the CC0 PBR texture sets. Idempotent;
# provision.sh calls it, and the web server's box build expects /root/geo-venv/bin/python.
source "$(dirname "$0")/env.sh"
IP=$(droplet_ip)
[ -n "$IP" ] || { echo "no droplet; run launch.sh"; exit 1; }
ssh -i "$KEY_FILE" root@"$IP" '
  set -e
  if [ ! -x /root/geo-venv/bin/python ]; then
    apt-get install -y -qq python3-venv python3-pip >/dev/null 2>&1 || true
    python3 -m venv /root/geo-venv
  fi
  /root/geo-venv/bin/pip install -q --upgrade pip
  /root/geo-venv/bin/pip install -q usd-core numpy pillow shapely mapbox-earcut rasterio requests \
      scipy trimesh pyproj laspy lazrs opencv-python-headless pyarrow
  /root/geo-venv/bin/pip install -q -e /root/vesper
  [ -d /root/vesper/textures/pbr/grass ] || (cd /root/vesper && /root/geo-venv/bin/python scripts/fetch_pbr.py)
  /root/geo-venv/bin/python -c "import pxr, shapely, rasterio, mapbox_earcut, trimesh, cv2, scipy; print(\"geo-venv ok\")"
'
