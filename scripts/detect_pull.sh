#!/usr/bin/env bash
# Pull the small, non-regenerable detector artefacts off the droplet:
# the review sample, the ONNX, the TensorRT benchmark and the metrics.
# The dataset itself (GBs of renders) stays on the box -- it is reproducible
# from gen_detect_dataset.py and a seed.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/infra/do/env.sh"
IP=$(droplet_ip)
[ -n "$IP" ] || { echo "no droplet running"; exit 1; }
mkdir -p "$REPO_ROOT/runs/detect"
# review/ is the annotated sample meant for human eyes; the rest is a filter for
# the few small files worth keeping. The dataset's own JPEGs are deliberately not
# matched -- they are gigabytes and gen_detect_dataset.py reproduces them from a seed.
rsync -az --info=stats1 -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new" \
  --include='review/***' \
  --include='*/' \
  --include='*.onnx' --include='*.json' --include='*.txt' \
  --exclude='*' \
  root@"$IP":vesper/scratch/ "$REPO_ROOT/runs/detect/"
echo "pulled -> runs/detect/"
