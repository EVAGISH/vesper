#!/usr/bin/env bash
# Pull run artifacts to the local runs/ mirror.
# rsyncs straight from the DO droplet.
#   scripts/capture_pull.sh            # pull all runs from the droplet
#   scripts/capture_pull.sh <run-id>   # pull one run
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/infra/do/env.sh" 2>/dev/null || true

IP=$(droplet_ip 2>/dev/null || true)
if [ -z "${IP:-}" ]; then
  echo "no droplet running"; exit 1
fi
SRC="root@$IP:vesper/runs/${1:-}"
rsync -az -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new" "$SRC/" "$REPO_ROOT/runs/${1:-}/"
echo "pulled from droplet -> runs/${1:-}"
"$REPO_ROOT/scripts/runs_prune.sh"
ls -t "$REPO_ROOT"/runs/*/overview.mp4 2>/dev/null | head -1 | xargs -I{} open {} 2>/dev/null || true
