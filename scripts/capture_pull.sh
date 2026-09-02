#!/usr/bin/env bash
# Pull run artifacts to the local runs/ mirror.
# Default: rsync straight from the DO droplet. Falls back to S3 if no droplet.
#   scripts/capture_pull.sh            # pull all runs from the droplet
#   scripts/capture_pull.sh <run-id>   # pull one run
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/infra/do/env.sh" 2>/dev/null || true

IP=$(droplet_ip 2>/dev/null || true)
if [ -n "${IP:-}" ]; then
  SRC="root@$IP:vesper/runs/${1:-}"
  rsync -az -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new" "$SRC/" "$REPO_ROOT/runs/${1:-}/"
  echo "pulled from droplet -> runs/${1:-}"
elif [ -n "${VESPER_BUCKET:-}" ]; then
  aws s3 sync "s3://$VESPER_BUCKET/runs/${1:-}" "$REPO_ROOT/runs/${1:-}"
  echo "pulled from s3 -> runs/${1:-}"
else
  echo "no droplet running and no VESPER_BUCKET set"; exit 1
fi
ls -t "$REPO_ROOT"/runs/*/overview.mp4 2>/dev/null | head -1 | xargs -I{} open {} 2>/dev/null || true
