#!/usr/bin/env bash
# Prune the local runs/ mirror: newest KEEP runs stay complete; older runs
# keep videos + parquet + manifests but lose their PNG frame dirs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEEP="${KEEP:-3}"
cd "$REPO_ROOT/runs" 2>/dev/null || exit 0
rm -rf probe
ls -1td */ 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r d; do
  rm -rf "$d"frames "$d"frames_*
done
echo "pruned (kept $KEEP full runs): $(du -sh . | cut -f1) in runs/"
