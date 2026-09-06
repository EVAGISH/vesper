#!/usr/bin/env bash
# Train the native search teacher on the droplet's GPU instead of the Mac.
#
# The native lane (vesper.native.NativeSearchEnv) is Isaac-free pure torch, so
# it runs on CUDA unchanged -- and the L40S is ~20x the Mac's MPS throughput
# (~90k vs ~4k env-steps/s at 4096 envs). This wrapper does the whole round
# trip: rsync the code up, run training inside the vesper-sim container (which
# carries a CUDA torch under /isaac-sim/python.sh), then pull the finished run
# dir back so it shows up in the local Runs tab exactly like a Mac run.
#
#   scripts/train_search_gpu.sh --iters 800 --w_frontier 0.9 --w_yaw_rate 0.05 \
#       --w_cover 0.4 --w_cover_stale 0.4 --resume runs/<id>/search.pt
#
# Any flags are passed straight through to train_search_native.py. --device is
# forced to cuda. A --resume checkpoint is rsynced up first so finetunes work.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IP="${VESPER_DROPLET_IP:-138.197.158.222}"
KEY="${KEY_FILE:-$HOME/.ssh/vesper.pem}"
SSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
RSYNC_E="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
REMOTE=/root/vesper
TAG="gpu-search"

# pull --tag (for the run-dir name) and --resume (needs uploading) out of argv
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  [[ "${ARGS[$i]}" == "--tag" ]] && TAG="${ARGS[$((i+1))]}"
  [[ "${ARGS[$i]}" == "--resume" ]] && RESUME="${ARGS[$((i+1))]}"
done

echo "[gpu] syncing code -> $IP:$REMOTE"
rsync -az -e "$RSYNC_E" --exclude=__pycache__ --exclude='*.pyc' \
  "$REPO/vesper/"  "root@$IP:$REMOTE/vesper/"
rsync -az -e "$RSYNC_E" --exclude=__pycache__ --exclude='*.pyc' \
  "$REPO/scripts/" "root@$IP:$REMOTE/scripts/"

if [[ -n "${RESUME:-}" ]]; then
  echo "[gpu] uploading resume checkpoint $RESUME"
  $SSH "root@$IP" "mkdir -p $REMOTE/$(dirname "$RESUME")"
  rsync -az -e "$RSYNC_E" "$REPO/$RESUME" "root@$IP:$REMOTE/$RESUME"
fi

echo "[gpu] launching training in the vesper-sim container (device cuda)"
# capture the run dir the script prints, so we know what to pull back
$SSH "root@$IP" "cd $REMOTE && docker compose -f docker/compose.yml run --rm \
  --name vsp_${TAG} sim /isaac-sim/python.sh scripts/train_search_native.py \
  --device cuda ${ARGS[*]} 2>&1 | tee /root/vesper/${TAG}.log"

RUN=$($SSH "root@$IP" "grep -oE 'runs/[0-9]{8}-[0-9]{6}-${TAG}' $REMOTE/${TAG}.log | head -1")
if [[ -z "$RUN" ]]; then
  echo "[gpu] WARN: could not parse run dir from log; pulling newest ${TAG} run"
  RUN=$($SSH "root@$IP" "ls -dt $REMOTE/runs/*-${TAG} 2>/dev/null | head -1 | sed 's#$REMOTE/##'")
fi
echo "[gpu] pulling $RUN back to the Mac"
mkdir -p "$REPO/$RUN"
rsync -az -e "$RSYNC_E" "root@$IP:$REMOTE/$RUN/" "$REPO/$RUN/"
echo "[gpu] done -> $REPO/$RUN/search.pt"
