#!/usr/bin/env bash
# Power off -> snapshot -> destroy. The daily "done for now" command:
# next launch.sh boots from the snapshot with docker image + shader cache intact.
# Snapshot storage ~ $0.06/GB/month.
source "$(dirname "$0")/env.sh"
ID=$(droplet_json | jq -r .id)
[ -n "$ID" ] && [ "$ID" != null ] || { echo "no $DROPLET_NAME droplet"; exit 0; }

# scratch/ holds the synthetic datasets, the RF-DETR venv and its checkpoints:
# tens of GB of regenerable working files. Snapshot storage is billed per GB per
# month, so they are cleared before the disk is imaged. Pull anything worth
# keeping first -- scripts/detect_pull.sh does the small artefacts.
IP=$(droplet_ip)
if [ -n "$IP" ]; then
  echo "clearing scratch/ so it stays out of the snapshot"
  ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new root@"$IP" \
    'du -sh vesper/scratch 2>/dev/null; rm -rf vesper/scratch' || true
fi

api POST "/droplets/$ID/actions" '{"type":"power_off"}' >/dev/null
echo -n "powering off"
until [ "$(api GET /droplets/$ID | jq -r .droplet.status)" = off ]; do echo -n .; sleep 5; done; echo

NAME="vesper-dev-$(date +%Y%m%d-%H%M)"
ACT=$(api POST "/droplets/$ID/actions" "{\"type\":\"snapshot\",\"name\":\"$NAME\"}" | jq -r .action.id)
echo -n "snapshotting as $NAME (takes a while)"
until [ "$(api GET /actions/$ACT | jq -r .action.status)" = completed ]; do echo -n .; sleep 15; done; echo

api DELETE "/droplets/$ID" && echo "droplet destroyed; relaunch with launch.sh"
