#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
IP=$(droplet_ip)
[ -n "$IP" ] || { echo "no $DROPLET_NAME droplet running"; exit 1; }
exec ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new root@"$IP" "$@"
