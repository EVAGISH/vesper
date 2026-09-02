#!/usr/bin/env bash
# Destroys the dev droplet (disk is LOST -- snapshot.sh first if you want state).
source "$(dirname "$0")/env.sh"
ID=$(droplet_json | jq -r .id)
[ -n "$ID" ] && [ "$ID" != null ] || { echo "no $DROPLET_NAME droplet"; exit 0; }
read -p "destroy droplet $ID ($DROPLET_NAME)? [y/N] " a; [ "$a" = y ] || exit 1
api DELETE "/droplets/$ID" && echo "destroyed"
