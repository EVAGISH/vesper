#!/usr/bin/env bash
# Shared config for DigitalOcean scripts. Reads repo-root .env.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && set -a && source "$REPO_ROOT/.env" && set +a
: "${DIGITALOCEAN_TOKEN:?Set DIGITALOCEAN_TOKEN in .env}"

export DO_REGION="${DO_REGION:-nyc2}"
export DO_SIZE="${DO_SIZE:-gpu-4000adax1-20gb}"       # RTX 4000 Ada 20GB ~$0.76/hr; big tier: gpu-l40sx1-48gb
export DO_IMAGE="${DO_IMAGE:-gpu-h100x1-base}"         # "NVIDIA AI/ML Ready" Ubuntu: driver + docker preinstalled
export DROPLET_NAME="${DROPLET_NAME:-vesper-dev}"
export FIREWALL_NAME="${FIREWALL_NAME:-vesper-dev-fw}"
export KEY_FILE="${KEY_FILE:-$HOME/.ssh/vesper.pem}"
KEY_FILE="${KEY_FILE/#\~/$HOME}"

api() { # api METHOD PATH [JSON_BODY]
  local method="$1" path="$2" body="${3:-}"
  curl -s -X "$method" -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
    -H "Content-Type: application/json" ${body:+-d "$body"} \
    "https://api.digitalocean.com/v2$path"
}

droplet_json() { api GET "/droplets?tag_name=vesper" | jq ".droplets[] | select(.name==\"$DROPLET_NAME\")"; }
droplet_ip()   { droplet_json | jq -r '.networks.v4[] | select(.type=="public") | .ip_address'; }
