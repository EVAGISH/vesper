#!/usr/bin/env bash
# First-boot setup on a fresh droplet: copy repo, NGC login, build the image.
# Run from the Mac after launch.sh. Needs NGC_API_KEY in .env.
source "$(dirname "$0")/env.sh"
: "${NGC_API_KEY:?Set NGC_API_KEY in .env (ngc.nvidia.com -> Setup -> Generate API Key)}"
IP=$(droplet_ip)
[ -n "$IP" ] || { echo "no droplet; run launch.sh"; exit 1; }

rsync -az -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new" \
  --exclude .git --exclude runs --exclude assets --exclude .env --exclude node_modules --exclude .next --exclude scratch --exclude .venv \
  "$REPO_ROOT/" root@"$IP":vesper/

ssh -i "$KEY_FILE" root@"$IP" "
  set -e
  docker compose version >/dev/null 2>&1 || {
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -sL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  }
  echo '$NGC_API_KEY' | docker login nvcr.io -u '\$oauthtoken' --password-stdin
  cd vesper/docker && docker compose build
"
# environments are built on the box (web/server/app.py -> scripts/build_geo_world.py)
bash "$(dirname "$0")/geo_env.sh"
echo "syncing tree assets + textures (the world build needs them)"
rsync -az -e "ssh -i $KEY_FILE" "$REPO_ROOT/assets/vegetation" root@"$IP":vesper/assets/
echo "provisioned. Next: infra/do/ssh.sh then"
echo "  cd vesper/docker && docker compose run --rm sim /isaac-sim/python.sh scripts/smoke_render.py"
