#!/usr/bin/env bash
# Launch the vesper dev GPU droplet. Uses the newest vesper snapshot if one
# exists (post-snapshot.sh relaunches skip the docker build), else the
# NVIDIA AI/ML Ready base image. Attaches the restricted firewall.
source "$(dirname "$0")/env.sh"

if [ -n "$(droplet_json)" ]; then
  echo "droplet $DROPLET_NAME already exists at $(droplet_ip)"; exit 0
fi

# --- ssh key: register vesper.pem's public half if DO doesn't have it
PUB=$(ssh-keygen -y -f "$KEY_FILE")
FP=$(ssh-keygen -E md5 -lf /dev/stdin <<<"$PUB" | awk '{print $2}' | sed 's/^MD5://')
if [ -z "$(api GET /account/keys | jq -r ".ssh_keys[] | select(.fingerprint==\"$FP\") | .id")" ]; then
  api POST /account/keys "{\"name\":\"vesper\",\"public_key\":\"$PUB\"}" >/dev/null
  echo "registered ssh key"
fi

# --- image: newest vesper snapshot, else base
IMAGE=$(api GET "/snapshots?resource_type=droplet&per_page=200" \
  | jq -r '[.snapshots[] | select(.name|startswith("vesper-dev"))] | sort_by(.created_at) | last | .id // empty')
IMAGE="${IMAGE:-$DO_IMAGE}"
echo "launching $DO_SIZE in $DO_REGION from image: $IMAGE"

RESP=$(api POST /droplets "{
  \"name\": \"$DROPLET_NAME\", \"region\": \"$DO_REGION\", \"size\": \"$DO_SIZE\",
  \"image\": \"$IMAGE\", \"ssh_keys\": [\"$FP\"], \"tags\": [\"vesper\"], \"monitoring\": true
}")
ID=$(echo "$RESP" | jq -r '.droplet.id // empty')
if [ -z "$ID" ]; then
  echo "create failed:"; echo "$RESP" | jq .
  echo "If the error says the size is unavailable/not found, open a DO support"
  echo "ticket asking to enable NVIDIA GPU Droplets for this team."
  exit 1
fi

echo -n "waiting for droplet $ID"
until [ "$(api GET /droplets/$ID | jq -r .droplet.status)" = active ]; do echo -n .; sleep 5; done
echo

# --- firewall (create once), assign droplet
MY_IP="${MY_IP:-$(curl -s https://api.ipify.org)}"
FW=$(api GET /firewalls | jq -r ".firewalls[] | select(.name==\"$FIREWALL_NAME\") | .id")
rule() { echo "{\"protocol\":\"$1\",\"ports\":\"$2\",\"sources\":{\"addresses\":[\"$MY_IP/32\"]}}"; }
if [ -z "$FW" ]; then
  api POST /firewalls "{
    \"name\": \"$FIREWALL_NAME\", \"droplet_ids\": [$ID],
    \"inbound_rules\": [$(rule tcp 22),$(rule tcp 49100),$(rule tcp 47995-48012),$(rule udp 47995-48012),$(rule tcp 49000-49007),$(rule udp 49000-49007)],
    \"outbound_rules\": [
      {\"protocol\":\"tcp\",\"ports\":\"0\",\"destinations\":{\"addresses\":[\"0.0.0.0/0\",\"::/0\"]}},
      {\"protocol\":\"udp\",\"ports\":\"0\",\"destinations\":{\"addresses\":[\"0.0.0.0/0\",\"::/0\"]}},
      {\"protocol\":\"icmp\",\"destinations\":{\"addresses\":[\"0.0.0.0/0\",\"::/0\"]}}
    ]
  }" | jq -r '.firewall.id // .message'
else
  api POST "/firewalls/$FW/droplets" "{\"droplet_ids\":[$ID]}" >/dev/null && echo "firewall attached"
fi

echo "ready: $(droplet_ip)   (~\$0.76-1.57/hr while it exists -- snapshot.sh or destroy.sh when done)"
