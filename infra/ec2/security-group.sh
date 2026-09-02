#!/usr/bin/env bash
# Creates the dev security group: SSH + Isaac Sim livestream, allowlisted to MY_IP only.
# The livestream has no meaningful auth -- never open these to 0.0.0.0/0.
source "$(dirname "$0")/env.sh"

MY_IP="${MY_IP:-$(curl -s https://checkip.amazonaws.com)}"
echo "Allowlisting ${MY_IP}/32"

SG_ID=$(aws ec2 create-security-group --region "$AWS_REGION" \
  --group-name "$SG_NAME" --description "vesper dev box" \
  --query GroupId --output text)

auth() { aws ec2 authorize-security-group-ingress --region "$AWS_REGION" \
  --group-id "$SG_ID" --protocol "$1" --port "$2" --cidr "${MY_IP}/32"; }

auth tcp 22
# Isaac Sim WebRTC livestream (signaling + media; ranges cover 4.5/5.x clients)
auth tcp 49100
auth tcp 47995-48012
auth udp 47995-48012
auth tcp 49000-49007
auth udp 49000-49007

echo "Security group: $SG_ID"
