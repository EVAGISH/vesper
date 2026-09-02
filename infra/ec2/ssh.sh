#!/usr/bin/env bash
# SSH to the running dev box (found by Name tag).
source "$(dirname "$0")/env.sh"
IP=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:Name,Values=$TAG_NAME" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
[ "$IP" != "None" ] || { echo "no running $TAG_NAME instance"; exit 1; }
exec ssh -i "${KEY_FILE/#\~/$HOME}" ubuntu@"$IP" "$@"
