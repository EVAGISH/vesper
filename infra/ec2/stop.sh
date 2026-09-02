#!/usr/bin/env bash
# Stops (not terminates) the dev box; EBS state survives, billing mostly stops.
source "$(dirname "$0")/env.sh"
ID=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:Name,Values=$TAG_NAME" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
[ "$ID" != "None" ] || { echo "no running $TAG_NAME instance"; exit 0; }
aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$ID"
echo "stopping $ID"
