#!/usr/bin/env bash
# Launches the dev box from the baked AMI. On-demand, 200GB gp3, tagged for ssh.sh/stop.sh.
source "$(dirname "$0")/env.sh"

: "${AMI_ID:?Set AMI_ID in .env (run infra/ami/build.sh first)}"
: "${KEY_NAME:?Set KEY_NAME in .env}"

SG_ID=$(aws ec2 describe-security-groups --region "$AWS_REGION" \
  --group-names "$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)

INSTANCE_ID=$(aws ec2 run-instances --region "$AWS_REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$IAM_PROFILE" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_VOLUME_GB,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_NAME}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Launched $INSTANCE_ID; waiting for IP..."
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
