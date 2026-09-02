#!/usr/bin/env bash
# Bakes the vesper dev AMI:
#   AWS Deep Learning Base GPU AMI (driver + docker + nvidia toolkit preinstalled)
#   + the Isaac Lab container image pre-pulled, so relaunches skip a ~20GB pull.
#
# Prereqs (one-time, manual):
#   - Service Quotas increase for G-family vCPUs approved
#   - NGC API key stored:  aws ssm put-parameter --name /vesper/ngc_api_key \
#       --type SecureString --value <key>
#   - infra/ec2/security-group.sh run once
source "$(dirname "$0")/../ec2/env.sh"
: "${KEY_NAME:?Set KEY_NAME in .env}"

ISAAC_LAB_TAG="${ISAAC_LAB_TAG:-2.3.0}"

# Latest DL Base GPU AMI (Ubuntu 22.04) via AWS's public SSM parameter
BASE_AMI=$(aws ssm get-parameter --region "$AWS_REGION" \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)
echo "Base AMI: $BASE_AMI"

SG_ID=$(aws ec2 describe-security-groups --region "$AWS_REGION" \
  --group-names "$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)

ID=$(aws ec2 run-instances --region "$AWS_REGION" \
  --image-id "$BASE_AMI" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$IAM_PROFILE" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_VOLUME_GB,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vesper-ami-bake}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Bake instance $ID at $IP; waiting for SSH..."
KEY="${KEY_FILE/#\~/$HOME}"
until ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 ubuntu@"$IP" true 2>/dev/null; do sleep 5; done

NGC_KEY=$(aws ssm get-parameter --region "$AWS_REGION" --name /vesper/ngc_api_key \
  --with-decryption --query Parameter.Value --output text)
ssh -i "$KEY" ubuntu@"$IP" "echo '$NGC_KEY' | docker login nvcr.io -u '\$oauthtoken' --password-stdin \
  && docker pull nvcr.io/nvidia/isaac-lab:$ISAAC_LAB_TAG \
  && docker logout nvcr.io"

echo "Creating AMI..."
AMI=$(aws ec2 create-image --region "$AWS_REGION" --instance-id "$ID" \
  --name "vesper-dev-$(date +%Y%m%d-%H%M)" --query ImageId --output text)
aws ec2 wait image-available --region "$AWS_REGION" --image-ids "$AMI"
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$ID" >/dev/null
echo "AMI ready: $AMI  ->  put AMI_ID=$AMI in .env"
