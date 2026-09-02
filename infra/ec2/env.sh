#!/usr/bin/env bash
# Shared config for infra scripts. Reads repo-root .env (copy .env.example).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && set -a && source "$REPO_ROOT/.env" && set +a

export AWS_REGION="${AWS_REGION:-us-east-1}"
export INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.2xlarge}"   # 1x L40S 48GB (Ada)
export TAG_NAME="${TAG_NAME:-vesper-dev}"
export SG_NAME="${SG_NAME:-vesper-dev-sg}"
export IAM_PROFILE="${IAM_PROFILE:-vesper-box}"        # instance profile: rw on $VESPER_BUCKET only
export ROOT_VOLUME_GB="${ROOT_VOLUME_GB:-200}"
export AMI_ID="${AMI_ID:-}"                            # set after infra/ami/build.sh produces one

# NOTE: fresh AWS accounts often have a vCPU quota of 0 for G-family instances.
# File the Service Quotas increase (Running On-Demand G and VT instances, >=8 vCPUs)
# before anything here will launch.
