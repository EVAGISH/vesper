#!/usr/bin/env bash
# Build the RF-DETR training venv under scratch/, on the GPU box.
#
# Deliberately NOT in the Isaac container: that image is 40 GB and rebuilt from
# a Dockerfile, and a pip install inside it dies with the container. Deliberately
# NOT in /root either -- scratch/ is what infra/do/snapshot.sh wipes, so a
# 6 GB torch install never rides along in the disk image we pay to store.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/scratch/venv-rfdetr"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
# tensorrt/tensorrt-bench pull the TensorRT Python API, polygraphy and pycuda, which
# is how rfdetr builds and times an engine -- there is no trtexec on this box.
"$VENV/bin/pip" install "rfdetr[train,onnx,tensorrt,tensorrt-bench]"
echo
"$VENV/bin/python" -c "
import torch, rfdetr
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('rfdetr', rfdetr.__version__ if hasattr(rfdetr,'__version__') else '?')
"
echo "venv ready: $VENV"
