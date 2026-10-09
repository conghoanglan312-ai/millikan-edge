#!/bin/bash
# Host toolchain for millikan-edge inside WSL2 (Ubuntu 22.04, aarch64, Python 3.10).
set -e
echo "== start $(date)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv python3-pip libgl1 libglib2.0-0 build-essential cmake > /dev/null
[ -d ~/venvs/rknn ] || python3 -m venv ~/venvs/rknn
. ~/venvs/rknn/bin/activate
pip install -q -U pip
B=https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit2/packages/arm64
mkdir -p ~/rknn-pkg && cd ~/rknn-pkg
curl -sSfLO $B/arm64_requirements_cp310.txt
curl -sSfLO $B/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install -r arm64_requirements_cp310.txt
pip install rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
python -c "from rknn.api import RKNN; import torch, onnx, numpy; print('rknn-toolkit2 OK | torch', torch.__version__, '| onnx', onnx.__version__, '| numpy', numpy.__version__)"
pip freeze > ~/rknn-pkg/freeze.txt
echo "== done $(date)"
