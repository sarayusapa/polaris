#!/usr/bin/env bash
# Build a conda env for PolaRiS matching Isaac Sim 5.1's supported stack (torch 2.7 / CUDA 12.8).
# Runs unattended; logs each stage with an explicit exit code so failures are diagnosable.
set -uo pipefail
ENVN="${1:-polaris}"
ROOT="/home/sra/sarayu/polaris"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"

stage () { echo; echo "==== [$(date +%H:%M:%S)] STAGE: $* ===="; }
CR () { conda run -n "$ENVN" --no-capture-output "$@"; }

stage "create env python=3.11"
conda create -y -n "$ENVN" python=3.11 || { echo "STAGE_FAIL create $?"; exit 10; }

stage "conda cuda-toolkit 12.8 (nvcc + headers, native)"
conda install -y -n "$ENVN" -c nvidia cuda-toolkit=12.8 || { echo "STAGE_FAIL cuda $?"; exit 11; }

stage "pip torch 2.7.1 + torchvision 0.22.1 (cu128)"
CR pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
  || { echo "STAGE_FAIL torch $?"; exit 12; }

stage "pip build tools + setuptools<81 (flatdict needs pkg_resources)"
CR pip install "setuptools<81" wheel ninja || { echo "STAGE_FAIL buildtools $?"; exit 13; }

stage "pip flatdict==4.0.1 without build isolation (uses env setuptools<81)"
CR pip install --no-build-isolation flatdict==4.0.1 || { echo "STAGE_FAIL flatdict $?"; exit 14; }

stage "pip isaaclab[all,isaacsim]==2.3.0 (nvidia index) — large, ~10GB"
CR pip install "isaaclab[all,isaacsim]==2.3.0" --extra-index-url https://pypi.nvidia.com \
  || { echo "STAGE_FAIL isaaclab $?"; exit 15; }

stage "pip polaris runtime deps"
CR pip install "mediapy>=1.2.4" "opencv-python>=4.11.0.86" "plyfile>=1.1.3" "tyro>=0.9.17" \
  || { echo "STAGE_FAIL polarisdeps $?"; exit 16; }

stage "pip editable local packages (splat kernels + openpi-client)"
CR pip install -e ./src/diff-surfel-rasterization -e ./src/simple-knn \
     -e ./third_party/openpi/packages/openpi-client \
  || { echo "STAGE_FAIL localpkgs $?"; exit 17; }

stage "pip editable polaris (no-deps; deps already satisfied)"
CR pip install -e . --no-deps || { echo "STAGE_FAIL polaris $?"; exit 18; }

stage "sanity: torch cuda build + polaris import"
CR python - <<'PY' || { echo "STAGE_FAIL sanity $?"; exit 19; }
import torch
print("torch", torch.__version__, "cuda-build", torch.version.cuda, "avail", torch.cuda.is_available())
import polaris, polaris.policy
print("clients:", list(polaris.policy.InferenceClient.REGISTERED_CLIENTS.keys()))
PY

echo; echo "==== [$(date +%H:%M:%S)] BUILD_CONDA_OK ===="
