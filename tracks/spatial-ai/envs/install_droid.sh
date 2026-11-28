#!/usr/bin/env bash
# Install DROID-SLAM into the current Python (Colab or any CUDA Linux box).
#   bash install_droid.sh <install_dir> [wheel_cache_dir]
# With a wheel cache (e.g. on Google Drive), the CUDA builds of lietorch and
# droid_backends run once per torch/CUDA/GPU combination.
set -euo pipefail
DROID_COMMIT=2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99  # main as of 2025-05-05
ENVS_DIR=$(cd "$(dirname "$0")" && pwd)
DIR=${1:-/content/DROID-SLAM}
CACHE=${2:-}
read -r TORCH CUDA ARCH < <(python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU: switch the Colab runtime to GPU"
major, minor = torch.cuda.get_device_capability()
print(torch.__version__.split("+")[0], "cu" + torch.version.cuda.replace(".", ""), f"{major}.{minor}")
PY
)
echo "torch $TORCH, $CUDA, GPU arch $ARCH"

# 1. Source at a pinned commit with lietorch + its eigen submodule (pytorch_scatter is not built).
if [ ! -d "$DIR/.git" ]; then
  git clone -q https://github.com/princeton-vl/DROID-SLAM.git "$DIR"
fi
git -C "$DIR" fetch -q origin "$DROID_COMMIT"
git -C "$DIR" checkout -q "$DROID_COMMIT"
git -C "$DIR" submodule update -q --init --recursive thirdparty/lietorch

# 2. Python dependencies. torch_scatter: prebuilt wheel if PyG has one, else the shim.
if pip install -q --only-binary :all: torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html" 2>/dev/null; then
  echo "torch_scatter: prebuilt wheel"
else
  pip install -q "$ENVS_DIR/torch_scatter_shim"
  echo "torch_scatter: no wheel for torch ${TORCH}+${CUDA}, installed pure-PyTorch shim"
fi
pip install -q tqdm evo scipy gdown opencv-python-headless pyyaml

# 3. Weights (droid.pth, Google Drive link from the DROID-SLAM README).
[ -f "$DIR/droid.pth" ] || gdown -q 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O "$DIR/droid.pth"

# 4. CUDA extensions, from cache if possible.
TAG="torch${TORCH}-${CUDA}-sm${ARCH/./}"
WHEEL_DIR=${CACHE:+$CACHE/droid-$TAG}
if [ -n "$WHEEL_DIR" ] && ls "$WHEEL_DIR"/lietorch-*.whl "$WHEEL_DIR"/droid_backends-*.whl >/dev/null 2>&1; then
  echo "installing cached wheels from $WHEEL_DIR"
  pip install -q --force-reinstall --no-deps "$WHEEL_DIR"/*.whl
else
  LT="$DIR/thirdparty/lietorch"
  # Newer PyTorch: no ::detail::scalar_type(ScalarType) and no .type() in dispatch macros.
  sed -i 's/::detail::scalar_type(the_type)/the_type/' "$LT/lietorch/include/dispatch.h"
  find "$LT/lietorch" "$DIR/src" \( -name '*.cu' -o -name '*.cpp' \) -exec sed -i '/DISPATCH/ s/\.type()/.scalar_type()/g' {} +
  # lietorch hard-codes sm_60..sm_75; CUDA 13 cannot build sm_60/70. Build for this GPU only.
  sed -i "/-gencode=arch=compute_/d" "$LT/setup.py"
  TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
  NVCC_VER=$( (nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null) | sed -n 's/.*release \([0-9.]*\).*/\1/p')
  echo "nvcc ${NVCC_VER:-missing}, torch built with CUDA $TORCH_CUDA_VER"
  if [ "${NVCC_VER%%.*}" != "${TORCH_CUDA_VER%%.*}" ]; then
    PKG="cuda-toolkit-${TORCH_CUDA_VER%%.*}-${TORCH_CUDA_VER#*.}"
    echo "installing $PKG to match torch"
    apt-get update -qq && apt-get install -y -qq "$PKG" >/dev/null
    export CUDA_HOME="/usr/local/cuda-${TORCH_CUDA_VER}" PATH="/usr/local/cuda-${TORCH_CUDA_VER}/bin:$PATH"
    nvcc --version | tail -1
  fi
  export TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS=$(nproc)
  rm -rf /tmp/droid_wheels && mkdir -p /tmp/droid_wheels
  pip wheel -q --no-build-isolation --no-deps "$LT" -w /tmp/droid_wheels
  pip wheel -q --no-build-isolation --no-deps "$DIR" -w /tmp/droid_wheels
  pip install -q --force-reinstall --no-deps /tmp/droid_wheels/*.whl
  if [ -n "$WHEEL_DIR" ]; then mkdir -p "$WHEEL_DIR" && cp /tmp/droid_wheels/*.whl "$WHEEL_DIR"/; fi
fi
python -c "import torch, lietorch, droid_backends; print('DROID-SLAM OK')"
