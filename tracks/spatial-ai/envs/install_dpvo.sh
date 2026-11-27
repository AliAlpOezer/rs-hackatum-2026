#!/usr/bin/env bash
# Install DPVO into the current Python (Colab or any CUDA Linux box).
#   bash install_dpvo.sh <install_dir> [wheel_cache_dir]
# With a wheel cache (e.g. on Google Drive), the ~10 min CUDA build runs once
# per torch/CUDA/GPU combination; later sessions install the cached wheel.
set -euo pipefail

DPVO_COMMIT=0ac95b656d1fda91c271d2a106460d19ad966fc7  # main as of 2026-09-20
DIR=${1:-/content/DPVO}
CACHE=${2:-}

read -r TORCH CUDA ARCH < <(python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU: switch the Colab runtime to GPU"
major, minor = torch.cuda.get_device_capability()
print(torch.__version__.split("+")[0], "cu" + torch.version.cuda.replace(".", ""), f"{major}.{minor}")
PY
)
echo "torch $TORCH, $CUDA, GPU arch $ARCH"

# 1. Source at a pinned commit (no submodules: viewer and classic loop closure are not needed).
if [ ! -d "$DIR/.git" ]; then
  git clone -q https://github.com/princeton-vl/DPVO.git "$DIR"
fi
git -C "$DIR" fetch -q origin "$DPVO_COMMIT"
git -C "$DIR" checkout -q "$DPVO_COMMIT"

# 2. Python dependencies; torch_scatter must match the installed torch + CUDA.
pip install -q torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
pip install -q numba tqdm einops pypose kornia plyfile evo yacs tensorboard opencv-python-headless pyyaml

# 3. Weights (dpvo.pth). Skip the TartanAir training pickle.
if [ ! -f "$DIR/dpvo.pth" ]; then
  wget -q -O "$DIR/models.zip" "https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t&dl=1"
  unzip -q -o "$DIR/models.zip" -d "$DIR" && rm "$DIR/models.zip"
fi

# 4. CUDA extensions, from cache if possible.
TAG="torch${TORCH}-${CUDA}-sm${ARCH/./}"
WHEEL_DIR=${CACHE:+$CACHE/dpvo-$TAG}
if [ -n "$WHEEL_DIR" ] && ls "$WHEEL_DIR"/dpvo-*.whl >/dev/null 2>&1; then
  echo "installing cached wheel from $WHEEL_DIR"
  pip install -q --force-reinstall --no-deps "$WHEEL_DIR"/dpvo-*.whl
else
  # PyTorch 2.x removed the old .type() overload used by the dispatch macros.
  find "$DIR/dpvo" \( -name '*.cu' -o -name '*.cpp' \) -exec sed -i '/DISPATCH/ s/\.type()/.scalar_type()/g' {} +
  sed -i 's/::detail::scalar_type(the_type)/the_type/' "$DIR/dpvo/lietorch/include/dispatch.h"
  if [ ! -d "$DIR/thirdparty/eigen-3.4.0" ]; then
    wget -q -O /tmp/eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q /tmp/eigen.zip -d "$DIR/thirdparty"
  fi
  export TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS=$(nproc)
  ( cd "$DIR" && pip wheel -q --no-build-isolation --no-deps . -w dist )
  pip install -q --force-reinstall --no-deps "$DIR"/dist/dpvo-*.whl
  if [ -n "$WHEEL_DIR" ]; then mkdir -p "$WHEEL_DIR" && cp "$DIR"/dist/dpvo-*.whl "$WHEEL_DIR"/; fi
fi

python -c "import torch, cuda_corr, cuda_ba, lietorch_backends; from dpvo.dpvo import DPVO; print('DPVO OK')"
