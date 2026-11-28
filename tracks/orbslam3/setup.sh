#!/usr/bin/env bash
# Clone and build ORB-SLAM3 at a pinned commit, plus Pangolin, outside git.
#
#   tracks/orbslam3/setup.sh          # build into tracks/orbslam3/{ORB_SLAM3,Pangolin}
#   tracks/orbslam3/setup.sh --deps   # first apt-install the build dependencies (Ubuntu 22.04/24.04)
#
# ORB-SLAM3 is GPLv3: its source never enters this repo. Our changes to it live in
# patches/*.patch and are applied here, in order, idempotently:
#   0001-build-cxx17.patch           -std=c++11 -> c++17 (Pangolin 0.9 headers need C++17)
#   0002-fastinit-v1-settings.patch  read IMU.fastInit from File.version 1.0 yaml files
#                                    (upstream only reads it in the legacy parser)
# Env overrides: ORB_SLAM3_DIR, PANGOLIN_DIR, JOBS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_COMMIT=4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4   # UZ-SLAMLab/ORB_SLAM3 master, v1.0 (2022-02-10)
PANGOLIN_TAG=v0.9.1
ORB_DIR="${ORB_SLAM3_DIR:-$HERE/ORB_SLAM3}"
PANGOLIN_DIR="${PANGOLIN_DIR:-$HERE/Pangolin}"
JOBS="${JOBS:-$(nproc)}"

if [[ "${1:-}" == "--deps" ]]; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git libeigen3-dev libopencv-dev \
    libboost-serialization-dev libssl-dev libglew-dev libepoxy-dev libgl1-mesa-dev xvfb ffmpeg
fi

# Pangolin (viewer library; installed into a local prefix, no sudo)
if [[ ! -d "$PANGOLIN_DIR/install/lib/cmake/Pangolin" ]]; then
  [[ -d "$PANGOLIN_DIR" ]] || git clone --depth 1 --branch "$PANGOLIN_TAG" \
    https://github.com/stevenlovegrove/Pangolin.git "$PANGOLIN_DIR"
  cmake -S "$PANGOLIN_DIR" -B "$PANGOLIN_DIR/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PANGOLIN_DIR/install" -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_TESTS=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN'  # libs find each other
  cmake --build "$PANGOLIN_DIR/build" -j "$JOBS"
  cmake --install "$PANGOLIN_DIR/build"
fi

# ORB-SLAM3 source at the pinned commit
if [[ ! -d "$ORB_DIR/.git" ]]; then
  git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$ORB_DIR"
  git -C "$ORB_DIR" checkout -q "$ORB_COMMIT"
fi
head="$(git -C "$ORB_DIR" rev-parse HEAD)"
[[ "$head" == "$ORB_COMMIT" ]] || echo "WARNING: $ORB_DIR is at $head, expected $ORB_COMMIT" >&2

for p in "$HERE"/patches/*.patch; do
  if git -C "$ORB_DIR" apply --reverse --check "$p" 2>/dev/null; then
    echo "patch already applied: $(basename "$p")"
  elif git -C "$ORB_DIR" apply --check "$p" 2>/dev/null; then
    git -C "$ORB_DIR" apply "$p" && echo "patch applied: $(basename "$p")"
  else
    echo "ERROR: $(basename "$p") does not apply to $ORB_DIR (local edits?). Resolve by hand." >&2
    exit 1
  fi
done

cd "$ORB_DIR"
for lib in DBoW2 g2o; do
  cmake -S "Thirdparty/$lib" -B "Thirdparty/$lib/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "Thirdparty/$lib/build" -j "$JOBS"
done
[[ -f Vocabulary/ORBvoc.txt ]] || tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
# RPATH so the examples find the locally installed Pangolin without LD_LIBRARY_PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPangolin_DIR="$PANGOLIN_DIR/install/lib/cmake/Pangolin" \
  -DCMAKE_BUILD_RPATH="$PANGOLIN_DIR/install/lib"
cmake --build build -j "$JOBS"

for exe in Examples/Stereo/stereo_euroc Examples/Stereo-Inertial/stereo_inertial_euroc \
           Examples/Monocular-Inertial/mono_inertial_euroc; do
  [[ -x "$exe" ]] || { echo "ERROR: $exe was not built" >&2; exit 1; }
done
echo "ORB-SLAM3 ready in $ORB_DIR"
