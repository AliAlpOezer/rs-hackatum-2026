---
topic: dpvo-colab
updated: 2026-11-27
---

## Spatial-AI compute is Google Colab

Alp's laptop has 4 GB RAM and no GPU, so every learned method runs on Colab
(decided 2026-11-27). Rejected: local runs (no CUDA). Teammate GPU or sponsor
credits stay as upgrades for VRAM-heavy dense methods (MASt3R-SLAM, VGGT-SLAM).

## DPVO interface facts (read from princeton-vl/DPVO main, 2026-11-27)

- `DPVO.terminate()` returns `(poses, tstamps)`; poses are `x y z qx qy qz qw`
  and already **camera-to-world** (`poses.inv()` inside terminate), matching our
  TUM contract.
- `tstamps` is the list of values passed as `tstamp` to `slam(tstamp, image, K)`,
  so a runner can pass real timestamps instead of frame indices.
- Calib file: one line `fx fy cx cy [k1 k2 p1 p2]`. `image_stream` undistorts
  when more than 4 values are given and crops H, W to multiples of 16;
  `video_stream` also halves resolution and intrinsics.
- Python deps include `torch_scatter` (must match the torch/CUDA build) and three
  CUDA extensions built by `pip install .` with Eigen 3.4 in `thirdparty/`.
  The repo's env pins Python 3.10, torch 2.3.1, numpy 1.26.4.
- Weights: `models.zip` (Dropbox link in `download_models_and_data.sh`) gives
  `dpvo.pth`; skip the TartanAir pickle, it is only for training.
- `evaluate_euroc.py` uses stride 2 and Sim(3)-aligned ATE against
  `datasets/euroc_groundtruth/*.txt` (TUM, body/IMU frame, not camera frame).

## EuRoC download location

The old `robotics.ethz.ch/~asl-datasets` links are retired. ETH Research
Collection (DOI 10.3929/ethz-b-000690084) hosts only group zips: machine_hall
12.7 GB, vicon_room1 6.0 GB, vicon_room2 6.0 GB, each holding
`<group>/<seq>/<seq>.bag` and `<seq>.zip` (ASL, rooted at `mav0/`). The server
honours HTTP range requests, so `data/tools/fetch_euroc.py` uses remotezip to
pull one inner zip (MH_01_easy 1.6 GB). Bitstream UUIDs came from the DSpace API
(`/server/api/core/items/bcaf173e-.../bundles`); the HTML record page returns
429/500 to scripts. Verified by remote listing on 2026-11-27.

## evo RPE has no time window

evo 1.38 `id_pairs_from_delta` accepts frames, metres, degrees and radians;
`Unit.seconds` raises FilterException. `eval/evaluate.py` converts the window to
frames at the median timestamp spacing.

## DPVO on PyTorch 2.x needs a source patch

Newer PyTorch rejects `tensor.type()` in AT_DISPATCH macros. `envs/install_dpvo.sh`
rewrites `.type()` to `.scalar_type()` on DISPATCH lines only (lietorch.cpp uses
`device().type()` legitimately) and makes `dispatch.h` use the ScalarType
directly. Source: DPVO issue #100 (torch 2.8 guide). Builds use
`--no-build-isolation` and `TORCH_CUDA_ARCH_LIST` set to the Colab GPU.

## Status (2026-11-27)

DPVO EuRoC pipeline written and tested locally on a synthetic EuRoC sequence
(converter frame math, evaluator scale/ATE recovery, runner I/O). **Not yet run
on a GPU**: the install script and the DPVO call itself are unverified until the
Colab notebook `tracks/spatial-ai/notebooks/dpvo_euroc.ipynb` runs. Success
check: MH_01_easy ATE Sim3 within DPVO's logged 0.063-0.105 m.

## Colab moved to torch 2.14 + CUDA 13 (seen 2026-11-28)

Colab T4 runtime reported torch 2.14.1, cu130, sm 7.5. PyG publishes no
torch-scatter wheel for it, and pip's isolated source build fails ("Getting
requirements to build wheel" without torch). DPVO only needs `scatter_sum`,
`scatter_softmax`, `scatter_max(...)[0]`, so `envs/torch_scatter_shim/` provides
them in plain PyTorch; `test_shim.py` checks them against per-group loops with
DPVO's call shapes (passed on CPU torch). `install_dpvo.sh` uses a prebuilt
wheel when one exists, else the shim. It also installs `cuda-toolkit-<torch CUDA>`
via apt when Colab's nvcc major version differs from torch's; that branch and
the CUDA 13 build of DPVO's extensions are not yet verified on Colab.

## R&S dataset facts (2026-11-28)

Checked from `dataset_extracted.zip`, seq01 converted and round-tripped:
- ZED X Mini, rectified stereo, 960x600, fx = fy = 370.37, 30 fps, baseline
  0.04996 m, 601 frames per recording (about 20 s).
- `imu.csv` columns are `timestamp_ms,ax,ay,az,gx,gy,gz` (accel first, gyro in
  deg/s), irregular about 200 Hz, starts about 0.4 s before the first frame.
- `calibration.json` `imu_to_left_camera` maps IMU -> left camera
  (T_cam_imu); `rs_to_seq.py` stores its inverse as `T_imu_cam`.
- `trajectory.txt` (seq01-10): `timestamp_ms tx ty tz qx qy qz qw`, first pose at
  the origin with a small non-identity rotation, first two rows identical.
- Teams pick one task; the detailed PDF contains download credentials, never commit it.
