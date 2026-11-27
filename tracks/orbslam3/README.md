# Track: ORB-SLAM3 (Teammate A)

Goal: a reliable classical baseline. ORB-SLAM3 mono-inertial on the handheld
data, exported in the shared TUM format.

## Plan

1. **Build** in Linux, WSL2 or Docker (native Windows is not worth the time).
   ORB-SLAM3 is GPLv3: clone it via `setup.sh` or a submodule, do not copy its
   source into this repo.
2. **Sanity run** on EuRoC MH_01 in mono-inertial mode; check ATE with `eval/`.
3. **Config for R&S data:** write `config/rs_device.yaml` from `data/<seq>/calib.yaml`
   (intrinsics, distortion, `T_b_c1`, IMU noise and random walk, IMU rate, fps).
4. **Run** on R&S sequences, save `KeyFrameTrajectory` and full frame
   trajectory to `results/<seq>/orbslam3_mi.txt` (TUM).
5. **Tune:** IMU noise from a static segment, `ORBextractor.nFeatures`,
   initialization (needs enough parallax and excitation); log failures.

## Report back

- Init time, tracking-lost count, reinit behavior per sequence.
- Where it fails (low texture, fast rotation, close range) - these cases are
  what the spatial-AI track should target.

## Layout

```
tracks/orbslam3/
  setup.sh      clone + build ORB-SLAM3 (pinned commit)
  Dockerfile    reproducible build
  config/       camera + IMU yaml per device
  run.sh        data/<seq> -> results/<seq>/orbslam3_mi.txt
  notes.md      experiment log
```
