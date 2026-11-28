---
topic: orbslam3-rs
updated: 2026-11-28
---

## Pipeline (issue #1)

`data/tools/zed_to_euroc.py` -> `tracks/orbslam3/run.sh <seq> <mode>` ->
`results/<seq>/orbslam3_<mode>.txt` (TUM, T_world_cam, s) -> `eval/evaluate.py`.
Modes: stereo, stereo_inertial (main), mono_inertial. Build: `tracks/orbslam3/setup.sh`
(ORB-SLAM3 4452a3c + Pangolin v0.9.1, outside git, GPLv3).

## Facts read from ORB-SLAM3 source (4452a3c)

- Settings v1.0 types values by spelling: a real key written `30` aborts. The
  generator writes reals with a dot.
- `Camera.type: "Rectified"` + stereo segfaults (null `originalCalib2_` printed in
  `operator<<(Settings)`). Use PinHole, zero distortion, `Stereo.T_c1_c2`.
- `IMU.fastInit` is read only by the legacy parser; `patches/0002` adds it to v1.0.
  It skips the `|avgA diff| < 0.5 m/s^2` "not enough acceleration" gate in
  `Tracking::StereoInitialization`.
- `*_inertial_euroc` need `imu[0] <= first frame` and `last frame < imu[-1]`.
- `SaveTrajectoryEuRoC`: ns, body pose for IMU sensors, largest map only.
- Build on GCC 13: C++17 for Pangolin 0.9 and `LoopClosing::mnFullBAIdx` bool -> int
  (`patches/0001`). Pangolin needs `CMAKE_INSTALL_RPATH=$ORIGIN` when installed
  to a local prefix (RUNPATH is not transitive).

## Verified (2026-11-28, no real data in that environment)

Synthetic sequence from `data/tools/synth_zed.py`, all three modes run end to end,
ATE SE3 2-12 mm, scale ~1.00. Real-data runs pending; log them in `tracks/orbslam3/notes.md`.
