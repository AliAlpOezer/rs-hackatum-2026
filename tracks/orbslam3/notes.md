# ORB-SLAM3 experiment log

## Runs

Init = first saved pose after the first frame. Tracked = frames in the saved
(largest) map. ATE/RPE from `eval/evaluate.py --metric` against `gt.txt`
(camera frame). Real-data rows still to fill: seq01, seq10, seq20 first.

| Date | Seq | Mode | Settings | Init | Tracked | Resets / maps | ATE SE3 [m] | RPE 1 s [m] | Scale | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-28 | synth seq01 | stereo_inertial | default, placeholder noise | 0.07 s | 598/600 | 0 / 1 | 0.0028 | 0.0037 | 1.0006 | VIBA1+2 done, exit 0 |
| 2026-11-28 | synth seq01 | stereo | default | 0.00 s | 600/600 | 0 / 1 | 0.0019 | 0.0025 | 1.0014 | exit 139 in viewer teardown after saving |
| 2026-11-28 | synth seq01 | mono_inertial | default, placeholder noise | 0.07 s | 598/600 | 0 / 1 | 0.0124 | 0.0094 | 0.9964 | metric scale from IMU; exit 139 in teardown |
| 2026-11-28 | synth seq01 | stereo_inertial | `--imu-noise quiet --imu-noise-scale 2 --fast-init` | 0.07 s | 598/600 | 0 / 1 | | | | fastInit read by patched build |
| | seq01 | stereo_inertial | | | | | | | | |
| | seq10 | stereo_inertial | | | | | | | | |
| | seq20 | stereo_inertial | | | | | | | | |

Synthetic = `data/tools/synth_zed.py` (20 s, 601 frames, textured room, IMU rotated
and offset from the camera, irregular ~5 ms IMU ending 1.3 ms before the last
frame). It checks formats and frame conventions, not real-world robustness: there is
no motion blur, rolling shutter or exposure change. Control: leaving the
stereo-inertial output as body poses (no `T_b_c1`) gives RPE 1 s 0.645 m instead of 0.004 m.

## Findings (2026-11-28, ORB-SLAM3 4452a3c, Ubuntu 24.04, GCC 13, OpenCV 4.6, Pangolin 0.9.1)

- **IMU past the end.** `stereo_inertial_euroc.cc:170` and `mono_inertial_euroc.cc:178`
  loop `while(imu[k] <= cam[ni])` without bounds checks. On the R&S data the IMU ends
  ~1 ms before the last frame, so the last frame reads past the vector. In our build,
  the old converter output did not crash: `LoadIMU` reserves 5000 entries and a 20 s
  sequence has ~4000, so the read lands in reserved, uninitialized memory. That is
  undefined behaviour; it may crash elsewhere. The converter now keeps only frames
  with `imu[0] <= t < imu[-1]` (strict, since `<=` also overruns on an exact tie)
  and logs the count in `conversion.yaml` (1 trailing frame on synth).
- **`Camera.type: "Rectified"` segfaults for stereo.** `Settings::readCamera2` never
  sets `originalCalib2_` for Rectified, and `operator<<(Settings)` dereferences it
  when printing. The generated yamls use `PinHole` with zero distortion and
  `Stereo.T_c1_c2` = [I, (b,0,0)]. ORB-SLAM3's internal stereoRectify then maps every
  pixel to itself (checked with the same OpenCV call: max offset 3e-5 px); if a right
  camera's cx differed slightly, it would unify the two instead.
- **`IMU.fastInit` is ignored in v1.0 yamls.** Upstream reads it only in
  `Tracking::ParseIMUParamFile` (legacy format); `newParameterLoader` drops it.
  `patches/0002` fixes that. The stereo-inertial init check needs
  `|avgA_cur - avgA_last| >= 0.5 m/s^2` between two consecutive frames, which gentle
  handheld motion can fail for a long time ("not enough acceleration" in the log,
  counted by `run.sh`).
- **Viewer teardown crash.** `stereo_euroc` and `mono_inertial_euroc` (viewer on, under
  `xvfb-run`) exit with SIGSEGV after "End of saving trajectory" (Qt timers stopped
  from another thread). The trajectory is complete; `run.sh` reports and continues.
- **Output.** `SaveTrajectoryEuRoC` writes ns (`1e9*t`, fixed, 6 decimals) and, for
  IMU sensors, `T_world_body`; only the largest map is saved, so a reset loses
  the frames before it from the file.

## Open items

- IMU noise values are placeholders. The synthetic quiet-window estimate is flagged
  "NOT static" (no still phase in that motion); check whether real sequences start still.
- Camera-IMU time offset assumed 0. Cross-correlating the gyro with the rotation rate
  of `trajectory.txt` would give an estimate (but the SDK trajectory already fuses the IMU).
- `trajectory.txt` may be ZED SDK tracking, not independent ground truth; to confirm
  with the data owner. GT velocities and biases are zero placeholders.
- Converter assumptions were checked on seq01 only; verify on a second sequence.
- World frame: ORB-SLAM3 inertial world is gravity-aligned at the first keyframe, not
  the R&S trajectory frame. `eval/` aligns; check whether the R&S scoring does.
