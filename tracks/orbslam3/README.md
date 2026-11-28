# Track: ORB-SLAM3 (Teammate A)

Goal: a reliable classical baseline. ORB-SLAM3 on the R&S ZED X Mini data,
exported in the shared TUM format. The data is rectified stereo + IMU, so the
main mode is **stereo-inertial** (metric scale from stereo, robustness from the
IMU); stereo and mono-inertial run from the same converted folder for comparison.

## Quick start

```bash
tracks/orbslam3/setup.sh --deps        # once: apt deps, Pangolin, ORB-SLAM3 at the pinned commit + patches
uv run python data/tools/zed_to_euroc.py ../dataset_extracted ../dataset_euroc --gray
tracks/orbslam3/run.sh ../dataset_euroc/seq01 stereo_inertial
uv run python data/tools/euroc_to_seq.py ../dataset_euroc/seq01 data/seq01   # gt.txt, camera frame
uv run python eval/evaluate.py data/seq01/gt.txt results/seq01/*.txt --metric
```

All sequences at once (convert, stereo check, three modes, benchmark table in
`<work>/reports/summary.md`):

```bash
tracks/orbslam3/run_all.sh ../dataset_extracted ../rs_work
```

`run.sh <seq> [stereo|stereo_inertial|mono_inertial]` writes
`results/<seq>/orbslam3_<mode>.txt` (TUM, `T_world_cam`, seconds) and keeps the
raw ORB-SLAM3 output and log in `results/<seq>/raw/`. It prints what goes into
`notes.md`: init delay, frames tracked, "not enough acceleration" count, resets,
maps, IMU init stages.

No dataset at hand? `data/tools/synth_zed.py /tmp/synth/seq01` writes a synthetic
sequence in the raw R&S format (textured room, stereo, IMU) that runs through the
same commands. All three modes score millimetre ATE on it (see `notes.md`).

## What each piece does

- `setup.sh`: clones ORB-SLAM3 (GPLv3, never committed) at a pinned commit and
  Pangolin v0.9.1 into this folder (gitignored), applies `patches/*.patch`, builds.
  - `0001-build-cxx17.patch`: C++17 for Pangolin 0.9; `LoopClosing::mnFullBAIdx` bool -> int.
  - `0002-fastinit-v1-settings.patch`: reads `IMU.fastInit` from `File.version: "1.0"`
    yamls. Upstream only reads it in the legacy parser. `1` skips the
    stereo-inertial "not enough acceleration" check at initialization.
- `data/tools/zed_to_euroc.py`: per sequence, EuRoC `mav0/`, `timestamps.txt`,
  `orbslam3_{stereo,stereo_inertial,mono_inertial}.yaml` generated from that
  sequence's `calibration.json`, a copy of the calibration, `conversion.yaml`.
- `orbslam3_to_tum.py`: ORB-SLAM3 EuRoC output (ns; IMU body pose for inertial
  modes) -> TUM `T_world_cam` seconds via `T_wc = T_wb * T_b_c1`.

## Settings choices (generated, not hand-edited)

| Key | Value | Why |
|---|---|---|
| `Camera.type` | `PinHole`, distortion 0, `Stereo.T_c1_c2` = [I, (b,0,0)] | `Rectified` segfaults for stereo in the stock build (null Camera2 in `Settings` printing) |
| `Camera1/2.*` | per-sequence fx, fy, cx, cy | fx differs between sequences (370.285 to 370.457) |
| `Camera.RGB` | 0 | PNGs come back BGR from `cv::imread`; use `--gray` anyway |
| `Stereo.ThDepth` | fx / 7.6 px ≈ 48.5 baselines (2.4 m) | same minimum disparity as EuRoC's 60 x 0.11 m |
| `ORBextractor.nFeatures` | EuRoC count x image area: 1900 stereo, 1600 mono | 960x600 vs 752x480 |
| `IMU.T_b_c1` | `T_BS` of `mav0/cam0/sensor.yaml` = inv(`imu_to_left_camera`) | left camera pose in the IMU frame |
| `IMU.Frequency` | measured median (~198 Hz) | IMU is irregular, 5 ms median |
| `IMU.Noise*`, `*Walk` | **placeholders** (EuRoC ADIS16448) | `--imu-noise quiet` / `--imu-noise-scale` to change; flagged in the yaml |

## Still to do

1. Run seq01, seq10, seq20 (and the rest) in all three modes; log init delay and
   failures per sequence in `notes.md`.
2. Tune IMU noise: a static segment if any sequence has one, else scale upward.
3. Camera-IMU time offset (assumed 0).
4. Failure cases (low texture, fast rotation, close range) feed the spatial-AI track.

## Layout

```
tracks/orbslam3/
  setup.sh            clone + build ORB-SLAM3 (pinned commit) and Pangolin
  patches/            our changes to ORB-SLAM3, applied by setup.sh
  run.sh              converted seq -> results/<seq>/orbslam3_<mode>.txt
  orbslam3_to_tum.py  ORB-SLAM3 EuRoC trajectory -> TUM T_world_cam seconds
  notes.md            experiment log
```
