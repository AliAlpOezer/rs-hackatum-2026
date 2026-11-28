# Data

Raw and normalized data are **not committed** (see `.gitignore`). Share them via
the team drive and place them here locally.

Every sequence follows the layout in [docs/architecture.md](../docs/architecture.md#contract-1-sequence-layout-input).
Converters from the R&S raw format go in `data/tools/`.

## R&S dataset

`dataset_extracted.zip` (7.5 GB, 20 recordings) from the challenge download.
Keep it out of git (too big for free Git LFS); share it via a team Drive folder.
Convert each recording:

```bash
python data/tools/rs_to_seq.py <dataset_extracted>/seq01 data/seq01
```

This decodes both videos to `frames/` and `frames_right/` (601 frames each for
seq01), converts the IMU to ns and rad/s, writes `calib.yaml` (no distortion,
`T_imu_cam`, baseline) and `gt.txt` (seq01-10 only), and keeps the original
timestamp strings in `timestamps_ms.txt` for the submission.
`eval/to_submission.py` turns any TUM result back into the R&S format.

## Tools

- `tools/fetch_euroc.py <seq> <dest>`: downloads one EuRoC sequence (only its own
  zip, via HTTP range requests; ETH hosts 6-13 GB group archives). Needs `remotezip`.
- `tools/zed_to_euroc.py <src> <out> --gray`: R&S ZED X Mini sequences (`left.mp4`,
  `right.mp4`, `imu.csv`, `calibration.json`, `trajectory.txt`) -> EuRoC ASL
  (`mav0/`, stereo + IMU, GT in IMU frame) plus everything ORB-SLAM3 needs per
  sequence: `timestamps.txt`, `orbslam3_{stereo,stereo_inertial,mono_inertial}.yaml`
  (from that sequence's calibration), `calibration.json`, `timestamps_ms.txt`
  (for `eval/to_submission.py`) and `conversion.yaml` (dropped frames, IMU timing,
  IMU noise source). Frames without IMU coverage are dropped (ORB-SLAM3's inertial
  loaders read past the IMU vector otherwise). IMU noise is a flagged placeholder
  unless `--imu-noise quiet`. `<src>` is one seq or a folder of `seq*`. Needs ffmpeg.
  Running it: [tracks/orbslam3](../tracks/orbslam3/README.md).
- `tools/check_stereo.py <euroc_seq>`: left/right consistency as ORB-SLAM3 sees it:
  stereo match rate, vertical offset, swapped cameras, frame offset between the videos.
  A healthy pair gives ~40-90 % (synthetic: 89 %); swapped gives ~6 %.
- `tools/synth_zed.py <out>`: synthetic sequence in the raw R&S format (stereo
  room + IMU + trajectory), for testing the pipeline without the dataset.
- `tools/euroc_to_seq.py <euroc_seq> data/<seq>`: EuRoC ASL -> shared layout,
  ground truth converted from IMU frame to camera frame.

## Sequence log

| Sequence | Source | Length | Motion | GT | Notes |
|---|---|---|---|---|---|
| | | | | | |

Public sequences for sanity checks: EuRoC MAV (MH_01), TUM-VI (room1).
