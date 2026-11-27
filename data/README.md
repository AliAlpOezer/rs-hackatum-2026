# Data

Raw and normalized data are **not committed** (see `.gitignore`). Share them via
the team drive and place them here locally.

Every sequence follows the layout in [docs/architecture.md](../docs/architecture.md#contract-1-sequence-layout-input).
Converters from the R&S raw format go in `data/tools/`.

## Tools

- `tools/fetch_euroc.py <seq> <dest>`: downloads one EuRoC sequence (only its own
  zip, via HTTP range requests; ETH hosts 6-13 GB group archives). Needs `remotezip`.
- `tools/zed_to_euroc.py <src> <out>`: R&S ZED X Mini sequences (`left.mp4`,
  `right.mp4`, `imu.csv`, `calibration.json`, `trajectory.txt`) -> EuRoC ASL
  (`mav0/`, stereo + IMU, GT in IMU frame). `<src>` is one seq or a folder of
  `seq*`. Needs ffmpeg.
- `tools/euroc_to_seq.py <euroc_seq> data/<seq>`: EuRoC ASL -> shared layout,
  ground truth converted from IMU frame to camera frame.

## Sequence log

| Sequence | Source | Length | Motion | GT | Notes |
|---|---|---|---|---|---|
| | | | | | |

Public sequences for sanity checks: EuRoC MAV (MH_01), TUM-VI (room1).
