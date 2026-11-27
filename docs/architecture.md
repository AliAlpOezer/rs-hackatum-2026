# Architecture

Four people build in parallel. The seams below keep the tracks independent and
make results comparable.

```
            data/<seq>/                       tracks/*/                eval/
 video ─┐                                ┌─ orbslam3 ─────┐
 imu   ─┼─► normalized sequence ─────────┼─ spatial-ai ───┼─► results/<seq>/<method>.txt ─► evo ─► table
 calib ─┘   (one layout for all)         └─ fusion (later)┘            (TUM format)
                                                                │
                                                                └─► patch stitching / demo
```

## Contract 1: sequence layout (input)

Every pipeline reads the same folder, produced once by the data owner:

```
data/<seq>/
  frames/<timestamp_ns>.png   or video.mp4 + frames.csv (timestamp_ns,frame_idx)
  imu.csv                     timestamp_ns,wx,wy,wz,ax,ay,az   (rad/s, m/s^2, device frame)
  calib.yaml                  intrinsics, distortion, T_imu_cam (camera pose in IMU frame), time offset
  gt.txt                      optional ground truth, TUM format
```

Converters from the raw R&S format live in `data/tools/`, never inside a track.

## Contract 2: trajectory output

Every method writes `results/<seq>/<method>.txt` in **TUM format**:

```
timestamp_s tx ty tz qx qy qz qw
```

- Pose is **camera-to-world** (`T_world_cam`), camera frame of the input video.
- Timestamps in seconds, same clock as `frames`.
- Monocular methods without scale: write as-is; eval aligns with Sim(3) and we
  report that it needed scale.

`results/` is gitignored except `results/summary.md`.

## Contract 3: evaluation

`eval/` runs [evo](https://github.com/MichaelGrupp/evo) on every
`results/<seq>/*.txt` against `gt.txt`:

- **ATE** (absolute trajectory error) after SE(3) or Sim(3) alignment.
- **RPE over 1 s / 0.5 m** - the number that matters for stitching radar patches.
- Runtime and failure rate (tracking lost, reinit count).

Without ground truth: loop-closure consistency, AprilTag/checkerboard reprojection,
and visual quality of stitched patches.

## Decisions

| Decision | Choice | Rejected | Why |
|---|---|---|---|
| Trajectory format | TUM text | KITTI, ROS bags | Timestamps included, evo-native, every SLAM repo can write it |
| Language for glue | Python 3.11 | C++ | ORB-SLAM3 stays C++, all glue and eval in Python; PyTorch needs 3.11 |
| Repo shape | Monorepo, one folder per track | One repo per track | One eval, one data layout, easy cross-track fusion |
| Visibility | Public | Private | Open for judges and the team; the R&S deck was shared publicly, so nothing here is confidential |

Note: ORB-SLAM3 is GPLv3. Keep it as an external build (submodule or script
that clones it) so the rest of the repo can be licensed freely.
