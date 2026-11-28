# ORB-SLAM3 results: stereo_inertial

One folder per sequence:

| File | Content |
|------|---------|
| `f_<seq>.txt`  | Pose of **every tracked frame** (full trajectory). Frames tracked while lost or in a discarded map are missing. |
| `kf_<seq>.txt` | Pose of **keyframes only** (sparse subset, after local BA / loop closure refinement). |
| `gt.txt`       | Ground truth, copied unchanged from `dataset_extracted/<seq>/trajectory.txt`. Timestamps in **milliseconds**. |
| `gt_s.txt`     | Same ground truth with timestamps converted to **seconds**, matching f_/kf_ files. Use this one for evaluation. |
| `<seq>.yaml`   | Exact settings file used for the run. |
| `<seq>.log`    | Full ORB-SLAM3 console output. |

All trajectory files are TUM format: `timestamp[s] tx ty tz qx qy qz qw`.

- Estimated poses (f_, kf_) and ground truth are both the **left camera** pose (T_world_cam).
- ORB-SLAM3 world frame = first frame of the biggest map, so the trajectory starts at identity.
  For stereo the scale is metric; still align before comparing (SE3, or Sim3 for mono).
- Only the **biggest map** is saved. If tracking was lost and a new map started, poses of the
  smaller maps are dropped, so f_ can have fewer lines than there are frames.

Example evaluation with evo:

    evo_ape tum seq01/gt_s.txt seq01/f_seq01.txt -a --plot
    evo_ape tum seq01/gt_s.txt seq01/kf_seq01.txt -a
