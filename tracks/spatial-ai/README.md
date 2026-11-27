# Track: Spatial AI (Alp)

Question this track answers: **can learned 3D perception place every radar
glimpse better than classical VI-SLAM on a handheld?** The classical baseline
(ORB-SLAM3) is Teammate A's; this track competes with it, then complements it.

The full hour-by-hour plan is in [PLAN.md](PLAN.md). Experiment log: `notes.md`.

## Why learned methods might win here

A handheld screener sweeps slowly, close to a body or surface, often over
low-texture material (clothing, walls). That is where sparse feature SLAM
struggles to initialize and track. Learned methods add:

1. **Robust tracking** without hand-crafted features (dense or patch-based matching).
2. **Dense geometry**: per-pixel depth and surface normals, not just pose. For
   radar, the distance and angle to the surface matter as much as the pose.
3. **Priors for scale** (metric depth heads), which IMU can then confirm.

## Candidate methods (repos checked 2026-11-27 via GitHub API)

| Method | Type | IMU | License | Notes |
|---|---|---|---|---|
| [DPVO](https://github.com/princeton-vl/DPVO) | Learned sparse-patch VO (+ DPV-SLAM) | no | MIT | Light, fast, low VRAM. **First baseline.** |
| [VGGT-SLAM 2.0](https://github.com/MIT-SPARK/VGGT-SLAM) | Feed-forward dense SLAM, submaps | no | BSD-2 (VGGT weights have their own license) | Strong on indoor 7-Scenes; VGGT about 60 frames per 24 GB GPU, hence submaps |
| [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) | Feed-forward dense SLAM | no | Custom (MASt3R weights non-commercial) | Strong on real handheld/aerial data; heavy on VRAM |
| [MASt3R-Fusion](https://github.com/GREAT-WHU/MASt3R-Fusion) | MASt3R + IMU (+GNSS) factor graph | **yes** | None stated | Closest ready-made match to video + IMU; code is public, maturity unknown |
| [MapAnything](https://github.com/facebookresearch/map-anything) | Feed-forward metric reconstruction | can take poses as input | Apache-2.0 | Metric scale + dense depth per chunk; good for the stitching surface |
| [CUT3R](https://github.com/CUT3R/CUT3R) | Streaming recurrent 3D model | no | Custom | Online metric pointmaps; known to drift |
| [VGGT](https://github.com/facebookresearch/vggt) | Feed-forward multi-view geometry | no | Custom | Building block for the above |

Check each license before anything goes into a product pitch: several weights
are research-only, which matters to R&S.

## Layout

```
tracks/spatial-ai/
  PLAN.md      phases, gates, deliverables
  notes.md     experiment log (what ran, settings, numbers, takeaway)
  envs/        environment files per method (cloud GPU)
  runners/     data/<seq> -> results/<seq>/<method>.txt (TUM)
  scale/       IMU-based scale + gravity alignment
  stitch/      dense-geometry patch placement (the demo)
```
