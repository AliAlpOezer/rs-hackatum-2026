# Track: Spatial AI (Alp)

Goal: find out whether learned methods give better handheld trajectories than
classical VI-SLAM for this device, especially at close range, low texture and
fast motion, and deliver at least one of them through the shared output contract.

Candidate list as of 2026-11 from memory; verify repos, licenses and checkpoints
before committing to one.

## Options

| Family | Methods | Strength | Weakness for us |
|---|---|---|---|
| Learned VO / SLAM | **DPVO**, DROID-SLAM | Robust tracking, light (DPVO), proven on TUM/EuRoC | Monocular: no metric scale; DROID needs a big GPU |
| 3D foundation-model SLAM | **MASt3R-SLAM**, VGGT (+ VGGT-Long), CUT3R, pi3 | Dense geometry, works with few features, no calibration needed | Heavy; scale from priors, not physics; mostly offline |
| Feed-forward reconstruction | DUSt3R / MASt3R, MapAnything | Poses + dense points for a short clip in one pass | Short sequences only; memory grows with frames |
| Metric depth priors | Depth Anything (metric), UniDepth, Metric3D | Gives scale to monocular VO | Absolute scale error at close range |
| Learned inertial odometry | TLIO, RoNIN, AirIO | IMU-only motion when vision fails | Trained on walking/drones, not hand sweeps |
| Hybrid | Learned VO + IMU preintegration (e.g. DPVO poses into a GTSAM factor graph) | Metric scale and gravity from IMU, robustness from learning | Most engineering work |

## Plan

1. **H0-H2:** environments for DPVO and MASt3R-SLAM; run each on one EuRoC
   sequence to confirm they work.
2. **H2-H10:** run both on the first R&S sequence, export TUM trajectories to
   `results/<seq>/dpvo.txt`, `results/<seq>/mast3r_slam.txt`.
3. **H10-H22:** fix scale: either align with IMU (estimate scale + gravity from
   preintegrated accelerometer) or a metric depth prior. Compare to ORB-SLAM3.
4. **H22-H30:** hybrid if time allows: learned VO as fallback when ORB-SLAM3
   loses tracking; or dense points from MASt3R for the stitching demo.

## Questions to answer for the pitch

- Where does each method win (texture, distance, speed), with numbers?
- Can it run on device-class hardware, or is it an offline refinement step?
- Does dense geometry help radar-patch placement beyond pose (surface normal,
  distance to target)?

## Layout

```
tracks/spatial-ai/
  envs/        conda / pip environment files per method
  runners/     one script per method: data/<seq> -> results/<seq>/<method>.txt
  notes.md     experiment log: what ran, settings, result, takeaway
```
