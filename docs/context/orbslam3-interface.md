---
topic: orbslam3-interface
updated: 2026-11-28
status: proposed, waiting on Jonas's fork details
---

# Seeing into and steering Jonas's ORB-SLAM3

## Goal

Make Jonas's ORB-SLAM3 fork observable and steerable from Python, so ML components
can read its internal state per frame and later feed signals back, without
duplicating or diverging from his SLAM work.

Technology-independent version: *we can watch every frame of his estimator, record
it, and test an ML change against his exact baseline on the same data.*

## Jonas's fork (checked 2026-11-28 via GitHub API)

- Our branch `origin/orb-slam-3` adds it as submodule `ORB_SLAM3` ->
  `git@github.com:JunkyDeveloper/ORB_SLAM3.git`, branch `ras-changes`, commit
  `eaf520a` ("changes for not fully correct dataset"). Not merged into `main`.
- 2 commits ahead of `UZ-SLAMLab/ORB_SLAM3` master, 0 behind. Changes:
  - build fixes: C++14, `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`, `mkdir -p`, Sophus
    without `-Werror`;
  - Pangolin viewer switched off in `stereo_euroc` and `mono_inertial_euroc` (runs headless);
  - new yaml key `IMU.fastInit` (Settings -> `Tracking::mFastInit`) that skips the
    acceleration check at stereo/RGB-D inertial initialization. That suggests he runs
    **stereo-inertial** and the R&S clips did not pass upstream's init check.
- Not in the fork or our repo yet: the R&S settings yaml, the run command, the
  trajectory output, or anything showing the 2 mm.

What this means for the design: his core is upstream plus a settings flag, so a
tap layer on top is cheap and unlikely to conflict. Since he already runs headless,
Rerun is the only viewer we need.

## Real seq01 baseline (Jonas, `origin/orb-slam-3` da3921e, scored 2026-11-28)

Scored with `eval/evaluate.py --metric` against his `results/seq01/gt.txt`:

| Mode | Poses | ATE Sim3 | RPE 1 s | Scale | ATE SE3 |
|---|---|---|---|---|---|
| stereo | 601/601 | 1.6 mm | 2.1 mm | 0.995 | 1.8 mm |
| stereo_inertial | 27 | 0.6 mm | n/a | 1.000 | 0.6 mm |
| mono_inertial | 15 | 8.2 mm | n/a | 0.32 | 36.5 mm |

- The 2 mm is real and comes from **pure stereo**. An earlier note here called it
  synthetic; that was wrong (the synthetic runs in `tracks/orbslam3/notes.md` are separate).
- Inertial modes keep under 1 s of trajectory: only the last map is saved after resets.
  Jonas is on it (fork `ras-changes` b635bdc "again still wrong dataset", +105 lines in
  `System.cc`); do not duplicate.
- 1.8 mm over a 2.6 m handheld path is suspiciously close. Strengthens the hypothesis
  that `trajectory.txt` is ZED SDK visual(-inertial) tracking from the same images, so
  the target is agreement with ZED's tracker, not truth.
- `eval/evaluate.py` crashed on trajectories shorter than the RPE window; it now reports
  `nan` (evo `FilterException` caught).
- Real motion (GT): seq01 extent 0.42 x 0.27 x 0.24 m, path 2.6 m; seq10 path 4.2 m,
  up to 20 mm per frame.

## Handoff (2026-11-28)

- Goal reframed by Alp: build a deep-learning or hybrid pose method that beats the
  ORB-SLAM3 baseline on real R&S data; enhancing Jonas's fork is secondary.
- Raw dataset (seq01-20, seq11-20 without GT) is on Alp's Google Drive, locally
  `G:\My Drive\dataset_extracted\` (laptop mount, read-only use).
- seq01 baseline exists (above); the Colab notebook
  `tracks/orbslam3/notebooks/orbslam3_baseline_colab.ipynb` (not yet run) is kept for
  other sequences.
- DROID-SLAM pipeline pushed on branch `spatial-ai/droid-slam` (7d81bbd): runner,
  `envs/install_droid.sh`, `notebooks/droid_rs_colab.ipynb`; syntax/lint checked only,
  never run on a GPU. Run it through the official Colab CLI in WSL, signed in as
  alialp@live.com (2026-11-28); seq01 can go up with `colab upload` from
  `G:\My Drive\dataset_extracted` if that account's Drive lacks the data. After it runs: compare droid_stereo vs orbslam3_stereo 1.8 mm. Next DROID-SLAM
  stereo as the first learned candidate.
- This doc was written on a local `main` behind `origin/main`; `INDEX.md` will
  conflict on pull (both sides appended a line; keep both).

## Where ImageNet-pretrained ResNets fit (assessed 2026-11-28)

Absolute pose regression from a ResNet (PoseNet style) is not a candidate: such
models behave like image retrieval and stay at cm-to-dm error (Sattler et al.,
CVPR 2019). Millimetre trajectories need geometry; ResNets fit as parts:

| Use | Backbone | Data need | Fit |
|---|---|---|---|
| Keypoint reliability (option B), BCE on patches | ResNet-18 ImageNet, fine-tuned | self-labelled by ORB-SLAM3 inliers | good |
| Frame quality / failure predictor (option C) | ResNet-18 features + SLAM state | GT seq01-10 | good |
| Masks for hands, people, specular areas before tracking | DeepLabv3 / Mask R-CNN ResNet-50 (COCO, ImageNet init) | none, pretrained | good, cheap |
| Monocular depth prior | Monodepth2-style ResNet encoder | weak; stereo already gives depth | low |
| Relative pose correction head on top of VIO | ResNet on frame pairs + IMU | 6k correlated frames, overfit risk | experimental |
| Place recognition / loop closure | NetVLAD (VGG16 ImageNet) | none | low: 20 s clips rarely loop |

## Claim to verify first

Jonas reports about 2 mm precision on one R&S sequence. Before building on it, pin down:

- which sequence, which mode (stereo-inertial or mono-inertial), which metric (ATE or
  RPE, SE(3) or Sim(3) alignment), and whether the config was tuned on that same
  sequence;
- where R&S `trajectory.txt` comes from. If it is ZED SDK positional tracking (itself
  stereo-inertial odometry), 2 mm means agreement with another VIO, not with truth;
- the number reproduced by `eval/evaluate.py`, not by his own script.

2 mm on a 20 s clip leaves little precision to gain. The ML case is more likely
**robustness and generalization** (other sequences, fast motion, low texture, lost
tracking) than shaving that one number.

## Invariants

1. **His fork stays the source of truth.** Our changes are an additive hook layer,
   offered back as a PR or branch on his fork, never a private copy. GPLv3: built
   outside this repo (`AGENTS.md`).
2. **Observation never changes behaviour.** With all ML switched off, the
   instrumented build reproduces his trajectory (checked by eval, same tolerance as
   a rerun of his own binary; ORB-SLAM3 is multithreaded, so not bit-for-bit).
3. **Every ML change is scored against that baseline** through the shared TUM
   contract, on sequences it was not tuned on.
4. **Every run leaves a record on disk** (TUM trajectory plus per-frame state), so
   ML work can run offline without the SLAM process.

## Components

| Component | Owner | Reason to change | Correctness check |
|---|---|---|---|
| SLAM core (Jonas's fork, C++) | Jonas | tracking and tuning | his ATE on the agreed sequence |
| Tap layer (C++, in fork) | Alp | what state we expose | read-only; baseline trajectory unchanged |
| Python binding (pybind11) | Alp | API for ML | `track()` loop reproduces his binary's trajectory |
| Recorder | Alp | dataset format for ML | files load, TUM scores in `eval/` |
| Viewer (Rerun) | Alp | what we want to look at | shows frames, keypoints, trajectory, map |
| ML fusion (later) | Alp | experiments | beats baseline on held-out sequences |

The tap fills one struct per frame after `System::TrackStereo` / `TrackMonocular`:

```
FrameState: t, T_cw, tracking_state, is_keyframe, n_matches, n_inliers,
            keypoints (u, v, map_point_id), map points seen (id, xyz),
            imu velocity and biases (inertial modes)
```

The binding exposes about six calls we own, rather than a full third-party wrapper:
`init(settings, vocab)`, `track(left, right, imu_batch, t) -> FrameState`,
`map_snapshot()`, `reset()`, `shutdown()`, and later the injection hooks.

## Seams

| Seam | Carrier | Contract | Failure |
|---|---|---|---|
| core -> tap | in-process C++ | `FrameState`, owned by the tap | tap only reads; core state is never written |
| tap -> Python | pybind11 call, synchronous | `FrameState` as numpy arrays | lost tracking is a *state*, not an exception; crashes raise |
| Python -> recorder | files per run id | `results/<seq>/<method>.txt` (TUM) + `runs/<id>/frames.parquet` | append-only; a partial run is marked incomplete |
| Python -> viewer | Rerun stream or `.rrd` file | Rerun entity paths | viewer down never blocks tracking |

Dependency direction: ML depends on the binding, the binding depends on the tap, the
tap depends on his core. His core depends on nothing of ours.

## Ways to interact, least invasive first

1. **Input side** (no core change): alter what we feed it - masks for moving or
   textureless regions, image enhancement, IMU preprocessing.
2. **Output side** (no core change): fuse `FrameState` with a learned VO (DPVO)
   in a separate estimator; take over when tracking is lost; learned uncertainty.
3. **Inside** (core hooks): pose priors, relocalization seeds, learned features.
   Only once 1 and 2 show a gap that needs it.

## ML options for generalization (proposed 2026-11-28)

Scoring is trajectory error on hidden seq11-20, so generalization *is* the score.
Data: seq01-10, about 600 frames each, with metric ground truth. Too little for
learning anything with many parameters from trajectories alone, so every option
below either has few parameters or gets its labels from ORB-SLAM3 itself.
ORB-SLAM3 is not differentiable (RANSAC, thresholds, keyframe decisions), so
settings cannot be fine-tuned by gradient descent.

| # | Idea | Loss / optimizer | Labels | Core change | Order |
|---|---|---|---|---|---|
| A | Learn the yaml (IMU noise, FAST thresholds, nFeatures, ThDepth; plus camera-IMU time offset, applied by shifting IMU timestamps since the yaml has no key for it) instead of hand-tuning | Cross-Entropy Method or CMA-ES on mean RPE, leave-one-sequence-out CV | GT seq01-10 | none, batch runs | first |
| B | Feature reliability: predict which ORB keypoints become tracking inliers; drop or reweight the rest | binary cross-entropy | ORB-SLAM3's own inlier flags via the tap (thousands per frame) | hook between extractor and tracking | second |
| C | Failure predictor: will this frame's pose be bad? Switch to DPVO/IMU propagation | binary cross-entropy | GT local RPE above a threshold | none (output side) | third |
| D | Swap ORB for SuperPoint + LightGlue, fine-tune on R&S | SuperPoint cell softmax CE, LightGlue assignment NLL | homographic / pose supervision | large: new descriptors, vocabulary, matcher | not in hackathon |

A must keep a held-out fold: a config tuned on all ten sequences repeats the
hand-tuning overfit with more evaluations.

## Compute

Alp's laptop (4 GB RAM, no GPU) cannot comfortably host the build plus the
vocabulary. Run it on Jonas's machine or a Linux box/Colab (CPU is enough for
ORB-SLAM3). View remotely through Rerun's web viewer or a saved `.rrd` file. That
replaces the Pangolin window.

## Rejected alternatives

| Option | Why not |
|---|---|
| ROS wrapper | ROS1 is end-of-life, heavy install, nothing else in the stack uses ROS |
| Separate process + socket (ZeroMQ) | a second process, serialization and failure mode; fallback if pybind11 fights his build |
| Parsing stdout or the final trajectory only | loses per-frame state; no way to steer |
| Pangolin GUI only | not scriptable, unusable remotely on this laptop |
| Third-party Python bindings | track upstream ORB-SLAM3, not his fork's changes |

## Plan

0. **Access**: fork URL and commit, exact run command, config yaml, sequence, how 2 mm was measured, which machine he runs on.
1. **Reproduce** his run on a machine we control; score with `eval/`. Gate: we get his number.
2. **Tap + binding**: Python loop over `data/<seq>` gives the same trajectory as his binary. Gate: baseline ATE unchanged.
3. **Recorder + Rerun viewer**: watch a run frame by frame.
4. **Run all sequences**: list where it degrades or loses tracking. That is the ML target list.
5. **First fusion experiment** picked from step 4 (likely output-side DPVO fallback).

## Open questions for Jonas

- Answered: repo, branch, commit and diff (see "Jonas's fork").
- Push the R&S settings yaml and the exact run command (stereo-inertial with `IMU.fastInit: 1`?).
- Sequence and exact metric behind "2 mm".
- Is his machine reachable for us (SSH/Tailscale), or do we build our own copy?

## IMU focus (Alp, 2026-11-28; DROID-SLAM dropped)

Jonas `origin/orb-slam-3` 270af5a: stereo (no IMU) on seq01-10, scored by us with
`eval/evaluate.py` against `trajectory.txt`: ATE SE3 1.2-2.0 mm on seq01-09,
**9.5 mm on seq10** (fastest motion); Sim3 scale 0.988-1.006.

IMU facts measured from `dataset_extracted` (seq01/05/10, script inline in the session):
- ~199 Hz, irregular (dt p1-p99 2.4-8.2 ms); |a| mean 9.805-9.817 (accel scale fine).
- **No static start**: gyro std 0.015-0.066 rad/s in the first 0.3 s, so noise and
  biases cannot come from a still segment.
- **Gentle excitation**: max ||a|-g| 0.40-1.14 m/s^2, max |w| 0.25-1.22 rad/s. Explains
  ORB-SLAM3's "not enough acceleration" init gate and weak IMU scale observability.
- **Time offset**: gyro-norm vs GT angular-speed cross-correlation peaks at -16 ms on
  seq10 (corr 0.990), -1/-2 ms on seq05/01. Sign: IMU stamps appear ~16 ms behind;
  verify by shifting and re-scoring. ORB-SLAM3 has no td key: shift in the converter.

Proposed IMU work, in order: (1) per-sequence time offset estimated against the stereo
trajectory (works on test seqs), (2) gyro bias init from gyro vs stereo rotation rate,
fastInit, (3) noise/random-walk from residuals, then CEM/CMA-ES leave-one-sequence-out
on noise scales, (4) learned gyro/accel correction network (1D CNN over IMU windows,
Brossard-style denoising) writing a corrected imu.csv, no core change, (5) per-sequence
fallback to stereo-only by a consistency check.

### Time-offset check (2026-11-28, scripts `../jonas_res/offsets.py`, `shift.py`, outside the repo)

Rotation-rate cross-correlation, 0.5 ms steps, seq01-10:
- **gyro vs stereo (camera-IMU offset)**: -0.5 to +3 ms on seq01/02/04-07, but
  **-31 ms seq03, -11.5 seq08, -10 seq09, -12 seq10** (corr 0.97-0.99). The offset varies
  per recording, so estimate it per sequence against the stereo trajectory (test seqs have no GT).
- **stereo vs GT**: a constant -6.5 to -8.5 ms on all ten. Shifting the stereo output by
  +-7.5 / -15 ms does *not* reduce ATE (seq01 1.9 -> 2.1-2.9 mm; seq10 9.5 -> 9.1-10.0 mm).
  So it is an artifact of rotation-rate comparison, not a usable output timing fix.
  Do not shift outputs.
- Still to do: rerun stereo-inertial with IMU timestamps shifted by the per-sequence
  gyro-vs-stereo offset (seq03, seq08-10 first) and compare with the unshifted run.
  Needs the ORB-SLAM3 build (Jonas, or the Colab CLI in WSL).

## Own IMU fusion instead of ORB-SLAM3 inertial (Alp, 2026-11-28)

Decision: no more ORB-SLAM3 work on our side. Treat Jonas's stereo trajectory as a
frozen visual measurement and build our own IMU fusion on top (loosely coupled).
Options assessed (from knowledge, not yet prototyped):
- ESKF (error-state EKF): IMU propagation 200 Hz, stereo pose updates 30 Hz; states
  p, v, R, gyro/accel bias, time offset td. Plus an RTS smoother: submission is offline,
  so the smoother can use future data.
- Batch factor graph (GTSAM, IMU preintegration + VO between-factors): most accurate
  offline; heavier dependency.
- Invariant EKF / UKF: better consistency, marginal gain at 20 s.
- ML: (a) differentiable ESKF in PyTorch, noise/offset parameters learned by gradient
  descent on GT error, leave-one-sequence-out; (b) learned per-frame covariance (small
  MLP on VO quality + IMU stats), AI-IMU style; (c) gyro denoising CNN (Brossard 2020);
  (d) learned inertial odometry (TLIO/RoNIN) rejected: built for pedestrians over long
  windows, not mm-level 20 s handheld.
Recommended: one PyTorch ESKF + RTS smoother (classical baseline), then make its
parameters learnable, then add learned covariance. Target: beat stereo-only
1.2-2.0 mm (seq01-09) and 9.5 mm (seq10).

Step 1 built: `tracks/imu_fusion/eskf.py` (ESKF + RTS, torch float64), results in
`tracks/imu_fusion/notes.md`: never worse than stereo, RPE 1 s down ~0.1-0.3 mm, ATE ~flat.
Next: step 2, learn the noise parameters by gradient descent, leave-one-sequence-out.

Step 2 done: learned constant noise = hand tuning (see `tracks/imu_fusion/notes.md`); next is step 3, adaptive per-frame trust.
