# Spatial AI plan (Alp)

Hours from kickoff (H0), matching [docs/ROADMAP.md](../../docs/ROADMAP.md).
Each phase ends in a **gate**: a decision based on numbers, written into `notes.md`.

## Goal and what "done" means

By feature freeze (H30), deliver:

1. At least **two learned methods** scored with the shared eval next to ORB-SLAM3
   (ATE, RPE over 1 s, scale error, lost-tracking count, runtime, VRAM).
2. **Metric-scale** trajectory from a learned method, with the IMU fixing scale and gravity.
3. **One demo only learned methods can do**: radar patches placed on a dense 3D
   surface (pose + depth + surface normal), not on a flat mosaic.
4. A clear answer for the pitch: *where* learned beats classical, and where not.

Non-goals: training or fine-tuning models (no time, no data), real-time on-device
inference, writing our own SLAM back end from scratch.

## Constraint: compute

My laptop has 4 GB RAM and no GPU. Everything runs on a **remote GPU**:

- Target: one 24 GB GPU (RTX 4090 / A10 / L4). DPVO fits in much less;
  MASt3R-SLAM and VGGT-SLAM can need far more on long clips.
- Sources, in order: a teammate's GPU machine, sponsor cloud credits at
  hackaTUM, then Colab Pro / RunPod / Lambda.
- Work over SSH + VS Code Remote or Jupyter; keep code in this repo, pull on the box.
- Decide this **in the first hour**. Without a GPU this track stalls.

## Phase 0 - Setup (H0 to H3)

- [ ] Get the GPU box; clone the repo; install CUDA PyTorch.
- [ ] Install **DPVO** first (lightest, MIT). Run on EuRoC MH_01 and score with
      `eval/` to prove the toolchain works end to end.
- [ ] Start installing **MASt3R-SLAM** and **VGGT-SLAM 2.0** in separate envs
      in the background (builds and model downloads are slow).
- [ ] Agree with Teammate C on `data/<seq>/` and get the first R&S sequence.
- [ ] Record **our own test clips** with a phone (video + IMU, e.g. a sensor
      logger app): slow close-range sweeps over clothing, a wall, a backpack,
      with an AprilTag/checkerboard in view for rough ground truth. This gives
      us the hard cases even if R&S data is easy.

**Gate H3:** DPVO trajectory on EuRoC scored. If not, fix the environment before
anything else.

## Phase 1 - First learned trajectories (H3 to H10)

- [ ] `runners/dpvo.py`: `data/<seq>` -> `results/<seq>/dpvo.txt` (TUM,
      camera-to-world, seconds).
- [ ] Run DPVO on all R&S sequences and our clips. Score with Sim(3) alignment.
- [ ] Get one dense method running (MASt3R-SLAM or VGGT-SLAM 2.0, whichever
      installed first). Same runner pattern.
- [ ] Compare with ORB-SLAM3 as soon as Teammate A has output. Note per
      sequence: who tracks, who fails, where.

**Gate H10:** pick the main learned method by RPE and robustness on *our*
data. Keep one sparse (DPVO) and one dense method; drop the rest.

## Phase 2 - Metric scale from the IMU (H10 to H18)

Monocular learned VO has no metric scale; the IMU has it. Two paths, try the
cheap one first:

- [ ] **Path A - own alignment (`scale/`)**, 2-3 hours. Given VO poses and IMU:
      estimate time offset (gyro vs VO rotation rate cross-correlation), then
      solve least squares for scale, gravity direction and accelerometer bias
      over a sliding window (the same idea as visual-inertial initialization).
      Output: metric TUM trajectory `<method>_imu.txt`.
- [ ] **Path B - MASt3R-Fusion**: ready-made MASt3R + IMU factor graph. Try it
      if Path A works and time allows, or if Path A fails. Treat its code as
      unverified until it runs on EuRoC.
- [ ] Cross-check with a metric depth prior (MapAnything) on a few chunks:
      do IMU scale and learned metric scale agree? Agreement is a good pitch
      slide; disagreement tells us which to trust.

**Gate H18:** metric scale error under ~5% on sequences with ground truth, or a
clear statement of why not.

## Phase 3 - The differentiator: geometry for radar patches (H18 to H26)

Classical SLAM gives a pose. Radar also needs **distance and angle to the
surface** at each glimpse. Dense learned geometry gives both.

- [ ] Per frame, take depth + normals from the dense method (or MapAnything on
      a chunk), in the metric frame from Phase 2.
- [ ] Define a "radar patch" stand-in: the image region at the device's
      boresight (center crop). Project each patch onto the reconstructed
      surface using pose + depth.
- [ ] `stitch/`: render a textured surface (point cloud or mesh) with all
      patches placed, vs. the flat 2D mosaic from Teammate C. Side by side for
      the pitch.
- [ ] With Teammate A: fallback fusion, where the learned VO takes over when
      ORB-SLAM3 loses tracking (switch on its tracking state, align the
      segments with Sim(3)). Only if both pipelines are stable.

**Gate H26:** demo renders on at least one sequence.

## Phase 4 - Results and pitch material (H26 to H30, then freeze)

- [ ] Final table in `results/summary.md` for every method x sequence.
- [ ] 3 plots: trajectories overlaid, RPE over time with failure regions
      marked, scale error.
- [ ] 2 slides: "where learned wins" (close range, low texture, fast turns,
      with numbers) and "what it costs" (VRAM, runtime, licenses).
- [ ] Demo video of the 3D patch placement.
- [ ] Clean runners so `README` reproduces every number.

## Risks

| Risk | Sign | Response |
|---|---|---|
| No GPU by H1 | Nothing runs | Shout early; use Colab with DPVO only; do the scale and stitch code on CPU with DPVO outputs |
| Dense methods run out of VRAM | OOM on long clips | Subsample frames (stride 2-5), cut into chunks, use the submap mode |
| R&S data has no ground truth | Cannot compute ATE | Our AprilTag clips; consistency metrics; visual quality of stitching |
| Close-range clips break everything | All methods lose tracking | That is the finding: show it, use IMU-only short-term propagation between good frames |
| Bad camera-IMU time sync | Scale estimate unstable | Estimate offset first (Phase 2); report it to the team |
| Weights not licensed for commercial use | R&S asks in Q&A | Have the license column ready; MIT/Apache/BSD options (DPVO, MapAnything) as the product path |

## Sync points with the team

- H2: data layout frozen (C). H10: first comparison with ORB-SLAM3 (A).
- H18: scale-corrected trajectories to C for stitching. H26: demo to D for the pitch.
