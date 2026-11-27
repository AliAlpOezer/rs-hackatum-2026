# Roadmap

Hackathon time, counted in hours from kickoff (H0). Adjust the clock to the real
submission deadline once known; keep the order.

Principle: **get one trajectory end to end first, then make it good.** A weak
pipeline that produces a scored trajectory and a stitched picture beats two
strong half-pipelines.

## Phase 0 - Align (H0 to H2)

| Owner | Deliverable |
|---|---|
| All | Ask mentors the questions in [challenge.md](challenge.md); decide Task 1 only, or Task 1 + Task 2 |
| C | Data in `data/<seq>/` layout, one sequence checked (frames + IMU + timestamps plotted) |
| A | ORB-SLAM3 builds (Linux/WSL2 or Docker) and runs on EuRoC MH_01 |
| Alp | GPU environment ready; shortlist of 3 spatial-AI methods to try (see track README) |
| D | 10 candidate Task 2 domains listed, 3 picked for mentor feedback |

**Checkpoint H2:** everyone has their environment; data layout frozen.

## Phase 1 - First trajectory (H2 to H10)

| Owner | Deliverable |
|---|---|
| A | ORB-SLAM3 **mono-inertial** on an R&S sequence, TUM output in `results/` |
| Alp | One feed-forward/learned method (e.g. DPVO or MASt3R-SLAM) on the same sequence, TUM output |
| C | `eval/` scores both; calibration sanity checks (time offset, IMU noise from static part) |
| D | Task 2 one-pager per domain; mentor feedback round |

**Checkpoint H10:** a results table with at least two methods on one sequence.
If ORB-SLAM3 cannot initialize (low texture, short clips), that is a finding and
the reason the spatial-AI track exists.

## Phase 2 - Make it good (H10 to H22)

| Owner | Deliverable |
|---|---|
| A | Tune: IMU noise params, init, feature count; fallback to mono / stereo-free modes |
| Alp | Scale recovery and robustness: metric depth prior or IMU fusion with the learned VO; try a second method |
| C | **Patch stitching**: project frames (stand-in for radar patches) into one mosaic using each trajectory - the visible proof of Task 1 |
| D | Pick the Task 2 story; mock the user flow; gather numbers (market, who pays) |

**Checkpoint H22:** best method chosen per sequence type, stitched mosaic exists.

## Phase 3 - Fuse and demo (H22 to H30)

| Owner | Deliverable |
|---|---|
| A + Alp | Optional fusion: learned VO as fallback when ORB-SLAM3 loses tracking, or learned depth for scale |
| C | Final result table, failure cases, plots for slides |
| D | Demo script and pitch deck skeleton |

**Checkpoint H30:** feature freeze. Only fixes after this.

## Phase 4 - Pitch (H30 to deadline)

- Demo video recorded (live demos fail; have the video as backup).
- Pitch: problem (patch without pose is useless) -> our pipeline -> numbers
  (RPE, failure rate, runtime) -> stitched result -> Task 2 vision.
- Repo cleaned: README with how to reproduce, results summary.

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| ORB-SLAM3 build hell on Windows | WSL2 or Docker image from day one; do not fight native Windows |
| No ground truth | Record our own sequence with an AprilTag board; consistency metrics |
| Close-range, low-texture motion breaks features | Learned dense methods; IMU-only short-term propagation |
| No GPU | Smaller models (DPVO), cloud GPU credits from sponsors, offline batch |
| Bad time sync camera-IMU | Estimate offset (Kalibr or cross-correlation of gyro vs visual rotation) |
