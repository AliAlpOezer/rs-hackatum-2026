# Challenge brief (paraphrased)

Source: R&S 5-minute pitch at hackaTUM 2026, shared on the hackaTUM Discord.
This is our summary; the deck itself is not committed.

## Context

Airport body scanners use radar imaging at a fixed gate. R&S is moving screening
off the gate and into the hand: one mobile device carrying **radar and a camera**.
A handheld sees only a small patch at a time, so the device must know where it
was for every patch to build one coherent image.

## Task 1 - Where is my handheld?

> One patch. Then a pose.

Use **video and IMU** to estimate the device trajectory. Each radar glimpse is
then placed in a common frame using that pose. Accuracy matters most in the
place the radar cares about: relative pose between nearby glimpses (drift over
seconds), and metric scale.

## Task 2 - What else?

Airport screening is taken. Propose a new application for the same device,
with a reason people would actually carry it: materials testing, industry,
medicine, infrastructure, rescue, or something unnamed.

## Detailed brief (2026-11-28)

From the detailed challenge description (PDF on the hackaTUM Discord; it holds
download credentials, so it is not committed):

- **Teams choose ONE task.** Task 1 is scored by measured trajectory error;
  Task 2 by the jury (technical use case 15, business case 15, sensor config 10, pitch 10).
- **Device:** ZED X Mini stereo camera with built-in IMU on the radar handheld.
- **Data:** 20 hand-moved recordings, about 20 s each. seq01-10 include
  `trajectory.txt` (ground truth, 30 Hz); **seq11-20 are the test set**.
  Per recording: `left.mp4`, `right.mp4` (rectified, 960x600, 30 fps),
  `imu.csv` (ms, m/s^2, deg/s, about 200 Hz), `timestamps.csv` (one per frame),
  `calibration.json` (intrinsics, 5 cm baseline, IMU -> left camera transform).
- **Deliverable:** `trajectory.txt` for seq11-20, one line per frame, same order
  and timestamps as `timestamps.csv`: `timestamp_ms tx ty tz qx qy qz qw`
  (metres, quaternion x y z w). Any method allowed.
- Ground truth starts at the origin; it is metric, so scale matters.

## Open questions for the R&S mentors (ask in the first hours)

- [ ] Exact error metric: ATE with or without alignment? RPE? Per-sequence average?
- [ ] Which frame is the ground truth in (left camera? gravity-aligned world?)
- [x] What data do we get? Video resolution and fps, IMU rate, timestamps,
      synchronized or not, any radar data or only camera + IMU?
- [ ] Is camera-IMU calibration (intrinsics, extrinsics, time offset) provided?
- [ ] Is there ground truth (mocap, robot arm, AprilTag board)? If not, how is
      Task 1 judged?
- [ ] Typical motion: slow sweeps close to a body or object? Distance to the
      surface? Texture (clothing, uniform walls)?
- [ ] Offline or real-time? Target hardware (phone-class, laptop, GPU)?
- [ ] Judging weight between Task 1 and Task 2, or pick one?
- [ ] Licensing constraints (ORB-SLAM3 is GPLv3)?
