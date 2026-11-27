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

## Open questions for the R&S mentors (ask in the first hours)

- [ ] What data do we get? Video resolution and fps, IMU rate, timestamps,
      synchronized or not, any radar data or only camera + IMU?
- [ ] Is camera-IMU calibration (intrinsics, extrinsics, time offset) provided?
- [ ] Is there ground truth (mocap, robot arm, AprilTag board)? If not, how is
      Task 1 judged?
- [ ] Typical motion: slow sweeps close to a body or object? Distance to the
      surface? Texture (clothing, uniform walls)?
- [ ] Offline or real-time? Target hardware (phone-class, laptop, GPU)?
- [ ] Judging weight between Task 1 and Task 2, or pick one?
- [ ] Licensing constraints (ORB-SLAM3 is GPLv3)?
