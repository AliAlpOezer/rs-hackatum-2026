# IMU fusion log

Step 1 (2026-11-28): `eskf.py`, ESKF + RTS on Jonas's ORB-SLAM3 stereo (origin/orb-slam-3
270af5a), hand-set noise, per-sequence offset from gyro vs stereo rotation rate.
ATE SE3 / RPE 1 s in mm, `eval/evaluate.py --metric` vs R&S `trajectory.txt`:

| Seq | stereo | ESKF pos 1.5 mm | ESKF pos 4 mm |
|---|---|---|---|
| seq01 | 1.9 / 2.4 | 1.9 / 2.1 | 1.9 / 2.1 |
| seq02 | 1.9 / 1.3 | 1.9 / 1.2 | 1.9 / 1.1 |
| seq03 | 1.2 / 1.4 | 1.2 / 1.4 | 1.2 / 1.4 |
| seq04 | 1.6 / 1.8 | 1.6 / 1.8 | 1.6 / 1.7 |
| seq05 | 1.3 / 1.4 | 1.2 / 1.3 | 1.2 / 1.3 |
| seq06 | 1.3 / 1.4 | 1.2 / 1.3 | 1.2 / 1.2 |
| seq07 | 1.2 / 1.6 | 1.1 / 1.5 | 1.1 / 1.5 |
| seq08 | 1.6 / 2.0 | 1.5 / 1.9 | 1.5 / 1.9 |
| seq09 | 2.0 / 1.9 | 2.0 / 1.8 | 2.0 / 1.8 |
| seq10 | 9.5 / 10.8 | 9.4 / 10.6 | 9.3 / 10.4 |

Never worse; gains are in RPE (jitter), ATE barely moves. |g| estimate 9.80-9.81.
Runtime ~1 min per sequence on the laptop CPU (Python loop over ~4000 IMU steps).
Next: step 2, learn the six noise parameters by gradient descent, leave-one-sequence-out.
