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

Step 2 (2026-11-28, done): `train_noise.py` learns log(noise) for the six parameters
with Adam (lr 0.15, 30 iterations per fold) on random 120-frame windows; loss = mean
squared camera-position error in mm^2 vs GT rigidly aligned (Umeyama SE3) to the stereo
world. 5 folds, two sequences held out each; held-out outputs ->
`results/seqNN/eskf_learned.txt`; params and log in `results/imu_fusion/`. First gradient check (seq10, 30 frames): pushes
`pos` up (trust stereo less) and `gyro` down. Score held-out files with `eval/evaluate.py`.

Step 2 results (held-out, ATE SE3 / RPE 1 s, mm):

| Seq | stereo | step 1 pos 1.5 | step 1 pos 4 | learned (CV) |
|---|---|---|---|---|
| seq01 | 1.9 / 2.4 | 1.9 / 2.1 | 1.9 / 2.1 | 1.9 / 2.0 |
| seq02 | 1.9 / 1.3 | 1.9 / 1.2 | 1.9 / 1.1 | 1.9 / 1.1 |
| seq03 | 1.2 / 1.4 | 1.2 / 1.4 | 1.2 / 1.4 | 1.2 / 1.4 |
| seq04 | 1.6 / 1.8 | 1.6 / 1.8 | 1.6 / 1.7 | 1.6 / 1.7 |
| seq05 | 1.3 / 1.4 | 1.2 / 1.3 | 1.2 / 1.3 | 1.2 / 1.3 |
| seq06 | 1.3 / 1.4 | 1.2 / 1.3 | 1.2 / 1.2 | 1.2 / 1.3 |
| seq07 | 1.2 / 1.6 | 1.1 / 1.5 | 1.1 / 1.5 | 1.1 / 1.6 |
| seq08 | 1.6 / 2.0 | 1.5 / 1.9 | 1.5 / 1.9 | 1.6 / 2.0 |
| seq09 | 2.0 / 1.9 | 2.0 / 1.8 | 2.0 / 1.8 | 2.0 / 1.8 |
| seq10 | 9.5 / 10.8 | 9.4 / 10.6 | 9.3 / 10.4 | 9.3 / 10.7 |

Learned per fold: pos 1.1-3.4 mm, rot 3.2-12.7 mrad, gyro 0.7-4.0e-3, accel 0.8-2.4e-2.
Conclusion: learning six constant noise values matches hand tuning, no further gain;
the fold-to-fold spread (5x on gyro) says the loss is flat in these parameters. A
constant-noise filter cannot fix seq10 (stereo-drift error, not jitter). Next lever:
step 3, per-frame adaptive trust (learned covariance), or a better visual measurement.

Step 3 (2026-11-28, done): `train_trust.py`, MLP 5->16->2 (tanh, zero-init last layer)
on per-frame features (stereo speed /0.2, stereo rot rate /0.5, gyro rate /0.5,
|stereo-gyro rate| /0.1, ||a|-g| /0.5) -> variance multipliers in [e^-2, e^2] on stereo
pos/rot; base noise = step 1 with pos 4 mm. Adam lr 0.02, 40 iters/fold, same folds/loss.
Outputs `results/seqNN/eskf_trust.txt`; multipliers and log in `results/imu_fusion/`.
