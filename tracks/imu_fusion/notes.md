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

Step 3 results: `python tracks/imu_fusion/score.py "<raw R&S root>" results`, all four
stages on the same timestamps (ATE SE3 mm / RPE 1 s mm / rotation RMS deg; steps 2-3
held out by 5-fold CV, step 1 = hand-set noise with pos 4 mm):

| Seq | ORB-SLAM3 stereo | step 1 ESKF | step 2 learned | step 3 trust |
|---|---|---|---|---|
| seq01 | 1.91 / 1.93 / 0.289 | 1.87 / 1.82 / 0.285 | 1.86 / 1.83 / 0.293 | 1.86 / 1.82 / 0.283 |
| seq02 | 1.92 / 1.72 / 0.173 | 1.88 / 1.63 / 0.174 | 1.89 / 1.64 / 0.208 | 1.89 / 1.63 / 0.181 |
| seq03 | 1.21 / 1.27 / 0.481 | 1.17 / 1.20 / 0.481 | 1.17 / 1.20 / 0.478 | 1.17 / 1.20 / 0.483 |
| seq04 | 1.58 / 1.65 / 0.409 | 1.57 / 1.57 / 0.420 | 1.56 / 1.57 / 0.492 | 1.55 / 1.57 / 0.471 |
| seq05 | 1.26 / 1.30 / 0.121 | 1.21 / 1.22 / 0.124 | 1.22 / 1.23 / 0.123 | 1.22 / 1.23 / 0.123 |
| seq06 | 1.25 / 1.47 / 0.073 | 1.19 / 1.37 / 0.070 | 1.19 / 1.35 / 0.068 | 1.19 / 1.38 / 0.068 |
| seq07 | 1.17 / 1.37 / 0.112 | 1.07 / 1.21 / 0.108 | 1.10 / 1.27 / 0.115 | 1.08 / 1.22 / 0.109 |
| seq08 | 1.65 / 1.80 / 0.285 | 1.54 / 1.57 / 0.282 | 1.56 / 1.61 / 0.282 | 1.54 / 1.58 / 0.282 |
| seq09 | 2.04 / 1.94 / 0.217 | 1.98 / 1.77 / 0.224 | 1.97 / 1.79 / 0.222 | 1.98 / 1.79 / 0.266 |
| seq10 | 9.47 / 9.06 / 0.763 | 9.28 / 8.60 / 0.743 | 9.33 / 8.74 / 0.749 | 9.35 / 8.79 / 0.710 |
| mean | 2.35 / 2.35 / 0.292 | 2.28 / 2.20 / 0.291 | 2.28 / 2.22 / 0.303 | 2.28 / 2.22 / 0.297 |

Learned multipliers (`trust_log.json`) split by fold: seq01-04 and seq09-10 learn pos
x0.2-1.0, rot x4.5-7.3 (trust stereo rotation less); seq05-08 learn pos x1.6-2.0, rot
x0.7-1.3. Opposite directions on different folds = no transferable signal in these features.

Conclusion: every fusion stage beats raw stereo (ATE -0.07 mm, RPE -0.13 mm on average),
but the gain is all in step 1. Learning constant noise (step 2) or per-frame trust
(step 3) does not beat the hand-set filter; step 3 only recovers part of the rotation
step 2 lost (0.303 -> 0.297 deg vs 0.291 for step 1). Caveat: step 1's pos 4 mm was
picked from two settings after seeing all sequences, so it has a small selection edge.
seq10 stays at ~9.3 mm: the error is stereo drift, which reweighting cannot remove.
Next lever is a better visual measurement, not more filter tuning.

Reproducibility: `results/seqNN/eskf.txt` was regenerated with the final `eskf.py
--pos 4e-3` (the original step 1 run kept only seq01/seq10); seq01, 05, 10 match the
step 1 table above. Earlier `eskf.py` revisions differ only by tensor noise inputs (step 2)
and the optional `rscale` hook (step 3), so the default path is unchanged.

evo comparison (2026-11-28): `uv run python tracks/imu_fusion/compare_evo.py <raw_root> results [--all]`
compares ORB-SLAM3 stereo, `results/seqNN/orbslam3-imu-fusion.txt` (step 1 ESKF) and ORB-SLAM3's own
stereo-inertial mode; table + per-sequence plots in `results/imu_fusion/compare/`. evo RPE is in the
local frame (relative pose), so it differs from `score.py`'s world-frame RPE.
Stereo was rerun on main (`results_euroc_color/stereo/seqNN/f_seqNN.txt`, differs from the 270af5a
run), so `orbslam3-imu-fusion.txt` was regenerated from the new stereo. Means (ATE / RPE 1 s mm /
RPE rot deg): stereo 2.17 / 2.40 / 0.229, fusion 2.11 / 2.22 / 0.220, stereo-inertial 2.35 / 2.32 /
0.229 (seq04 lost tracking, 218/602 poses). Fusion beats stereo on ATE in all 10 sequences and
beats stereo-inertial on ATE in 7 of 10; stereo-inertial wins seq10 RPE (7.81 vs 8.26 mm).

ESKF on stereo-inertial (2026-11-28): same filter, input = ORB-SLAM3 stereo-inertial
(`results/seqNN/orbslam3-stereo-inertial-imu-fusion.txt`, `si_fusion` in compare_evo.py).
Means: stereo-inertial 2.35 / 2.32 / 0.229 -> si_fusion 2.29 / 2.22 / 0.224; ATE better on all 10
sequences, best seq10 RPE overall (7.38 mm). Still behind fusion on plain stereo in ATE (2.11).
Bug fixed on the way: gravity was averaged over all IMU samples, also those outside the camera
time range; seq04 stereo-inertial (218/602 poses) gave |g| 9.16 and ATE 4.35 mm. Now only IMU
inside the trajectory's time range (|g| 9.81, ATE 1.26 mm); full-coverage outputs move < 0.004 mm.
