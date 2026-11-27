# Evaluation

Scores every `results/<seq>/<method>.txt` (TUM format) against `data/<seq>/gt.txt`.

```bash
pip install evo
python eval/evaluate.py data/<seq>/gt.txt results/<seq>/*.txt            # monocular methods
python eval/evaluate.py data/<seq>/gt.txt results/<seq>/*.txt --metric   # + SE(3) ATE for metric-scale methods
```

It prints a markdown row per method:

- **ATE Sim3**: translation RMSE after aligning rotation, translation and scale.
- **RPE 1 s**: translation RMSE over 1 s windows (after Sim3), the number that
  matters for stitching nearby radar patches. evo only supports frame, metre and
  angle windows, so the script converts seconds to frames at the median rate.
- **Scale**: the Sim3 factor. Far from 1.0 means the method's own scale is wrong.
- **ATE SE3** (`--metric`): no scale correction; only meaningful for methods
  that claim metric scale (IMU-aided, metric depth).

For plots: `evo_traj tum results/<seq>/*.txt --ref data/<seq>/gt.txt -as --plot`.

Record the numbers in `results/summary.md`:

| Sequence | Method | ATE Sim3 [m] | RPE 1 s [m] | Scale | Lost/reinit | Runtime |
|---|---|---|---|---|---|---|
