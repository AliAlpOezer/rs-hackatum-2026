# Evaluation

Scores every `results/<seq>/<method>.txt` (TUM format) against `data/<seq>/gt.txt`.

```bash
pip install evo
# absolute error; -as aligns with scale (needed for monocular methods)
evo_ape tum data/<seq>/gt.txt results/<seq>/<method>.txt -a -as --plot
# relative error over 1 s windows - the metric that matters for patch stitching
evo_rpe tum data/<seq>/gt.txt results/<seq>/<method>.txt -a --delta 1 --delta_unit s
# all methods on one plot
evo_traj tum results/<seq>/*.txt --ref data/<seq>/gt.txt -a --plot
```

Record the numbers in `results/summary.md`:

| Sequence | Method | ATE RMSE [m] | RPE 1 s [m] | Scale aligned? | Lost/reinit | Runtime |
|---|---|---|---|---|---|---|
