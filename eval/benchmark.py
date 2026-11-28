"""Benchmark every method on every sequence and write one markdown table.

    python eval/benchmark.py <results_dir> --gt-root <data_root> --frames-root <euroc_root> [-o summary.md]

Scores results/<seq>/<method>.txt (TUM, T_world_cam, s) against <data_root>/<seq>/gt.txt
with the metrics of eval/evaluate.py: ATE after SE(3) alignment (metric methods),
ATE after Sim(3), RPE over 1 s, Sim(3) scale. Coverage = poses / frames in
<euroc_root>/<seq>/timestamps.txt, the number that matters for sequences without
ground truth (seq11-20). A per-method summary gives medians over the sequences with GT.
"""

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
from evo.core import filters, geometry, sync
from evo.tools import file_interface

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ate, rpe


def score(gt: Path, est: Path, window_s: float, max_diff: float) -> dict:
    ref = file_interface.read_tum_trajectory_file(str(gt))
    traj = file_interface.read_tum_trajectory_file(str(est))
    ref, traj = sync.associate_trajectories(ref, traj, max_diff=max_diff)
    out = {}
    for name, scale in (("ate_se3", False), ("ate_sim3", True)):
        t = copy.deepcopy(traj)
        _, _, s = t.align(ref, correct_scale=scale)
        out[name] = ate(ref, t)
        if scale:
            out["scale"] = s
            try:
                out["rpe"] = rpe(ref, t, window_s)
            except filters.FilterException:  # trajectory shorter than the window
                out["rpe"] = math.nan
    return out


def fmt(v, unit=1.0, digits=1):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * unit:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--gt-root", type=Path, required=True, help="folder of <seq>/gt.txt")
    parser.add_argument("--frames-root", type=Path, required=True, help="folder of <seq>/timestamps.txt")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--max-diff", type=float, default=0.01)
    parser.add_argument("-o", "--out", type=Path, help="write the markdown here too")
    args = parser.parse_args()

    rows = []
    for seq_dir in sorted(p for p in args.results.iterdir() if p.is_dir()):
        seq = seq_dir.name
        ts = args.frames_root / seq / "timestamps.txt"
        n_frames = len(ts.read_text().split()) if ts.exists() else None
        gt = args.gt_root / seq / "gt.txt"
        for est in sorted(seq_dir.glob("*.txt")):
            n_poses = len(np.loadtxt(est, ndmin=2))
            row = {"seq": seq, "method": est.stem, "poses": n_poses, "frames": n_frames,
                   "coverage": n_poses / n_frames if n_frames else None, "note": ""}
            if gt.exists() and n_poses:
                try:
                    row.update(score(gt, est, args.window, args.max_diff))
                except (sync.SyncException, geometry.GeometryException) as e:  # no overlap, degenerate fit
                    row["note"] = f"not scored: {type(e).__name__}"
            elif not gt.exists():
                row["note"] = "no GT"
            rows.append(row)

    lines = ["## Per sequence", "",
             "| Seq | Method | Poses | Coverage % | ATE SE3 [mm] | ATE Sim3 [mm] | RPE 1 s [mm] | Scale | Note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seq']} | {r['method']} | {r['poses']}/{r['frames'] or '?'} | {fmt(r['coverage'], 100)} | "
                     f"{fmt(r.get('ate_se3'), 1e3)} | {fmt(r.get('ate_sim3'), 1e3)} | {fmt(r.get('rpe'), 1e3)} | "
                     f"{fmt(r.get('scale'), 1, 3)} | {r['note']} |")

    lines += ["", "## Per method (medians over sequences with GT; coverage over all)", "",
              ("| Method | Seqs | Full coverage (>=95 %) | Median coverage % | Median ATE SE3 [mm] "
               "| Worst ATE SE3 [mm] | Median RPE 1 s [mm] | Median scale |"), "|---|---|---|---|---|---|---|---|"]
    for m in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == m]
        cov = [r["coverage"] for r in rs if r["coverage"] is not None]
        scored = [r for r in rs if "ate_se3" in r]
        med = {k: float(np.nanmedian([r[k] for r in scored])) if scored else None
               for k in ("ate_se3", "rpe", "scale")}
        worst = max((r["ate_se3"] for r in scored), default=None)
        lines.append(f"| {m} | {len(rs)} | {sum(c >= 0.95 for c in cov)}/{len(cov)} | "
                     f"{fmt(float(np.median(cov)) if cov else None, 100)} | {fmt(med['ate_se3'], 1e3)} | "
                     f"{fmt(worst, 1e3)} | {fmt(med['rpe'], 1e3)} | {fmt(med['scale'], 1, 3)} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
