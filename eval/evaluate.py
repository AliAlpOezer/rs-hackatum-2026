"""Score TUM trajectories against ground truth and print a markdown table.

    python eval/evaluate.py data/MH_01_easy/gt.txt results/MH_01_easy/*.txt

ATE: translation RMSE after Sim(3) alignment (monocular methods have no scale),
plus SE(3) alignment when --metric is set. RPE: translation RMSE over a time
window (default 1 s) after the same Sim(3) alignment. Scale is the Sim(3)
factor; far from 1.0 means the method's scale is wrong.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
from evo.core import filters, metrics, sync
from evo.tools import file_interface


def ate(ref, est) -> float:
    m = metrics.APE(metrics.PoseRelation.translation_part)
    m.process_data((ref, est))
    return m.get_statistic(metrics.StatisticsType.rmse)


def rpe(ref, est, window_s: float) -> float:
    # evo has no time-based delta; convert the window to frames at the median rate.
    dt = float(np.median(np.diff(est.timestamps)))
    m = metrics.RPE(metrics.PoseRelation.translation_part, delta=max(1, round(window_s / dt)),
                    delta_unit=metrics.Unit.frames, all_pairs=False)
    try:
        m.process_data((ref, est))
    except filters.FilterException:  # trajectory shorter than the window (e.g. lost tracking)
        return float("nan")
    return m.get_statistic(metrics.StatisticsType.rmse)


def score(gt_path: Path, est_path: Path, window_s: float, max_diff: float, metric: bool) -> dict:
    ref = file_interface.read_tum_trajectory_file(str(gt_path))
    est = file_interface.read_tum_trajectory_file(str(est_path))
    ref, est = sync.associate_trajectories(ref, est, max_diff=max_diff)

    sim3 = copy.deepcopy(est)
    _, _, scale = sim3.align(ref, correct_scale=True)
    row = {
        "method": est_path.stem,
        "poses": est.num_poses,
        "ate_sim3": ate(ref, sim3),
        "rpe": rpe(ref, sim3, window_s),
        "scale": scale,
    }
    if metric:
        se3 = copy.deepcopy(est)
        se3.align(ref, correct_scale=False)
        row["ate_se3"] = ate(ref, se3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gt", type=Path, help="data/<seq>/gt.txt")
    parser.add_argument("est", type=Path, nargs="+", help="results/<seq>/<method>.txt")
    parser.add_argument("--window", type=float, default=1.0, help="RPE window in seconds")
    parser.add_argument("--max-diff", type=float, default=0.01, help="timestamp association tolerance [s]")
    parser.add_argument("--metric", action="store_true", help="also report SE(3) ATE (metric-scale methods)")
    args = parser.parse_args()

    rows = [score(args.gt, e, args.window, args.max_diff, args.metric) for e in args.est]
    cols = ["method", "poses", "ate_sim3", "rpe", "scale"] + (["ate_se3"] if args.metric else [])
    names = {"ate_sim3": "ATE Sim3 [m]", "rpe": f"RPE {args.window:g} s [m]",
             "ate_se3": "ATE SE3 [m]", "scale": "Scale"}
    print(f"Sequence: {args.gt.parent.name}\n")
    print("| " + " | ".join(names.get(c, c.title()) for c in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
