"""Compare ORB-SLAM3 stereo with and without IMU fusion using evo: metrics table and plots.

    uv run python tracks/imu_fusion/compare_evo.py <raw_root> results [--out DIR] [--all]

<raw_root>/seqNN/trajectory.txt is the R&S ground truth (ms timestamps). Estimates:
stereo = results/results_euroc_color/stereo/f_seqNN.txt, imu_fusion =
results/seqNN/orbslam3-imu-fusion.txt; --all adds eskf_learned / eskf_trust (steps 2-3).
Every estimate is SE(3)-aligned to GT (both are metric). Reports ATE translation, RPE 1 s
translation and RPE 1 s rotation, writes <out>/compare.md and one <out>/seqNN.png per sequence.
"""

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from evaluate import ate, rpe  # noqa: E402

METHODS = {
    "stereo": "results_euroc_color/stereo/f_{seq}.txt",
    "imu_fusion": "{seq}/orbslam3-imu-fusion.txt",
}
EXTRA = {"eskf_learned": "{seq}/eskf_learned.txt", "eskf_trust": "{seq}/eskf_trust.txt"}
COLORS = {"stereo": "tab:blue", "imu_fusion": "tab:orange", "eskf_learned": "tab:green", "eskf_trust": "tab:red"}


def load_gt(path: Path) -> PoseTrajectory3D:
    d = np.loadtxt(path)
    # TUM quaternion x y z w -> evo w x y z
    return PoseTrajectory3D(d[:, 1:4], d[:, [7, 4, 5, 6]], d[:, 0] / 1e3)


def rpe_rot(ref, est, window_s: float) -> float:
    dt = float(np.median(np.diff(est.timestamps)))
    m = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, delta=max(1, round(window_s / dt)),
                    delta_unit=metrics.Unit.frames, all_pairs=False)
    m.process_data((ref, est))
    return m.get_statistic(metrics.StatisticsType.rmse)


def evaluate(gt: PoseTrajectory3D, est_path: Path, window_s: float, max_diff: float):
    est = file_interface.read_tum_trajectory_file(str(est_path))
    ref, est = sync.associate_trajectories(copy.deepcopy(gt), est, max_diff=max_diff)
    est.align(ref, correct_scale=False)
    # GT body frame differs from T_world_cam by a constant rotation; after the world alignment,
    # remove it by right-multiplying (positions unchanged, so ATE is unaffected)
    q = lambda t: Rotation.from_quat(t.orientations_quat_wxyz[:, [1, 2, 3, 0]])
    off = (q(est).inv() * q(ref)).mean().as_matrix()
    T = np.eye(4)
    T[:3, :3] = off
    est.transform(T, right_mul=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    row = {"ate": ate(ref, est) * 1e3, "rpe": rpe(ref, est, window_s) * 1e3, "rpe_rot": rpe_rot(ref, est, window_s)}
    return row, ref, est, ape.error


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw_root", type=Path, help="R&S root with seqNN/trajectory.txt")
    ap.add_argument("results", type=Path, help="results root")
    ap.add_argument("--out", type=Path, default=Path("results/imu_fusion/compare"))
    ap.add_argument("--all", action="store_true", help="also compare eskf_learned / eskf_trust")
    ap.add_argument("--window", type=float, default=1.0, help="RPE window [s]")
    ap.add_argument("--max-diff", type=float, default=0.01, help="timestamp association tolerance [s]")
    a = ap.parse_args()
    methods = {**METHODS, **(EXTRA if a.all else {})}
    a.out.mkdir(parents=True, exist_ok=True)

    lines = ["| Seq | " + " | ".join(methods) + " | Δ fusion − stereo |", "|---|" + "---|" * (len(methods) + 1)]
    totals = {m: [] for m in methods}
    fmt = lambda r: f"{r['ate']:.2f} / {r['rpe']:.2f} / {r['rpe_rot']:.3f}"
    for gt_path in sorted(a.raw_root.glob("seq*/trajectory.txt")):
        seq = gt_path.parent.name
        gt = load_gt(gt_path)
        rows, trajs = {}, {}
        for m, pattern in methods.items():
            f = a.results / pattern.format(seq=seq)
            if f.exists():
                rows[m], ref, est, err = evaluate(gt, f, a.window, a.max_diff)
                trajs[m] = (ref, est, err)
                totals[m].append(rows[m])
        if not rows:
            continue
        delta = "-"
        if "stereo" in rows and "imu_fusion" in rows:
            s, f = rows["stereo"], rows["imu_fusion"]
            delta = f"{f['ate'] - s['ate']:+.2f} / {f['rpe'] - s['rpe']:+.2f} / {f['rpe_rot'] - s['rpe_rot']:+.3f}"
        lines.append(f"| {seq} | " + " | ".join(fmt(rows[m]) if m in rows else "-" for m in methods) + f" | {delta} |")

        fig = plt.figure(figsize=(13, 5))
        ax = plot.prepare_axis(fig, plot.PlotMode.xy, subplot_arg=121)
        ref0 = next(iter(trajs.values()))[0]
        plot.traj(ax, plot.PlotMode.xy, ref0, "--", "gray", "ground truth")
        for m, (_, est, _) in trajs.items():
            plot.traj(ax, plot.PlotMode.xy, est, "-", COLORS[m], m)
        ax.set_title(f"{seq} top-down (SE3-aligned)")
        ax.legend()
        ax2 = fig.add_subplot(122)
        for m, (ref, _, err) in trajs.items():
            ax2.plot(ref.timestamps - ref.timestamps[0], err * 1e3, color=COLORS[m], label=f"{m} ({rows[m]['ate']:.2f} mm)")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("APE translation [mm]")
        ax2.set_title("position error over time")
        ax2.legend()
        fig.savefig(a.out / f"{seq}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    mean = lambda m: {k: float(np.mean([r[k] for r in totals[m]])) for k in ("ate", "rpe", "rpe_rot")}
    means = {m: mean(m) for m in methods if totals[m]}
    delta = "-"
    if "stereo" in means and "imu_fusion" in means:
        s, f = means["stereo"], means["imu_fusion"]
        delta = f"{f['ate'] - s['ate']:+.2f} / {f['rpe'] - s['rpe']:+.2f} / {f['rpe_rot'] - s['rpe_rot']:+.3f}"
    lines.append("| mean | " + " | ".join(fmt(means[m]) if m in means else "-" for m in methods) + f" | {delta} |")
    lines.append(f"\nATE SE3 mm / RPE {a.window:g} s mm / RPE {a.window:g} s rotation deg (evo, SE3 alignment)")
    text = "\n".join(lines)
    print(text)
    (a.out / "compare.md").write_text(text + "\n")
    print(f"\nplots + table -> {a.out}/")


if __name__ == "__main__":
    main()
