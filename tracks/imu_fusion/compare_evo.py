"""Compare ORB-SLAM3 stereo with and without IMU fusion using evo: metrics table and plots.

    uv run python tracks/imu_fusion/compare_evo.py <raw_root> results [--out DIR] [--all]

<raw_root>/seqNN/trajectory.txt is the R&S ground truth (ms timestamps). Estimates:
stereo = results/results_euroc_color/stereo/seqNN/f_seqNN.txt, imu_fusion =
results/seqNN/orbslam3-imu-fusion.txt, stereo_inertial = ORB-SLAM3's own tightly coupled
mode (results/results_euroc_color/stereo_inertial/seqNN/f_seqNN.txt), si_fusion = the ESKF run on
stereo_inertial (results/seqNN/orbslam3-stereo-inertial-imu-fusion.txt); --all adds eskf_learned / eskf_trust (steps 2-3).
Every estimate is SE(3)-aligned to GT (both are metric). Reports ATE translation, RPE 1 s
translation and RPE 1 s rotation, writes <out>/compare.md, per sequence <out>/seqNN.png (trajectory +
APE over time) and <out>/seqNN_rpe.png (RPE over time), and <out>/rpe_summary.png (RPE per sequence).
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
from evaluate import ate  # noqa: E402

METHODS = {
    "stereo": "results_euroc_color/stereo/{seq}/f_{seq}.txt",
    "imu_fusion": "{seq}/orbslam3-imu-fusion.txt",
    "stereo_inertial": "results_euroc_color/stereo_inertial/{seq}/f_{seq}.txt",
    "si_fusion": "{seq}/orbslam3-stereo-inertial-imu-fusion.txt",
}
EXTRA = {"eskf_learned": "{seq}/eskf_learned.txt", "eskf_trust": "{seq}/eskf_trust.txt"}
COLORS = {"stereo": "tab:blue", "imu_fusion": "tab:orange", "stereo_inertial": "tab:purple", "si_fusion": "tab:brown", "eskf_learned": "tab:green",
          "eskf_trust": "tab:red"}


def load_gt(path: Path) -> PoseTrajectory3D:
    d = np.loadtxt(path)
    # TUM quaternion x y z w -> evo w x y z
    return PoseTrajectory3D(d[:, 1:4], d[:, [7, 4, 5, 6]], d[:, 0] / 1e3)


def rpe_metric(ref, est, relation, window_s: float) -> metrics.RPE:
    # evo has no time-based delta; convert the window to frames at the median rate (as eval/evaluate.py)
    dt = float(np.median(np.diff(est.timestamps)))
    m = metrics.RPE(relation, delta=max(1, round(window_s / dt)), delta_unit=metrics.Unit.frames, all_pairs=False)
    m.process_data((ref, est))
    return m


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
    rt = rpe_metric(ref, est, metrics.PoseRelation.translation_part, window_s)
    rr = rpe_metric(ref, est, metrics.PoseRelation.rotation_angle_deg, window_s)
    rmse = metrics.StatisticsType.rmse
    row = {"n": ref.num_poses, "ate": ate(ref, est) * 1e3, "rpe": rt.get_statistic(rmse) * 1e3,
           "rpe_rot": rr.get_statistic(rmse)}
    # RPE errors are stamped at the start of each 1 s pair
    t_rpe = ref.timestamps[np.array(rt.delta_ids)]
    return row, ref, est, {"ape": ape.error, "t_rpe": t_rpe, "rpe": rt.error, "rpe_rot": rr.error}


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
    seqs = []
    fmt = lambda r: f"{r['ate']:.2f} / {r['rpe']:.2f} / {r['rpe_rot']:.3f}"
    for gt_path in sorted(a.raw_root.glob("seq*/trajectory.txt")):
        seq = gt_path.parent.name
        gt = load_gt(gt_path)
        rows, trajs = {}, {}
        for m, pattern in methods.items():
            f = a.results / pattern.format(seq=seq)
            if f.exists():
                rows[m], ref, est, err = evaluate(gt, f, a.window, a.max_diff)
                rows[m]["seq"] = seq
                trajs[m] = (ref, est, err)
                totals[m].append(rows[m])
        if not rows:
            continue
        delta = "-"
        if "stereo" in rows and "imu_fusion" in rows:
            s, f = rows["stereo"], rows["imu_fusion"]
            delta = f"{f['ate'] - s['ate']:+.2f} / {f['rpe'] - s['rpe']:+.2f} / {f['rpe_rot'] - s['rpe_rot']:+.3f}"
        cell = lambda m: fmt(rows[m]) + (f" ({rows[m]['n']}/{gt.num_poses} poses)" if rows[m]["n"] < 0.9 * gt.num_poses else "")
        lines.append(f"| {seq} | " + " | ".join(cell(m) if m in rows else "-" for m in methods) + f" | {delta} |")

        fig = plt.figure(figsize=(13, 5))
        ax = plot.prepare_axis(fig, plot.PlotMode.xy, subplot_arg=121)
        ref0 = next(iter(trajs.values()))[0]
        plot.traj(ax, plot.PlotMode.xy, ref0, "--", "gray", "ground truth")
        for m, (_, est, _) in trajs.items():
            plot.traj(ax, plot.PlotMode.xy, est, "-", COLORS[m], m)
        ax.set_title(f"{seq} top-down (SE3-aligned)")
        ax.legend()
        ax2 = fig.add_subplot(122)
        t0 = gt.timestamps[0]
        for m, (ref, _, err) in trajs.items():
            ax2.plot(ref.timestamps - t0, err["ape"] * 1e3, color=COLORS[m], label=f"{m} ({rows[m]['ate']:.2f} mm)")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("APE translation [mm]")
        ax2.set_title("position error over time")
        ax2.legend()
        fig.savefig(a.out / f"{seq}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        for m, (_, _, err) in trajs.items():
            ax3.plot(err["t_rpe"] - t0, err["rpe"] * 1e3, color=COLORS[m], label=f"{m} ({rows[m]['rpe']:.2f} mm)")
            ax4.plot(err["t_rpe"] - t0, err["rpe_rot"], color=COLORS[m], label=f"{m} ({rows[m]['rpe_rot']:.3f} deg)")
        ax3.set_ylabel(f"RPE {a.window:g} s translation [mm]")
        ax3.set_title(f"{seq} relative pose error over {a.window:g} s windows (RMSE in legend)")
        ax3.legend()
        ax4.set_ylabel(f"RPE {a.window:g} s rotation [deg]")
        ax4.set_xlabel("time [s] (start of window)")
        ax4.legend()
        fig.savefig(a.out / f"{seq}_rpe.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        seqs.append(seq)

    # overview: RPE RMSE per sequence and method (missing / partial runs left as gaps)
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    x = np.arange(len(seqs))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = {k: [np.nan] * len(seqs) for k in ("rpe", "rpe_rot")}
        for r in totals[m]:
            j = seqs.index(r["seq"])
            vals["rpe"][j], vals["rpe_rot"][j] = r["rpe"], r["rpe_rot"]
        axs[0].bar(x + (i - (len(methods) - 1) / 2) * w, vals["rpe"], w, color=COLORS[m], label=m)
        axs[1].bar(x + (i - (len(methods) - 1) / 2) * w, vals["rpe_rot"], w, color=COLORS[m], label=m)
    axs[0].set_ylabel(f"RPE {a.window:g} s translation RMSE [mm]")
    axs[0].set_title("RPE per sequence")
    axs[0].legend(ncol=len(methods))
    axs[1].set_ylabel(f"RPE {a.window:g} s rotation RMSE [deg]")
    axs[1].set_xticks(x, seqs)
    fig.savefig(a.out / "rpe_summary.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    mean = lambda m: {k: float(np.mean([r[k] for r in totals[m]])) for k in ("ate", "rpe", "rpe_rot")}
    means = {m: mean(m) for m in methods if totals[m]}
    delta = "-"
    if "stereo" in means and "imu_fusion" in means:
        s, f = means["stereo"], means["imu_fusion"]
        delta = f"{f['ate'] - s['ate']:+.2f} / {f['rpe'] - s['rpe']:+.2f} / {f['rpe_rot'] - s['rpe_rot']:+.3f}"
    lines.append("| mean | " + " | ".join(fmt(means[m]) if m in means else "-" for m in methods) + f" | {delta} |")
    lines.append("\nA pose count marks a trajectory that lost tracking: its errors cover only the tracked part.")
    lines.append(f"ATE SE3 mm / RPE {a.window:g} s mm / RPE {a.window:g} s rotation deg (evo, SE3 alignment)")
    text = "\n".join(lines)
    print(text)
    (a.out / "compare.md").write_text(text + "\n")
    print(f"\nplots + table -> {a.out}/")


if __name__ == "__main__":
    main()
