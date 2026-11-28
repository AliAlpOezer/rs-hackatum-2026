"""Compare the fusion steps per sequence: ATE, RPE and rotation error after SE(3) alignment.

Complements eval/evaluate.py, which does not report rotation error. Estimates are read
from results/<seq>/<method>.txt (Jonas's stereo from results/results_euroc_color/stereo/),
ground truth from <gt_dir>/<seq>/trajectory.txt (the raw R&S layout, ms timestamps).

    python tracks/imu_fusion/score.py "G:/My Drive/dataset_extracted" results
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

METHODS = {
    "orbslam3_stereo": "results_euroc_color/stereo/f_{seq}.txt",
    "eskf": "{seq}/eskf.txt",
    "eskf_learned": "{seq}/eskf_learned.txt",
    "eskf_trust": "{seq}/eskf_trust.txt",
}
RPE_FRAMES = 30  # ~1 s at 30 Hz


def match(gt, est):
    _, ig, ie = np.intersect1d(np.round(gt[:, 0], 4), np.round(est[:, 0], 4), return_indices=True)
    return gt[ig], est[ie]


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src))
    W = np.eye(3)
    W[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ W @ Vt
    return Rm, mu_d - Rm @ mu_s


def score(gt, est):
    g, e = match(gt, est)
    Rm, t = umeyama(e[:, 1:4], g[:, 1:4])
    p = e[:, 1:4] @ Rm.T + t
    err = np.linalg.norm(p - g[:, 1:4], axis=1)
    rg = R.from_quat(g[:, 4:8])
    re = R.from_matrix(Rm) * R.from_quat(e[:, 4:8])
    off = (re.inv() * rg).mean()  # constant body-frame convention offset
    rot = np.rad2deg(((re * off).inv() * rg).magnitude())
    k = RPE_FRAMES
    rpe = np.linalg.norm((g[k:, 1:4] - g[:-k, 1:4]) - (p[k:] - p[:-k]), axis=1)
    rms = lambda x: float(np.sqrt((x**2).mean()))
    return {"ate": rms(err) * 1e3, "rpe": rms(rpe) * 1e3, "rot": rms(rot)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gt_dir", type=Path, help="raw R&S root with <seq>/trajectory.txt")
    ap.add_argument("results", type=Path, help="results root with <seq>/<method>.txt")
    a = ap.parse_args()

    print("| Seq | " + " | ".join(METHODS) + " |")
    print("|---|" + "---|" * len(METHODS))
    totals = {m: [] for m in METHODS}
    for gt_path in sorted(a.gt_dir.glob("seq*/trajectory.txt")):
        seq = gt_path.parent.name
        gt = np.loadtxt(gt_path)
        gt[:, 0] /= 1e3  # raw R&S timestamps are ms, results are s
        cells = []
        for m in METHODS:
            f = a.results / METHODS[m].format(seq=seq)
            if not f.exists():
                cells.append("-")
                continue
            s = score(gt, np.loadtxt(f))
            totals[m].append(s)
            cells.append(f"{s['ate']:.2f} / {s['rpe']:.2f} / {s['rot']:.3f}")
        print(f"| {seq} | " + " | ".join(cells) + " |")
    cells = []
    for m in METHODS:
        ss = totals[m]
        cells.append(
            " / ".join(f"{np.mean([s[k] for s in ss]):.{3 if k == 'rot' else 2}f}" for k in ("ate", "rpe", "rot"))
            if ss
            else "-"
        )
    print("| mean | " + " | ".join(cells) + " |")
    print("\nATE SE3 mm / RPE 1 s mm / rotation RMS deg")


if __name__ == "__main__":
    main()
