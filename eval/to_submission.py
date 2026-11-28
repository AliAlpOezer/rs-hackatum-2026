"""Turn a TUM trajectory into the R&S submission format.

    python eval/to_submission.py results/seq11/dpvo.txt data/seq11 submission/seq11/trajectory.txt

Output: one line per frame of timestamps.csv, same order, original timestamp
strings, `timestamp_ms tx ty tz qx qy qz qw`. Poses at frames the method skipped
(stride, dropped frames) are interpolated (linear position, slerp rotation);
frames outside the estimate are held at the nearest pose.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("est", type=Path, help="TUM trajectory, seconds")
    parser.add_argument("seq", type=Path, help="data/<seq> with timestamps_ms.txt")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    stamps_ms = (args.seq / "timestamps_ms.txt").read_text().split()
    t_query = np.array([float(s) for s in stamps_ms]) * 1e-3

    est = np.loadtxt(args.est)
    est = est[np.unique(est[:, 0], return_index=True)[1]]
    t, pos, rot = est[:, 0], est[:, 1:4], Rotation.from_quat(est[:, 4:8])

    tq = np.clip(t_query, t[0], t[-1])
    pos_q = np.column_stack([np.interp(tq, t, pos[:, i]) for i in range(3)])
    quat_q = Slerp(t, rot)(tq).as_quat()  # x y z w

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="\n") as f:
        for s, p, q in zip(stamps_ms, pos_q, quat_q):
            f.write(f"{s} {p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}\n")
    covered = np.mean((t_query >= t[0]) & (t_query <= t[-1]))
    print(f"{args.out}: {len(stamps_ms)} poses, {covered:.0%} inside the estimate's time range")


if __name__ == "__main__":
    main()
