"""Convert an ORB-SLAM3 EuRoC-style trajectory to the TUM contract (T_world_cam, seconds).

    python tracks/orbslam3/orbslam3_to_tum.py f_seq01.txt results/seq01/orbslam3_stereo_inertial.txt \
        --frame body --settings <seq>/orbslam3_stereo_inertial.yaml
    python tracks/orbslam3/orbslam3_to_tum.py f_seq01.txt results/seq01/orbslam3_stereo.txt --frame camera

System::SaveTrajectoryEuRoC (src/System.cc) writes `1e9*t tx ty tz qx qy qz qw`:
the IMU body pose T_world_body for IMU_* sensors (stereo_inertial, mono_inertial),
the left camera pose T_world_cam otherwise. --frame body applies
T_world_cam = T_world_body @ T_body_cam with T_body_cam = IMU.T_b_c1 from the settings.
Only the largest map is saved; frames tracked in other maps are missing.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_T_b_c1(settings: Path) -> np.ndarray:
    fs = cv2.FileStorage(str(settings), cv2.FILE_STORAGE_READ)
    node = fs.getNode("IMU.T_b_c1")
    if node.empty():
        raise SystemExit(f"{settings} has no IMU.T_b_c1 (not an inertial settings file)")
    return node.mat().astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("est", type=Path, help="ORB-SLAM3 f_<name>.txt or kf_<name>.txt (EuRoC, ns)")
    parser.add_argument("out", type=Path, help="results/<seq>/<method>.txt")
    parser.add_argument("--frame", choices=("body", "camera"), required=True,
                        help="body for inertial runs, camera for visual-only runs")
    parser.add_argument("--settings", type=Path, help="ORB-SLAM3 yaml with IMU.T_b_c1 (needed for --frame body)")
    args = parser.parse_args()

    d = np.loadtxt(args.est, comments="#", ndmin=2)
    if len(d) == 0:
        raise SystemExit(f"{args.est} is empty (ORB-SLAM3 never initialized?)")
    d = d[np.unique(d[:, 0], return_index=True)[1]]  # sorted, one pose per timestamp
    t_s = d[:, 0] * 1e-9
    R, p = Rotation.from_quat(d[:, 4:8]), d[:, 1:4]
    if args.frame == "body":
        if args.settings is None:
            raise SystemExit("--frame body needs --settings")
        T_bc = read_T_b_c1(args.settings)
        p = p + R.apply(T_bc[:3, 3])
        R = R * Rotation.from_matrix(T_bc[:3, :3])

    if len(t_s) > 1 and not 1e-3 < np.median(np.diff(t_s)) < 10:
        raise SystemExit(f"{args.est}: odd frame spacing {np.median(np.diff(t_s)):.3g} s, expected ns input")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out, np.column_stack([t_s, p, R.as_quat()]), fmt="%.9f",
               header="timestamp_s tx ty tz qx qy qz qw  (T_world_cam, left camera)")
    print(f"{args.out}: {len(t_s)} poses, {t_s[-1] - t_s[0]:.2f} s, from {args.frame} poses")


if __name__ == "__main__":
    main()
