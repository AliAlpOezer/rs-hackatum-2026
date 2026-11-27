"""Convert one EuRoC MAV sequence (ASL format) into the shared sequence layout.

    python data/tools/euroc_to_seq.py <path>/MH_01_easy data/MH_01_easy

Writes frames/ (symlink to cam0 images, already named <timestamp_ns>.png),
imu.csv, calib.yaml and gt.txt (TUM, camera frame) as docs/architecture.md defines.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def load_sensor_yaml(path: Path) -> dict:
    # EuRoC sensor.yaml starts with a "%YAML:1.0" directive that PyYAML rejects.
    text = "\n".join(line for line in path.read_text().splitlines() if not line.startswith("%"))
    return yaml.safe_load(text)


def matrix(entry: dict) -> np.ndarray:
    return np.array(entry["data"], dtype=float).reshape(entry["rows"], entry["cols"])


def link_frames(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst, target_is_directory=True)
    except OSError:  # no symlink privilege (Windows): copy instead
        shutil.copytree(src, dst)


def write_imu(src: Path, dst: Path) -> None:
    # EuRoC columns already match ours: t_ns, wx, wy, wz, ax, ay, az.
    with src.open() as f_in, dst.open("w", newline="") as f_out:
        reader = csv.reader(row for row in f_in if not row.startswith("#"))
        writer = csv.writer(f_out)
        writer.writerow(["timestamp_ns", "wx", "wy", "wz", "ax", "ay", "az"])
        for row in reader:
            writer.writerow([c.strip() for c in row[:7]])


def write_calib(cam: dict, imu: dict, dst: Path) -> np.ndarray:
    T_imu_cam = matrix(cam["T_BS"]) @ np.linalg.inv(matrix(imu["T_BS"]))
    calib = {
        "camera": {
            "model": cam["camera_model"],
            "width": cam["resolution"][0],
            "height": cam["resolution"][1],
            "intrinsics": cam["intrinsics"],  # fx fy cx cy
            "distortion_model": cam["distortion_model"],
            "distortion": cam["distortion_coefficients"],  # k1 k2 p1 p2
            "rate_hz": cam["rate_hz"],
        },
        "imu": {
            "rate_hz": imu["rate_hz"],
            "gyroscope_noise_density": imu["gyroscope_noise_density"],
            "gyroscope_random_walk": imu["gyroscope_random_walk"],
            "accelerometer_noise_density": imu["accelerometer_noise_density"],
            "accelerometer_random_walk": imu["accelerometer_random_walk"],
        },
        # camera pose in the IMU frame: p_imu = T_imu_cam @ p_cam
        "T_imu_cam": T_imu_cam.round(12).tolist(),
        "time_offset_s": 0.0,  # t_imu = t_cam + offset; EuRoC is hardware-synced
    }
    dst.write_text(yaml.safe_dump(calib, sort_keys=False))
    return T_imu_cam


def write_gt(src: Path, T_imu_cam: np.ndarray, dst: Path) -> None:
    # Ground truth is the IMU (body) pose in world; we store the camera pose.
    data = np.loadtxt(src, delimiter=",", comments="#")
    t_s = data[:, 0] * 1e-9
    p, q_wxyz = data[:, 1:4], data[:, 4:8]
    R_wb = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])
    R_wc = R_wb * Rotation.from_matrix(T_imu_cam[:3, :3])
    p_wc = p + R_wb.apply(T_imu_cam[:3, 3])
    tum = np.column_stack([t_s, p_wc, R_wc.as_quat()])  # qx qy qz qw
    np.savetxt(dst, tum, fmt="%.9f", header="timestamp_s tx ty tz qx qy qz qw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("euroc_seq", type=Path, help="folder that contains mav0/")
    parser.add_argument("out", type=Path, help="target data/<seq> folder")
    args = parser.parse_args()

    mav0 = args.euroc_seq / "mav0"
    if not mav0.is_dir():
        raise SystemExit(f"{mav0} not found; pass the folder that contains mav0/")
    args.out.mkdir(parents=True, exist_ok=True)

    cam = load_sensor_yaml(mav0 / "cam0" / "sensor.yaml")
    imu = load_sensor_yaml(mav0 / "imu0" / "sensor.yaml")

    link_frames(mav0 / "cam0" / "data", args.out / "frames")
    write_imu(mav0 / "imu0" / "data.csv", args.out / "imu.csv")
    T_imu_cam = write_calib(cam, imu, args.out / "calib.yaml")
    write_gt(mav0 / "state_groundtruth_estimate0" / "data.csv", T_imu_cam, args.out / "gt.txt")

    n_frames = len(list((args.out / "frames").glob("*.png")))
    print(f"{args.out}: {n_frames} frames, imu.csv, calib.yaml, gt.txt")


if __name__ == "__main__":
    main()
