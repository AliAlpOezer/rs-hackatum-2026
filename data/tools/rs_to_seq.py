"""Convert one R&S recording (ZED X Mini stereo + IMU) into the shared sequence layout.

    python data/tools/rs_to_seq.py <dataset_extracted>/seq01 data/seq01 [--no-right]

Writes frames/ and frames_right/ (<timestamp_ns>.png, decoded from left/right.mp4),
imu.csv (ns, rad/s, m/s^2), calib.yaml and, when trajectory.txt exists, gt.txt
(TUM, seconds). Images are already rectified, so distortion is zero.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def read_timestamps_ms(path: Path) -> list[str]:
    """Keep the original strings: the submission must repeat them exactly."""
    with path.open() as f:
        rows = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    return rows[1:] if not rows[0][0].isdigit() else rows


def ms_to_ns(ms: str) -> int:
    whole, _, frac = ms.partition(".")
    return int(whole) * 1_000_000 + int((frac + "000000")[:6])


def extract_frames(video: Path, out_dir: Path, stamps_ns: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if n < len(stamps_ns):
            cv2.imwrite(str(out_dir / f"{stamps_ns[n]}.png"), image)
        n += 1
    cap.release()
    if n != len(stamps_ns):
        raise SystemExit(f"{video}: {n} frames but {len(stamps_ns)} timestamps")


def write_imu(src: Path, dst: Path) -> None:
    # R&S: timestamp_ms, ax, ay, az [m/s^2], gx, gy, gz [deg/s]
    with src.open() as f_in, dst.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["timestamp_ns", "wx", "wy", "wz", "ax", "ay", "az"])
        for r in reader:
            w = np.radians([float(r["gx"]), float(r["gy"]), float(r["gz"])])
            writer.writerow([ms_to_ns(r["timestamp_ms"]), *(f"{v:.9f}" for v in w),
                             r["ax"], r["ay"], r["az"]])


def write_calib(calib: dict, dst: Path) -> None:
    left, right = calib["left"], calib["right"]
    T_cam_imu = np.array(calib["imu_to_left_camera"]["matrix_4x4_rowmajor"]).reshape(4, 4)
    T_left_right = np.eye(4)
    T_left_right[:3, :3] = np.array(calib["stereo_transform"]["rotation_matrix_3x3_rowmajor"]).reshape(3, 3)
    T_left_right[:3, 3] = calib["stereo_transform"]["translation_m"]
    out = {
        "camera": {
            "model": "pinhole",
            "width": left["width"],
            "height": left["height"],
            "intrinsics": [left["fx"], left["fy"], left["cx"], left["cy"]],
            "distortion_model": "none",
            "distortion": [],
            "rate_hz": calib["fps"],
            "source": f"{calib['camera_model']}, {calib['images']}",
        },
        "camera_right": {"intrinsics": [right["fx"], right["fy"], right["cx"], right["cy"]]},
        "baseline_m": calib["baseline_m"],
        "T_left_right": T_left_right.tolist(),  # right camera pose in left camera frame
        # camera pose in the IMU frame: p_imu = T_imu_cam @ p_cam (SDK gives IMU -> camera)
        "T_imu_cam": np.linalg.inv(T_cam_imu).round(12).tolist(),
        "time_offset_s": 0.0,
    }
    dst.write_text(yaml.safe_dump(out, sort_keys=False))


def write_gt(src: Path, dst: Path) -> None:
    rows = np.loadtxt(src)
    rows[:, 0] *= 1e-3  # ms -> s
    np.savetxt(dst, rows, fmt="%.9f", header="timestamp_s tx ty tz qx qy qz qw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rs_seq", type=Path, help="R&S recording folder (left.mp4, imu.csv, ...)")
    parser.add_argument("out", type=Path, help="target data/<seq> folder")
    parser.add_argument("--no-right", action="store_true", help="skip decoding right.mp4")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stamps_ms = read_timestamps_ms(args.rs_seq / "timestamps.csv")
    stamps_ns = [ms_to_ns(s) for s in stamps_ms]
    (args.out / "timestamps_ms.txt").write_text("\n".join(stamps_ms) + "\n")

    extract_frames(args.rs_seq / "left.mp4", args.out / "frames", stamps_ns)
    if not args.no_right:
        extract_frames(args.rs_seq / "right.mp4", args.out / "frames_right", stamps_ns)
    write_imu(args.rs_seq / "imu.csv", args.out / "imu.csv")
    write_calib(json.loads((args.rs_seq / "calibration.json").read_text()), args.out / "calib.yaml")
    has_gt = (args.rs_seq / "trajectory.txt").exists()
    if has_gt:
        write_gt(args.rs_seq / "trajectory.txt", args.out / "gt.txt")
    print(f"{args.out}: {len(stamps_ns)} frames, gt={'yes' if has_gt else 'no (test sequence)'}")


if __name__ == "__main__":
    main()
