"""Convert R&S ZED X Mini sequences (left.mp4, right.mp4, imu.csv, ...) to EuRoC ASL.

    python data/tools/zed_to_euroc.py ../dataset_extracted ../dataset_euroc
    python data/tools/zed_to_euroc.py ../dataset_extracted/seq01 ../dataset_euroc/seq01

The first argument is one sequence folder (has calibration.json) or a folder of
seq* folders. Output per sequence is <out>/mav0/{cam0,cam1,imu0,state_groundtruth_estimate0},
readable by data/tools/euroc_to_seq.py. Needs ffmpeg on PATH.

Assumptions (checked on seq01: trajectory rotation rate matches the gyro):
- imu.csv gyro is deg/s (calibration.json says so), accel m/s^2.
- trajectory.txt is the left-camera pose T_world_cam, time in ms, same clock as frames.
- calibration.json imu_to_left_camera maps IMU coordinates to left-camera coordinates.
EuRoC ground truth is the body pose, so the body frame here is the IMU.
IMU noise values in imu0/sensor.yaml are placeholders, not calibrated.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

IMU_NOISE = {  # placeholders (EuRoC ADIS16448 order of magnitude), not measured for the ZED
    "gyroscope_noise_density": 1.7e-4,
    "gyroscope_random_walk": 1.9e-5,
    "accelerometer_noise_density": 2.0e-3,
    "accelerometer_random_walk": 3.0e-3,
}


def ms_to_ns(t_ms) -> np.ndarray:
    return np.round(np.asarray(t_ms, dtype=float) * 1e6).astype(np.int64)


def yaml_matrix(name: str, m: np.ndarray) -> str:
    data = ", ".join(f"{v:.12g}" for v in m.ravel())
    return f"{name}:\n  rows: {m.shape[0]}\n  cols: {m.shape[1]}\n  data: [{data}]\n"


def decode_video(video: Path, stamps_ns: np.ndarray, out_dir: Path, gray: bool) -> None:
    """Decode every frame of `video` to out_dir/<timestamp_ns>.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-fps_mode", "passthrough"]
        cmd += ["-pix_fmt", "gray" if gray else "rgb24", str(Path(tmp) / "%06d.png")]
        subprocess.run(cmd, check=True)
        frames = sorted(Path(tmp).glob("*.png"))
        if len(frames) != len(stamps_ns):
            raise SystemExit(f"{video}: {len(frames)} frames but {len(stamps_ns)} timestamps")
        for frame, t in zip(frames, stamps_ns):
            shutil.move(frame, out_dir / f"{t}.png")


def write_frame_index(stamps_ns: np.ndarray, dst: Path) -> None:
    lines = ["#timestamp [ns],filename"] + [f"{t},{t}.png" for t in stamps_ns]
    dst.write_text("\n".join(lines) + "\n")


def write_imu(src: Path, dst: Path) -> float:
    d = np.loadtxt(src, delimiter=",", skiprows=1)  # t_ms ax ay az gx gy gz
    out = np.column_stack([ms_to_ns(d[:, 0]), np.deg2rad(d[:, 4:7]), d[:, 1:4]])
    header = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
              "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")
    with dst.open("w") as f:
        f.write(header + "\n")
        for row in out:
            f.write(f"{int(row[0])}," + ",".join(f"{v:.9f}" for v in row[1:]) + "\n")
    return 1e3 / float(np.median(np.diff(d[:, 0])))


def write_sensor_yamls(calib: dict, imu_rate: float, mav0: Path) -> np.ndarray:
    T_cam_imu = np.array(calib["imu_to_left_camera"]["matrix_4x4_rowmajor"]).reshape(4, 4)
    T_imu_cam0 = np.linalg.inv(T_cam_imu)  # EuRoC T_BS: sensor pose in the body (IMU) frame
    T_cam0_cam1 = np.eye(4)
    T_cam0_cam1[:3, 3] = calib["stereo_transform"]["translation_m"]
    T_cam0_cam1[:3, :3] = np.array(calib["stereo_transform"]["rotation_matrix_3x3_rowmajor"]).reshape(3, 3)
    for name, side, T_bs in (("cam0", "left", T_imu_cam0), ("cam1", "right", T_imu_cam0 @ T_cam0_cam1)):
        c = calib[side]
        text = "%YAML:1.0\n" + yaml_matrix("T_BS", T_bs)
        text += f"rate_hz: {calib['fps']}\n"
        text += f"resolution: [{c['width']}, {c['height']}]\n"
        text += "camera_model: pinhole\n"
        text += f"intrinsics: [{c['fx']:.9g}, {c['fy']:.9g}, {c['cx']:.9g}, {c['cy']:.9g}]\n"
        text += "distortion_model: radial-tangential\n"
        text += "distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n"  # images are rectified
        (mav0 / name / "sensor.yaml").write_text(text)
    text = "%YAML:1.0\n" + yaml_matrix("T_BS", np.eye(4))
    text += f"rate_hz: {imu_rate:.1f}\n"
    text += "".join(f"{k}: {v}\n" for k, v in IMU_NOISE.items())
    (mav0 / "imu0" / "sensor.yaml").write_text(text)
    return T_cam_imu


def write_gt(src: Path, T_cam_imu: np.ndarray, dst: Path) -> None:
    d = np.loadtxt(src)  # t_ms x y z qx qy qz qw, left camera in world
    R_wc = Rotation.from_quat(d[:, 4:8])
    R_ci, t_ci = T_cam_imu[:3, :3], T_cam_imu[:3, 3]
    R_wi = R_wc * Rotation.from_matrix(R_ci)  # T_w_imu = T_w_cam @ T_cam_imu
    p_wi = d[:, 1:4] + R_wc.apply(t_ci)
    q_wxyz = R_wi.as_quat()[:, [3, 0, 1, 2]]
    header = ("#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
              "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],"
              "v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],"
              "b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],"
              "b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]")
    with dst.open("w") as f:
        f.write(header + "\n")
        for t, p, q in zip(ms_to_ns(d[:, 0]), p_wi, q_wxyz):
            vals = ",".join(f"{v:.9f}" for v in [*p, *q, *([0.0] * 9)])  # velocity and biases unknown
            f.write(f"{t},{vals}\n")


def convert(seq: Path, out: Path, gray: bool) -> None:
    mav0 = out / "mav0"
    for sub in ("cam0/data", "cam1/data", "imu0", "state_groundtruth_estimate0"):
        (mav0 / sub).mkdir(parents=True, exist_ok=True)

    calib = json.loads((seq / "calibration.json").read_text())
    stamps_ns = ms_to_ns(np.loadtxt(seq / "timestamps.csv", skiprows=1))
    for cam, video in (("cam0", "left.mp4"), ("cam1", "right.mp4")):
        decode_video(seq / video, stamps_ns, mav0 / cam / "data", gray)
        write_frame_index(stamps_ns, mav0 / cam / "data.csv")
    imu_rate = write_imu(seq / "imu.csv", mav0 / "imu0" / "data.csv")
    T_cam_imu = write_sensor_yamls(calib, imu_rate, mav0)

    gt = seq / "trajectory.txt"
    if gt.exists():
        write_gt(gt, T_cam_imu, mav0 / "state_groundtruth_estimate0" / "data.csv")
    else:
        (mav0 / "state_groundtruth_estimate0").rmdir()
    print(f"{seq.name}: {len(stamps_ns)} stereo frames, imu {imu_rate:.0f} Hz, "
          f"gt {'yes' if gt.exists() else 'none'} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="one sequence folder, or a folder of seq* folders")
    parser.add_argument("out", type=Path, help="target folder (per-sequence subfolders when src is a parent)")
    parser.add_argument("--gray", action="store_true", help="write 8-bit grayscale PNGs like EuRoC")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")
    if (args.src / "calibration.json").exists():
        jobs = [(args.src, args.out)]
    else:
        jobs = [(s, args.out / s.name) for s in sorted(args.src.glob("seq*")) if (s / "calibration.json").exists()]
    if not jobs:
        raise SystemExit(f"no sequences (calibration.json) under {args.src}")
    for seq, out in jobs:
        convert(seq, out, args.gray)


if __name__ == "__main__":
    main()
