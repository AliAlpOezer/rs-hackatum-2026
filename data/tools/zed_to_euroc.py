"""Convert R&S ZED X Mini sequences (left.mp4, right.mp4, imu.csv, ...) to EuRoC ASL + ORB-SLAM3 inputs.

    python data/tools/zed_to_euroc.py ../dataset_extracted ../dataset_euroc --gray
    python data/tools/zed_to_euroc.py ../dataset_extracted/seq01 ../dataset_euroc/seq01 --gray

The first argument is one sequence folder (has calibration.json) or a folder of
seq* folders. Needs ffmpeg on PATH. Output per sequence <out>/:

    mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}   EuRoC ASL (read by euroc_to_seq.py)
    timestamps.txt                  cam0 frame names without .png, the ORB-SLAM3 times file
    orbslam3_stereo.yaml            ORB-SLAM3 settings (File.version 1.0), per sequence
    orbslam3_stereo_inertial.yaml
    orbslam3_mono_inertial.yaml
    calibration.json                copy of the source calibration
    timestamps_ms.txt               original frame timestamp strings, all frames (eval/to_submission.py)
    conversion.yaml                 what was dropped, IMU timing, IMU noise source

Frames without IMU coverage are dropped: ORB-SLAM3's *_inertial_euroc loaders
index the IMU vector without bounds checks, so they need imu[0] <= first frame
and last frame < imu[-1]. On the R&S data the IMU ends ~1 ms before the last frame.

Assumptions (checked on seq01: trajectory rotation rate matches the gyro):
- imu.csv gyro is deg/s (calibration.json says so), accel m/s^2.
- trajectory.txt is the left-camera pose T_world_cam, time in ms, same clock as frames.
- calibration.json imu_to_left_camera maps IMU coordinates to left-camera coordinates.
- camera-IMU time offset is 0 (not estimated).
EuRoC ground truth is the body pose, so the body frame here is the IMU.
IMU noise: by default PLACEHOLDERS (EuRoC ADIS16448 values), not calibrated for the
ZED X Mini. --imu-noise quiet takes the white-noise terms from the quietest IMU
window instead (an upper bound unless the device was really still); random walks
stay placeholders either way. conversion.yaml and the yaml comments say which was used.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

IMU_NOISE = {  # placeholders (EuRoC ADIS16448 order of magnitude), not measured for the ZED
    "gyroscope_noise_density": 1.7e-4,  # rad/s/sqrt(Hz)
    "gyroscope_random_walk": 1.9e-5,  # rad/s^2/sqrt(Hz)
    "accelerometer_noise_density": 2.0e-3,  # m/s^2/sqrt(Hz)
    "accelerometer_random_walk": 3.0e-3,  # m/s^3/sqrt(Hz)
}
# EuRoC reference values that the ORB-SLAM3 parameters below are scaled from
EUROC_FX, EUROC_TH_DEPTH, EUROC_PIXELS = 458.654, 60.0, 752 * 480
EUROC_FEATURES = {"stereo": 1200, "stereo_inertial": 1200, "mono_inertial": 1000}


def ms_to_ns(t_ms) -> np.ndarray:
    return np.round(np.asarray(t_ms, dtype=float) * 1e6).astype(np.int64)


def read_stamp_strings(path: Path) -> list[str]:
    rows = [line.split(",")[0].strip() for line in path.read_text().splitlines()]
    return [r for r in rows if r and (r[0].isdigit() or r[0] == ".")]


def yaml_matrix(name: str, m: np.ndarray) -> str:
    data = ", ".join(f"{v:.12g}" for v in m.ravel())
    return f"{name}:\n  rows: {m.shape[0]}\n  cols: {m.shape[1]}\n  data: [{data}]\n"


def cv_matrix(name: str, m: np.ndarray) -> str:
    data = ", ".join(real(v) for v in m.ravel())
    return f"{name}: !!opencv-matrix\n  rows: {m.shape[0]}\n  cols: {m.shape[1]}\n  dt: f\n  data: [{data}]"


def real(v: float) -> str:
    """cv::FileStorage types by spelling and ORB-SLAM3 aborts on an int where it wants a real."""
    s = f"{float(v):.10g}"
    return s if re.search(r"[.e]", s) else s + ".0"


def decode_video(video: Path, names: list, out_dir: Path, gray: bool) -> None:
    """Decode every frame of `video`; frame i becomes out_dir/<names[i]>.png, None drops it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-fps_mode", "passthrough"]
        cmd += ["-pix_fmt", "gray" if gray else "rgb24", str(Path(tmp) / "%06d.png")]
        subprocess.run(cmd, check=True)
        frames = sorted(Path(tmp).glob("*.png"))
        if len(frames) != len(names):
            raise SystemExit(f"{video}: {len(frames)} frames but {len(names)} timestamps")
        for frame, name in zip(frames, names):
            if name is not None:
                shutil.move(frame, out_dir / f"{name}.png")


def write_frame_index(stamps_ns: np.ndarray, dst: Path) -> None:
    lines = ["#timestamp [ns],filename"] + [f"{t},{t}.png" for t in stamps_ns]
    dst.write_text("\n".join(lines) + "\n")


def load_imu(src: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """imu.csv -> (t_ns, gyro rad/s, accel m/s^2, samples removed), sorted, unique timestamps."""
    d = np.loadtxt(src, delimiter=",", skiprows=1)  # t_ms ax ay az gx gy gz
    t = ms_to_ns(d[:, 0])
    t_unique, idx = np.unique(t, return_index=True)  # sorts; keeps the first of duplicates
    d = d[idx]
    return t_unique, np.deg2rad(d[:, 4:7]), d[:, 1:4], len(t) - len(t_unique)


def write_imu(t_ns: np.ndarray, gyro: np.ndarray, acc: np.ndarray, dst: Path) -> None:
    header = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
              "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")
    with dst.open("w") as f:
        f.write(header + "\n")
        for t, w, a in zip(t_ns, gyro, acc):
            f.write(f"{t}," + ",".join(f"{v:.9f}" for v in (*w, *a)) + "\n")


def quiet_window_noise(t_ns: np.ndarray, gyro: np.ndarray, acc: np.ndarray, win_s: float = 1.0) -> dict:
    """White-noise densities from the IMU window with the least gyro variance.

    sigma_density = sigma_discrete * sqrt(dt). Any motion in the window inflates
    the result, so on handheld data it is an upper bound, not a calibration.
    """
    t = (t_ns - t_ns[0]) * 1e-9
    dt = float(np.median(np.diff(t)))
    best = None
    for start in np.arange(0.0, max(t[-1] - win_s, 0.0) + 1e-9, win_s / 4):
        m = (t >= start) & (t < start + win_s)
        if m.sum() < 20:
            continue
        g_std, a_std = gyro[m].std(axis=0), acc[m].std(axis=0)
        if best is None or g_std.sum() < best["g_std"].sum():
            best = {"start": start, "g_std": g_std, "a_std": a_std}
    if best is None:
        raise SystemExit("IMU too short to estimate noise")
    g, a = float(best["g_std"].mean()), float(best["a_std"].mean())
    return {
        "window_start_s": round(float(best["start"]), 3),
        "window_s": win_s,
        "gyro_std_rad_s": g,
        "acc_std_m_s2": a,
        "looks_static": bool(g < 0.01 and a < 0.05),  # heuristic: MEMS at rest is well below both
        "gyroscope_noise_density": g * np.sqrt(dt),
        "accelerometer_noise_density": a * np.sqrt(dt),
    }


def stereo_extrinsics(calib: dict) -> np.ndarray:
    T = np.eye(4)  # right camera pose in the left camera frame (T_c1_c2)
    T[:3, :3] = np.array(calib["stereo_transform"]["rotation_matrix_3x3_rowmajor"]).reshape(3, 3)
    T[:3, 3] = calib["stereo_transform"]["translation_m"]
    return T


def write_sensor_yamls(calib: dict, imu_rate: float, noise: dict, mav0: Path) -> np.ndarray:
    """EuRoC sensor.yaml files; returns T_imu_cam0 (EuRoC T_BS of cam0)."""
    T_cam_imu = np.array(calib["imu_to_left_camera"]["matrix_4x4_rowmajor"]).reshape(4, 4)
    T_imu_cam0 = np.linalg.inv(T_cam_imu)  # EuRoC T_BS: sensor pose in the body (IMU) frame
    for name, side, T_bs in (("cam0", "left", T_imu_cam0), ("cam1", "right", T_imu_cam0 @ stereo_extrinsics(calib))):
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
    text += "".join(f"{k}: {v:.6g}\n" for k, v in noise.items() if k in IMU_NOISE)
    (mav0 / "imu0" / "sensor.yaml").write_text(text)
    return T_imu_cam0


def orbslam3_yaml(kind: str, calib: dict, T_imu_cam0: np.ndarray, imu_rate: float, noise: dict,
                  noise_note: str, gray: bool, fast_init: bool, n_features: int | None,
                  th_depth: float | None) -> str:
    """ORB-SLAM3 settings, File.version 1.0. kind: stereo, stereo_inertial or mono_inertial."""
    left, right = calib["left"], calib["right"]
    stereo = kind.startswith("stereo")
    T_c1_c2 = stereo_extrinsics(calib)
    baseline = float(np.linalg.norm(T_c1_c2[:3, 3]))
    if n_features is None:  # EuRoC count scaled by image area, to the nearest 100
        n_features = int(round(EUROC_FEATURES[kind] * left["width"] * left["height"] / EUROC_PIXELS, -2))
    if th_depth is None:  # same minimum disparity as EuRoC's 60 baselines (~7.6 px)
        th_depth = round(left["fx"] / (EUROC_FX / EUROC_TH_DEPTH), 1)

    out = ["%YAML:1.0", f"# Generated by data/tools/zed_to_euroc.py for {kind}, do not edit by hand.",
           'File.version: "1.0"', ""]
    # PinHole, not "Rectified": with Rectified stereo the stock build dereferences a null
    # Camera2 calibration when printing settings (Settings.cc operator<<) and segfaults.
    # With R = I and zero distortion ORB-SLAM3's own stereoRectify maps each pixel to itself.
    out += ["# Images are already rectified: zero distortion, identity stereo rotation",
            'Camera.type: "PinHole"']
    for i, c in ((1, left), (2, right)) if stereo else ((1, left),):
        out += [f"Camera{i}.{k}: {real(c[k])}" for k in ("fx", "fy", "cx", "cy")]
        out += [f"Camera{i}.{k}: 0.0" for k in ("k1", "k2", "p1", "p2")]
    out += [f"Camera.width: {int(left['width'])}", f"Camera.height: {int(left['height'])}",
            f"Camera.fps: {round(calib['fps'])}",
            f"Camera.RGB: 0  # {'ignored, images are grayscale' if gray else 'PNGs load as BGR via cv::imread'}"]
    if stereo:
        out += ["", f"# Close/far threshold in baselines: {th_depth * baseline:.2f} m at b = {baseline:.4f} m",
                f"Stereo.ThDepth: {real(th_depth)}"]
        out += ["# Right camera pose in the left camera frame", cv_matrix("Stereo.T_c1_c2", T_c1_c2)]
    if kind.endswith("inertial"):
        out += ["", "# Left camera pose in the IMU (body) frame = mav0/cam0/sensor.yaml T_BS",
                cv_matrix("IMU.T_b_c1", T_imu_cam0),
                f"# IMU noise: {noise_note}",
                f"IMU.NoiseGyro: {real(noise['gyroscope_noise_density'])}",
                f"IMU.NoiseAcc: {real(noise['accelerometer_noise_density'])}",
                f"IMU.GyroWalk: {real(noise['gyroscope_random_walk'])}",
                f"IMU.AccWalk: {real(noise['accelerometer_random_walk'])}",
                f"IMU.Frequency: {real(round(imu_rate, 1))}",
                "IMU.InsertKFsWhenLost: 1",
                "# 1 skips the stereo-inertial 'not enough acceleration' init check;",
                "# needs tracks/orbslam3/patches/0002 (ignored by a stock build)",
                f"IMU.fastInit: {int(fast_init)}"]
    out += ["", f"ORBextractor.nFeatures: {n_features}", "ORBextractor.scaleFactor: 1.2",
            "ORBextractor.nLevels: 8", "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7", ""]
    out += [f"Viewer.{k}: {v}" for k, v in (
        ("KeyFrameSize", "0.05"), ("KeyFrameLineWidth", "1.0"), ("GraphLineWidth", "0.9"),
        ("PointSize", "2.0"), ("CameraSize", "0.08"), ("CameraLineWidth", "3.0"),
        ("ViewpointX", "0.0"), ("ViewpointY", "-0.7"), ("ViewpointZ", "-1.8"), ("ViewpointF", "500.0"))]
    return "\n".join(out) + "\n"


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


def check_stereo(calib: dict, name: str) -> None:
    t = np.asarray(calib["stereo_transform"]["translation_m"], dtype=float)
    b = float(calib.get("baseline_m", np.linalg.norm(t)))
    if not (t[0] > 0 and abs(t[0] - b) < 1e-3 and np.abs(t[1:]).max() < 1e-3):
        print(f"WARNING {name}: stereo translation {t} is not [+baseline, 0, 0]; "
              "check whether stereo_transform is right-in-left (T_c1_c2)")


def convert(seq: Path, out: Path, args: argparse.Namespace) -> None:
    mav0 = out / "mav0"
    for sub in ("cam0/data", "cam1/data", "imu0", "state_groundtruth_estimate0"):
        (mav0 / sub).mkdir(parents=True, exist_ok=True)

    calib = json.loads((seq / "calibration.json").read_text())
    check_stereo(calib, seq.name)
    stamp_strings = read_stamp_strings(seq / "timestamps.csv")
    stamps_ns = ms_to_ns([float(s) for s in stamp_strings])
    imu_t, gyro, acc, imu_removed = load_imu(seq / "imu.csv")

    # ORB-SLAM3 inertial loaders need imu[0] <= frame and frame < imu[-1] (no bounds checks)
    keep = (stamps_ns >= imu_t[0]) & (stamps_ns < imu_t[-1])
    n_lead = int(np.argmax(keep)) if keep.any() else len(keep)
    n_trail = len(keep) - n_lead - int(keep.sum())
    if not keep.any() or n_lead + n_trail != int((~keep).sum()):
        raise SystemExit(f"{seq.name}: IMU does not cover a contiguous run of frames")
    kept_ns = stamps_ns[keep]
    names = [int(t) if k else None for t, k in zip(stamps_ns, keep)]

    for cam, video in (("cam0", "left.mp4"), ("cam1", "right.mp4")):
        decode_video(seq / video, names, mav0 / cam / "data", args.gray)
        write_frame_index(kept_ns, mav0 / cam / "data.csv")
    (out / "timestamps.txt").write_text("".join(f"{t}\n" for t in kept_ns))

    write_imu(imu_t, gyro, acc, mav0 / "imu0" / "data.csv")
    imu_dt_ms = np.diff(imu_t) * 1e-6
    imu_rate = 1e3 / float(np.median(imu_dt_ms))

    quiet = quiet_window_noise(imu_t, gyro, acc)
    noise = dict(IMU_NOISE)
    if args.imu_noise == "quiet":
        noise["gyroscope_noise_density"] = quiet["gyroscope_noise_density"]
        noise["accelerometer_noise_density"] = quiet["accelerometer_noise_density"]
        note = (f"white noise from quietest {quiet['window_s']} s window at +{quiet['window_start_s']} s "
                f"({'static' if quiet['looks_static'] else 'NOT static, upper bound'}), "
                "random walks PLACEHOLDER (EuRoC ADIS16448)")
    else:
        note = "PLACEHOLDER (EuRoC ADIS16448 values), not measured for the ZED X Mini"
    noise = {k: v * args.imu_noise_scale for k, v in noise.items()}
    if args.imu_noise_scale != 1.0:
        note += f", all scaled x{args.imu_noise_scale:g}"

    T_imu_cam0 = write_sensor_yamls(calib, imu_rate, noise, mav0)
    for kind in ("stereo", "stereo_inertial", "mono_inertial"):
        text = orbslam3_yaml(kind, calib, T_imu_cam0, imu_rate, noise, note, args.gray,
                             args.fast_init, args.n_features, args.th_depth)
        (out / f"orbslam3_{kind}.yaml").write_text(text)

    shutil.copy2(seq / "calibration.json", out / "calibration.json")
    (out / "timestamps_ms.txt").write_text("\n".join(stamp_strings) + "\n")

    gt = seq / "trajectory.txt"
    if gt.exists():
        write_gt(gt, np.linalg.inv(T_imu_cam0), mav0 / "state_groundtruth_estimate0" / "data.csv")
    else:
        shutil.rmtree(mav0 / "state_groundtruth_estimate0")

    info = [
        f"source: {seq.resolve()}",
        f"frames_in: {len(stamps_ns)}",
        f"frames_out: {len(kept_ns)}",
        f"dropped_leading_no_imu: {n_lead}",
        f"dropped_trailing_no_imu: {n_trail}",
        f"imu_samples: {len(imu_t)}",
        f"imu_duplicate_timestamps_removed: {imu_removed}",
        f"imu_rate_hz: {imu_rate:.2f}",
        f"imu_dt_ms: {{min: {imu_dt_ms.min():.3f}, median: {np.median(imu_dt_ms):.3f}, max: {imu_dt_ms.max():.3f}}}",
        f"imu_lead_before_first_frame_s: {(kept_ns[0] - imu_t[0]) * 1e-9:.4f}",
        f"imu_tail_after_last_frame_s: {(imu_t[-1] - kept_ns[-1]) * 1e-9:.4f}",
        "camera_imu_time_offset_s: 0.0  # assumed, not estimated",
        f"imu_noise_used: \"{note}\"",
        "imu_noise_values: {" + ", ".join(f"{k}: {v:.6g}" for k, v in noise.items()) + "}",
        "imu_quiet_window: {" + ", ".join(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}"
                                          for k, v in quiet.items()) + "}",
        f"images: {'gray' if args.gray else 'color (BGR via cv::imread)'}",
        f"ground_truth: {'yes (ZED SDK trajectory, IMU frame)' if gt.exists() else 'none'}",
    ]
    (out / "conversion.yaml").write_text("\n".join(info) + "\n")
    print(f"{seq.name}: {len(kept_ns)}/{len(stamps_ns)} stereo frames (dropped {n_lead} leading, "
          f"{n_trail} trailing without IMU), imu {imu_rate:.1f} Hz dt {imu_dt_ms.min():.1f}-"
          f"{imu_dt_ms.max():.1f} ms, noise {args.imu_noise}, gt {'yes' if gt.exists() else 'none'} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="one sequence folder, or a folder of seq* folders")
    parser.add_argument("out", type=Path, help="target folder (per-sequence subfolders when src is a parent)")
    parser.add_argument("--gray", action="store_true", help="write 8-bit grayscale PNGs like EuRoC (recommended)")
    parser.add_argument("--imu-noise", choices=("placeholder", "quiet"), default="placeholder",
                        help="IMU white-noise source for the yamls (random walks are always placeholders)")
    parser.add_argument("--imu-noise-scale", type=float, default=1.0, help="multiply all four IMU noise terms")
    parser.add_argument("--fast-init", action="store_true", help="IMU.fastInit: 1 (needs the ORB-SLAM3 patch)")
    parser.add_argument("--n-features", type=int, help="ORBextractor.nFeatures (default: EuRoC scaled by area)")
    parser.add_argument("--th-depth", type=float, help="Stereo.ThDepth in baselines (default: EuRoC disparity)")
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
        convert(seq, out, args)


if __name__ == "__main__":
    main()
