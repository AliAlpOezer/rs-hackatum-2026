"""Write a synthetic sequence in the raw R&S ZED X Mini format, to test the pipeline without the dataset.

    python data/tools/synth_zed.py /tmp/synth/seq01 [--seconds 20] [--motion 1.0]

A rectified stereo camera moves handheld-style inside a textured room. Output
matches dataset_extracted/seqNN: left.mp4, right.mp4, timestamps.csv (ms),
imu.csv (timestamp_ms,ax,ay,az,gx,gy,gz; gyro deg/s), calibration.json and
trajectory.txt (left camera T_world_cam, ms). Like the real data, the IMU is
irregular (~5 ms, with gaps), starts 0.4 s before the first frame and ends
1.3 ms before the last one. The IMU is mounted rotated and offset from the left
camera, so a wrong transform direction shows up as a bad trajectory.
World frame is gravity-aligned, z up. Needs ffmpeg (libx264).
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

W, H, FX, CX, CY, FPS, BASELINE = 960, 600, 370.37, 480.0, 300.0, 30, 0.0499582
ROOM = np.array([[-2.5, 2.5], [-1.5, 3.0], [-1.2, 1.5]])  # x, y, z walls [m], camera starts at 0
TEXEL = 0.004  # texture resolution [m per texel]
G_W = np.array([0.0, 0.0, -9.81])
R_WC0 = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # cam x->world x, y->-z (down), z->+y
# IMU in the left-camera frame: x forward, y left, z up, tilted 2 deg, offset a few cm
R_CI = Rotation.from_euler("xyz", [1.0, -2.0, 1.5], degrees=True).as_matrix() @ np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
T_CI = np.array([0.023, 0.002, -0.015])


def camera_pose(tau: np.ndarray, motion: float) -> tuple[Rotation, np.ndarray]:
    """Left camera pose T_world_cam at times tau [s] from the first frame (vectorized)."""
    tau = np.atleast_1d(tau)
    s = lambda a, f, ph=0.0: a * np.sin(2 * np.pi * f * tau + ph)
    s0 = lambda a, f, ph=0.0: a * np.sin(ph)  # same term at tau = 0
    terms = [((0.30, 0.31), (0.04, 1.7, 1.0)), ((0.20, 0.23, 0.5), (0.03, 1.3)),
             ((0.12, 0.41, 2.0), (0.03, 1.9, 0.3))]
    p = np.column_stack([s(*a) + s(*b) - s0(*a) - s0(*b) for a, b in terms]) * motion  # p(0) = 0
    euler = np.column_stack([
        s(0.35, 0.17) + s(0.02, 2.1),          # yaw about world z
        s(0.15, 0.29, 1.0) + s(0.015, 1.6),    # pitch
        s(0.08, 0.37, 2.0),                    # roll
    ]) * motion
    R = Rotation.from_euler("zxy", euler) * Rotation.from_matrix(R_WC0)
    return R, p


def make_texture(rng: np.random.Generator, w_m: float, h_m: float) -> np.ndarray:
    w, h = int(w_m / TEXEL) + 2, int(h_m / TEXEL) + 2
    tex = np.zeros((h, w), np.float32)
    for scale, amp in ((64, 70), (16, 50), (4, 30)):
        noise = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
        tex += amp * cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(tex, 0, 255).astype(np.uint8)
    for _ in range(int(w_m * h_m * 120)):  # blobs and boxes give ORB corners at several scales
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(3, 25))
        val = int(rng.integers(0, 256))
        if rng.random() < 0.5:
            cv2.rectangle(img, c, (c[0] + r, c[1] + int(rng.integers(3, 25))), val, -1)
        else:
            cv2.circle(img, c, r, val, -1)
    return img


class Room:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.planes = []  # (axis, value, in-plane axes, texture)
        for axis in range(3):
            u, v = [a for a in range(3) if a != axis]
            for value in ROOM[axis]:
                tex = make_texture(rng, np.ptp(ROOM[u]), np.ptp(ROOM[v]))
                self.planes.append((axis, value, u, v, tex))
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        self.rays_c = np.stack([(xs - CX) / FX, (ys - CY) / FX, np.ones_like(xs)], -1).reshape(-1, 3)

    def render(self, R_wc: np.ndarray, p_wc: np.ndarray) -> np.ndarray:
        d = self.rays_c @ R_wc.T
        best_t = np.full(len(d), np.inf)
        out = np.zeros(len(d), np.float32)
        for axis, value, u, v, tex in self.planes:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (value - p_wc[axis]) / d[:, axis]
            hit = (t > 0) & (t < best_t)
            if not hit.any():
                continue
            pts = p_wc + t[hit, None] * d[hit]
            mx = np.clip((pts[:, u] - ROOM[u, 0]) / TEXEL, 0, tex.shape[1] - 1.001)
            my = np.clip((pts[:, v] - ROOM[v, 0]) / TEXEL, 0, tex.shape[0] - 1.001)
            x0, y0 = mx.astype(int), my.astype(int)
            fx, fy = mx - x0, my - y0  # bilinear
            out[hit] = ((1 - fy) * ((1 - fx) * tex[y0, x0] + fx * tex[y0, x0 + 1])
                        + fy * ((1 - fx) * tex[y0 + 1, x0] + fx * tex[y0 + 1, x0 + 1]))
            best_t[hit] = t[hit]
        return out.reshape(H, W).astype(np.uint8)


def encode(frames_cmd_out: Path) -> subprocess.Popen:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(frames_cmd_out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def imu_samples(tau_cam: np.ndarray, motion: float, rng: np.random.Generator) -> np.ndarray:
    """Rows tau, ax, ay, az [m/s^2], gx, gy, gz [deg/s] in the IMU frame (accel first, as in R&S)."""
    t, ts = tau_cam[0] - 0.4, []
    while t < tau_cam[-1] - 0.0013:
        ts.append(t)
        t += 0.00504 + rng.uniform(-0.0005, 0.0005) + (rng.uniform(0.005, 0.02) if rng.random() < 0.01 else 0)
    ts = np.array(ts)
    ts[-1] = tau_cam[-1] - 0.0013  # last IMU sample just before the last frame, like the real data

    def body(tt):
        R_wc, p_wc = camera_pose(tt, motion)
        return R_wc * Rotation.from_matrix(R_CI), p_wc + R_wc.apply(T_CI)

    h = 1e-3
    R_m, p_m = body(ts - h)
    R_0, p_0 = body(ts)
    R_p, p_p = body(ts + h)
    gyro = (R_m.inv() * R_p).as_rotvec() / (2 * h)  # body-frame angular rate
    acc_w = (p_p - 2 * p_0 + p_m) / h ** 2
    acc = R_0.inv().apply(acc_w - G_W)  # specific force in the IMU frame
    gyro += np.array([0.002, -0.001, 0.0015]) + rng.normal(0, 1.7e-4 * np.sqrt(200), gyro.shape)
    acc += np.array([0.02, -0.015, 0.01]) + rng.normal(0, 2.0e-3 * np.sqrt(200), acc.shape)
    return np.column_stack([ts, acc, np.rad2deg(gyro)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="target folder, e.g. /tmp/synth/seq01")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--motion", type=float, default=1.0, help="scale of translation and rotation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1)
    t0_ms = 749479.912
    n = int(args.seconds * FPS) + 1
    tau_cam = np.arange(n) / FPS + rng.uniform(-2e-4, 2e-4, n)
    tau_cam[0] = 0.0
    stamps = [f"{t0_ms + 1e3 * t:.3f}" for t in tau_cam]
    tau_cam = np.array([float(s) for s in stamps]) * 1e-3 - t0_ms * 1e-3  # use the rounded times

    room = Room(args.seed)
    R_wc, p_wc = camera_pose(tau_cam, args.motion)
    left, right = encode(args.out / "left.mp4"), encode(args.out / "right.mp4")
    for i in range(n):
        Rm = R_wc[i].as_matrix()
        left.stdin.write(room.render(Rm, p_wc[i]).tobytes())
        right.stdin.write(room.render(Rm, p_wc[i] + Rm[:, 0] * BASELINE).tobytes())
    for proc in (left, right):
        proc.stdin.close()
        if proc.wait():
            raise SystemExit("ffmpeg failed")

    (args.out / "timestamps.csv").write_text("timestamp_ms\n" + "\n".join(stamps) + "\n")
    imu = imu_samples(tau_cam, args.motion, rng)
    with (args.out / "imu.csv").open("w") as f:
        f.write("timestamp_ms,ax,ay,az,gx,gy,gz\n")
        for row in imu:
            f.write(f"{t0_ms + 1e3 * row[0]:.3f}," + ",".join(f"{v:.6f}" for v in row[1:]) + "\n")
    traj = np.column_stack([np.array([float(s) for s in stamps]), p_wc, R_wc.as_quat()])
    np.savetxt(args.out / "trajectory.txt", traj, fmt="%.6f")

    T_ci = np.eye(4)
    T_ci[:3, :3], T_ci[:3, 3] = R_CI, T_CI
    cam = {"fx": FX, "fy": FX, "cx": CX, "cy": CY, "width": W, "height": H}
    calib = {
        "camera_model": "ZED X Mini (synthetic, data/tools/synth_zed.py)",
        "images": "rectified",
        "sdk_version": "synthetic",
        "fps": FPS,
        "left": cam,
        "right": dict(cam),
        "baseline_m": BASELINE,
        "stereo_transform": {"translation_m": [BASELINE, 0.0, 0.0],
                             "rotation_matrix_3x3_rowmajor": np.eye(3).ravel().tolist()},
        "imu_to_left_camera": {"matrix_4x4_rowmajor": T_ci.ravel().tolist()},
        "imu_units": {"gyro": "deg/s", "accel": "m/s^2"},
    }
    (args.out / "calibration.json").write_text(json.dumps(calib, indent=2))
    print(f"{args.out}: {n} stereo frames, {len(imu)} IMU samples, {args.seconds:g} s")


if __name__ == "__main__":
    main()
