"""Loosely coupled stereo + IMU fusion: error-state Kalman filter with an RTS smoother.

    uv run --with torch python tracks/imu_fusion/eskf.py <raw_rs_seq> <stereo_tum> <out_tum>

<raw_rs_seq> is an R&S recording folder (imu.csv, calibration.json); <stereo_tum> a
camera trajectory (TUM, T_world_cam, seconds), e.g. Jonas's ORB-SLAM3 stereo output.
IMU at ~200 Hz propagates the state; every stereo pose is a measurement. Offline, so a
Rauch-Tung-Striebel pass smooths with future data. Written in torch (float64) so the
noise parameters can later be learned by gradient descent (step 2).

State: p, v, R (IMU body in world), gyro bias, accel bias; 15-D error state with
right-multiplied rotation error. The camera-IMU time offset is estimated per sequence
by cross-correlating gyro and stereo rotation rates, then removed from the IMU stamps.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


# ---------- SO(3) ----------
def skew(w):
    o = torch.zeros(())
    return torch.stack([torch.stack([o, -w[2], w[1]]), torch.stack([w[2], o, -w[0]]),
                        torch.stack([-w[1], w[0], o])])


def so3_exp(w):
    th = torch.linalg.norm(w)
    K = skew(w)
    if th < 1e-9:
        return torch.eye(3) + K
    return torch.eye(3) + torch.sin(th) / th * K + (1 - torch.cos(th)) / th**2 * K @ K


def so3_log(R):
    c = ((torch.trace(R) - 1) / 2).clamp(-1 + 1e-12, 1 - 1e-12)
    th = torch.arccos(c)
    v = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / 2 if th < 1e-9 else th / (2 * torch.sin(th)) * v


def quat_to_R(q):  # x y z w
    x, y, z, w = q
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R.detach().numpy()).as_quat()


# ---------- data ----------
def load_imu(seq: Path):
    d = np.genfromtxt(seq / "imu.csv", delimiter=",", names=True)
    t = d["timestamp_ms"] / 1e3
    acc = np.c_[d["ax"], d["ay"], d["az"]]
    gyr = np.deg2rad(np.c_[d["gx"], d["gy"], d["gz"]])
    return t, acc, gyr


def load_T_b_c(seq: Path) -> np.ndarray:
    T_c_b = np.array(json.loads((seq / "calibration.json").read_text())
                     ["imu_to_left_camera"]["matrix_4x4_rowmajor"]).reshape(4, 4)
    return np.linalg.inv(T_c_b)


def rot_rate(t, quats):
    from scipy.spatial.transform import Rotation
    r = Rotation.from_quat(quats)
    dt = np.diff(t)
    ok = dt > 1e-4
    return ((t[1:] + t[:-1]) / 2)[ok], (r[1:] * r[:-1].inv()).magnitude()[ok] / dt[ok]


def estimate_offset(t_imu, gyr, t_cam, q_cam, rng=0.05, step=0.0005) -> float:
    """Seconds to subtract from IMU stamps so the gyro norm lines up with the stereo rotation rate."""
    ts, ws = rot_rate(t_cam, q_cam)
    gn = np.linalg.norm(gyr, axis=1)
    m = (ts > t_imu[0] + rng) & (ts < t_imu[-1] - rng)
    scores = [(np.corrcoef(np.interp(ts[m] + o, t_imu, gn), ws[m])[0, 1], o)
              for o in np.arange(-rng, rng + 1e-9, step)]
    return max(scores)[1]


# ---------- filter ----------
@dataclass
class Noise:
    gyro: float = 2e-3       # rad/s/sqrt(Hz)
    accel: float = 2e-2      # m/s^2/sqrt(Hz)
    gyro_walk: float = 1e-4  # rad/s^2/sqrt(Hz)
    accel_walk: float = 1e-3  # m/s^3/sqrt(Hz)
    pos: float = 1.5e-3      # m, stereo position measurement
    rot: float = 3.5e-3      # rad, stereo rotation measurement


def run(t_imu, acc, gyr, t_cam, T_wb_meas, g_w, noise: Noise):
    """Forward ESKF over the merged IMU/camera timeline, then RTS. Returns smoothed (p, R) at camera times."""
    I3, Z3 = torch.eye(3), torch.zeros(3, 3)
    p = T_wb_meas[0][:3, 3].clone()
    R = T_wb_meas[0][:3, :3].clone()
    v = (T_wb_meas[1][:3, 3] - T_wb_meas[0][:3, 3]) / (t_cam[1] - t_cam[0])
    bg, ba = torch.zeros(3), torch.zeros(3)
    P = torch.diag(torch.tensor([1e-6] * 3 + [1e-2] * 3 + [1e-5] * 3 + [1e-4] * 3 + [1e-2] * 3))
    def d3(x):
        return torch.as_tensor(x).reshape(1).expand(3) ** 2
    Qc = torch.diag(torch.cat([d3(noise.accel), d3(noise.gyro), d3(noise.gyro_walk), d3(noise.accel_walk)]))
    Rm = torch.diag(torch.cat([d3(noise.pos), d3(noise.rot)]))
    H = torch.zeros(6, 15)
    H[:3, :3] = I3
    H[3:, 6:9] = I3

    hist = []  # per step: (state_pred, P_pred, state_post, P_post, Phi, is_cam_index)
    k, t = 0, t_cam[0]
    while k < len(t_imu) and t_imu[k] <= t:
        k += 1
    a_m, w_m = acc[max(k - 1, 0)], gyr[max(k - 1, 0)]

    def propagate(dt, a_m, w_m):
        nonlocal p, v, R, P
        a = a_m - ba
        w = w_m - bg
        F = torch.zeros(15, 15)
        F[0:3, 3:6] = I3
        F[3:6, 6:9] = -R @ skew(a)
        F[3:6, 12:15] = -R
        F[6:9, 6:9] = -skew(w)
        F[6:9, 9:12] = -I3
        G = torch.zeros(15, 12)
        G[3:6, 0:3] = -R
        G[6:9, 3:6] = -I3
        G[9:12, 6:9] = I3
        G[12:15, 9:12] = I3
        Phi = torch.eye(15) + F * dt
        aw = R @ a + g_w
        p = p + v * dt + 0.5 * aw * dt**2
        v = v + aw * dt
        R = R @ so3_exp(w * dt)
        P = Phi @ P @ Phi.T + G @ Qc @ G.T * dt
        return Phi

    def snapshot():
        return (p.clone(), v.clone(), R.clone(), bg.clone(), ba.clone())

    for j in range(len(t_cam)):
        tf = t_cam[j]
        Phi_acc = torch.eye(15)
        if j > 0:
            while k < len(t_imu) and t_imu[k] < tf:
                Phi_acc = propagate(t_imu[k] - t, a_m, w_m) @ Phi_acc
                t = t_imu[k]
                a_m, w_m = acc[k], gyr[k]
                k += 1
            if tf > t:
                Phi_acc = propagate(tf - t, a_m, w_m) @ Phi_acc
                t = tf
        x_pred, P_pred = snapshot(), P.clone()
        # stereo pose update
        Tm = T_wb_meas[j]
        r = torch.cat([Tm[:3, 3] - p, so3_log(R.T @ Tm[:3, :3])])
        S = H @ P @ H.T + Rm
        K = P @ H.T @ torch.linalg.inv(S)
        dx = K @ r
        P = (torch.eye(15) - K @ H) @ P
        p, v, R = p + dx[0:3], v + dx[3:6], R @ so3_exp(dx[6:9])
        bg, ba = bg + dx[9:12], ba + dx[12:15]
        hist.append((x_pred, P_pred, snapshot(), P.clone(), Phi_acc))

    # RTS over camera steps (Phi_acc spans the IMU steps between two camera times)
    xs, Ps = hist[-1][2], hist[-1][3]
    out = [xs]
    for j in range(len(hist) - 2, -1, -1):
        x_post, P_post = hist[j][2], hist[j][3]
        x_pred_n, P_pred_n, Phi_n = hist[j + 1][0], hist[j + 1][1], hist[j + 1][4]
        C = P_post @ Phi_n.T @ torch.linalg.inv(P_pred_n)
        d = torch.cat([xs[0] - x_pred_n[0], xs[1] - x_pred_n[1], so3_log(x_pred_n[2].T @ xs[2]),
                       xs[3] - x_pred_n[3], xs[4] - x_pred_n[4]])
        dx = C @ d
        xs = (x_post[0] + dx[0:3], x_post[1] + dx[3:6], x_post[2] @ so3_exp(dx[6:9]),
              x_post[3] + dx[9:12], x_post[4] + dx[12:15])
        Ps = P_post + C @ (Ps - P_pred_n) @ C.T
        out.append(xs)
    out.reverse()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seq", type=Path, help="raw R&S sequence folder (imu.csv, calibration.json)")
    ap.add_argument("stereo", type=Path, help="stereo camera trajectory, TUM T_world_cam")
    ap.add_argument("out", type=Path, help="output TUM T_world_cam")
    ap.add_argument("--no-offset", action="store_true", help="skip the camera-IMU time offset estimate")
    for f, val in Noise().__dict__.items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=float, default=val)
    a = ap.parse_args()
    noise = Noise(**{f: getattr(a, f) for f in Noise().__dict__})

    t_imu, acc, gyr = load_imu(a.seq)
    cam = np.loadtxt(a.stereo)
    t_cam, q_cam = cam[:, 0], cam[:, 4:8]
    offset = 0.0 if a.no_offset else estimate_offset(t_imu, gyr, t_cam, q_cam)
    t_imu = t_imu - offset

    T_b_c = torch.tensor(load_T_b_c(a.seq))
    T_c_b = torch.linalg.inv(T_b_c)
    T_wb = []
    for row in cam:
        T = torch.eye(4)
        T[:3, :3] = quat_to_R(row[4:8])
        T[:3, 3] = torch.tensor(row[1:4])
        T_wb.append(T @ T_c_b)
    # gravity in the stereo world: mean linear acceleration over a handheld clip is ~0
    Rb = np.array([T[:3, :3].numpy() for T in T_wb])
    idx = np.clip(np.searchsorted(t_cam, t_imu), 0, len(t_cam) - 1)
    f_w = np.einsum("nij,nj->ni", Rb[idx], acc)
    g_w = torch.tensor(-f_w.mean(0))

    states = run(torch.tensor(t_imu).tolist(), torch.tensor(acc), torch.tensor(gyr),
                 t_cam.tolist(), T_wb, g_w, noise)
    rows = []
    for tc, (p, _, R, _, _) in zip(t_cam, states):
        T = torch.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        Twc = T @ T_b_c
        rows.append([tc, *Twc[:3, 3].tolist(), *R_to_quat(Twc[:3, :3])])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out, np.array(rows), fmt="%.9f")
    print(f"{a.seq.name}: offset {offset * 1e3:+.1f} ms, |g| {float(torch.linalg.norm(g_w)):.3f}, "
          f"{len(rows)} poses -> {a.out}")


if __name__ == "__main__":
    main()
