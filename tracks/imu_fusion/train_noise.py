"""Step 2: learn the six ESKF noise parameters by gradient descent, cross-validated by sequence.

    uv run --with torch python tracks/imu_fusion/train_noise.py <raw_root> <stereo_dir> <gt_dir> <out_dir>

<raw_root>/seqNN: R&S recordings; <stereo_dir>/orbslam3_stereo_seqNN.txt: stereo TUM;
<gt_dir>/gt_seqNN.txt: ground truth TUM (seconds). Folds hold out two sequences each;
parameters are trained on random 4 s windows of the other eight (loss: mean squared
camera-position error of the smoothed trajectory vs rigidly aligned ground truth), then
the held-out sequences are filtered in full and written to <out_dir>/eskf_learned_seqNN.txt.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from eskf import Noise, estimate_offset, load_T_b_c, load_imu, quat_to_R, run, R_to_quat

SEQS = [f"seq{i:02d}" for i in range(1, 11)]
FOLDS = [SEQS[i:i + 2] for i in range(0, 10, 2)]
NAMES = list(Noise().__dict__)


def umeyama_se3(src, dst):
    """R, t with dst ~ R src + t (no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, md - R @ ms


def prepare(raw: Path, stereo: Path, gt: Path):
    t_imu, acc, gyr = load_imu(raw)
    cam = np.loadtxt(stereo)
    t_cam = cam[:, 0]
    offset = estimate_offset(t_imu, gyr, t_cam, cam[:, 4:8])
    t_imu = t_imu - offset
    T_b_c = torch.tensor(load_T_b_c(raw))
    T_c_b = torch.linalg.inv(T_b_c)
    T_wb = []
    for row in cam:
        T = torch.eye(4)
        T[:3, :3], T[:3, 3] = quat_to_R(row[4:8]), torch.tensor(row[1:4])
        T_wb.append(T @ T_c_b)
    Rb = np.array([T[:3, :3].numpy() for T in T_wb])
    idx = np.clip(np.searchsorted(t_cam, t_imu), 0, len(t_cam) - 1)
    g_w = torch.tensor(-np.einsum("nij,nj->ni", Rb[idx], acc).mean(0))
    g = np.loadtxt(gt)
    gp = np.c_[[np.interp(t_cam, g[:, 0], g[:, k]) for k in (1, 2, 3)]].T
    Ra, ta = umeyama_se3(gp, cam[:, 1:4])
    return dict(t_imu=t_imu, acc=torch.tensor(acc), gyr=torch.tensor(gyr), t_cam=t_cam, T_wb=T_wb,
                g_w=g_w, T_b_c=T_b_c, gt=torch.tensor(gp @ Ra.T + ta), cam=cam, offset=offset)


def cam_positions(states, T_b_c):
    return torch.stack([p + R @ T_b_c[:3, 3] for p, _, R, _, _ in states])


def window_loss(d, log_params, j0, n):
    noise = Noise(**dict(zip(NAMES, torch.exp(log_params))))
    sl = slice(j0, j0 + n)
    states = run(d["t_imu"].tolist(), d["acc"], d["gyr"], d["t_cam"][sl].tolist(), d["T_wb"][sl], d["g_w"], noise)
    return ((cam_positions(states, d["T_b_c"]) - d["gt"][sl]) * 1e3).pow(2).sum(1).mean()  # mm^2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("stereo_dir", type=Path)
    ap.add_argument("gt_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--window", type=int, default=120, help="frames per training window (4 s)")
    ap.add_argument("--lr", type=float, default=0.15)
    a = ap.parse_args()
    random.seed(0)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    data = {s: prepare(a.raw_root / s, a.stereo_dir / f"orbslam3_stereo_{s}.txt", a.gt_dir / f"gt_{s}.txt")
            for s in SEQS}
    log = []
    for held in FOLDS:
        train = [s for s in SEQS if s not in held]
        lp = torch.log(torch.tensor([getattr(Noise(), n) for n in NAMES])).requires_grad_()
        opt = torch.optim.Adam([lp], lr=a.lr)
        for it in range(a.iters):
            s = random.choice(train)
            j0 = random.randrange(0, len(data[s]["t_cam"]) - a.window)
            opt.zero_grad()
            loss = window_loss(data[s], lp, j0, a.window)
            loss.backward()
            opt.step()
            print(f"fold {held} it {it} {s} rmse {loss.item() ** 0.5:.3f} mm", flush=True)
        learned = dict(zip(NAMES, torch.exp(lp).tolist()))
        log.append({"held_out": held, "params": learned})
        print("learned", json.dumps(learned), flush=True)
        with torch.no_grad():
            for s in held:
                d = data[s]
                states = run(d["t_imu"].tolist(), d["acc"], d["gyr"], d["t_cam"].tolist(), d["T_wb"], d["g_w"],
                             Noise(**learned))
                rows = []
                for tc, (p, _, R, _, _) in zip(d["t_cam"], states):
                    T = torch.eye(4)
                    T[:3, :3], T[:3, 3] = R, p
                    Twc = T @ d["T_b_c"]
                    rows.append([tc, *Twc[:3, 3].tolist(), *R_to_quat(Twc[:3, :3])])
                np.savetxt(a.out_dir / f"eskf_learned_{s}.txt", np.array(rows), fmt="%.9f")
        (a.out_dir / "learned_params.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
