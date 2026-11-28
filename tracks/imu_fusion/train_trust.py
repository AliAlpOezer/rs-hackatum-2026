"""Step 3: learn per-frame trust in the stereo measurement, cross-validated by sequence.

    uv run --with torch python tracks/imu_fusion/train_trust.py <raw_root> <stereo_dir> <gt_dir> <out_dir>

A small MLP maps per-frame features that need no ground truth (stereo speed and rotation
rate, gyro rate, stereo-vs-gyro rotation-rate disagreement, acceleration beyond gravity)
to multipliers on the stereo position / rotation variances. The last layer starts at zero,
so training starts from the step 1 filter (hand-set noise, stereo position 4 mm).
Same folds, windows and loss as train_noise.py; held-out outputs -> eskf_trust_seqNN.txt.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from eskf import Noise, R_to_quat, rot_rate, run
from train_noise import FOLDS, SEQS, cam_positions, prepare

BASE = Noise(pos=4e-3)


def features(d) -> torch.Tensor:
    t, cam = d["t_cam"], d["cam"]
    _, w_st = rot_rate(t, cam[:, 4:8])
    w_st = np.r_[w_st[:1], w_st] if len(w_st) == len(t) - 1 else np.interp(t, t[1:], w_st)
    v_st = np.r_[0, np.linalg.norm(np.diff(cam[:, 1:4], axis=0), axis=1) / np.diff(t)]
    ti = np.asarray(d["t_imu"])
    gn = np.linalg.norm(d["gyr"].numpy(), axis=1)
    an = np.abs(np.linalg.norm(d["acc"].numpy(), axis=1) - 9.81)
    w_g = np.interp(t, ti, gn)
    a_x = np.interp(t, ti, an)
    f = np.c_[v_st / 0.2, w_st / 0.5, w_g / 0.5, np.abs(w_st - w_g) / 0.1, a_x / 0.5]
    return torch.tensor(f)


class Trust(torch.nn.Module):
    def __init__(self, n_in=5, hidden=16):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 2))
        torch.nn.init.zeros_(self.net[2].weight)
        torch.nn.init.zeros_(self.net[2].bias)

    def forward(self, f):
        return torch.exp(2.0 * torch.tanh(self.net(f)))  # variance multipliers in [e^-2, e^2]


def filt(d, model, sl):
    rs = model(d["feat"][sl])
    return run(d["t_imu"].tolist(), d["acc"], d["gyr"], d["t_cam"][sl].tolist(), d["T_wb"][sl], d["g_w"], BASE, rs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for n in ("raw_root", "stereo_dir", "gt_dir", "out_dir"):
        ap.add_argument(n, type=Path)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.02)
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    data = {s: prepare(a.raw_root / s, a.stereo_dir / f"orbslam3_stereo_{s}.txt", a.gt_dir / f"gt_{s}.txt")
            for s in SEQS}
    for d in data.values():
        d["feat"] = features(d)
    log = []
    for held in FOLDS:
        train = [s for s in SEQS if s not in held]
        model = Trust()
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        for it in range(a.iters):
            s = random.choice(train)
            d = data[s]
            j0 = random.randrange(0, len(d["t_cam"]) - a.window)
            sl = slice(j0, j0 + a.window)
            opt.zero_grad()
            loss = ((cam_positions(filt(d, model, sl), d["T_b_c"]) - d["gt"][sl]) * 1e3).pow(2).sum(1).mean()
            loss.backward()
            opt.step()
            print(f"fold {held} it {it} {s} rmse {loss.item() ** 0.5:.3f} mm", flush=True)
        with torch.no_grad():
            for s in held:
                d = data[s]
                full = slice(0, len(d["t_cam"]))
                rs = model(d["feat"])
                log.append({"seq": s, "pos_mult_mean": float(rs[:, 0].mean()), "pos_mult_max": float(rs[:, 0].max()),
                            "rot_mult_mean": float(rs[:, 1].mean())})
                rows = []
                for tc, (p, _, R, _, _) in zip(d["t_cam"], filt(d, model, full)):
                    T = torch.eye(4)
                    T[:3, :3], T[:3, 3] = R, p
                    Twc = T @ d["T_b_c"]
                    rows.append([tc, *Twc[:3, 3].tolist(), *R_to_quat(Twc[:3, :3])])
                np.savetxt(a.out_dir / f"eskf_trust_{s}.txt", np.array(rows), fmt="%.9f")
        (a.out_dir / "trust_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
