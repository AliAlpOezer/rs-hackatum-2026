"""Diagnose left/right stereo consistency of a converted sequence, the way ORB-SLAM3 sees it.

    python data/tools/check_stereo.py ../dataset_euroc/seq01 [--step 30] [--max-offset 3]

Reads <seq>/mav0/cam{0,1}/data/*.png (zed_to_euroc.py output) and, for every
--step-th frame, extracts ORB features in both images and reports:

- stereo match rate: share of left keypoints that find a right keypoint under
  ORB-SLAM3's rule (Frame::ComputeStereoMatches): same row within 2 px x 1.2^octave,
  disparity 0..fx (right x <= left x), Hamming distance < 75. EuRoC-like data
  gives roughly 40-70 %; ~10 % means the pair is not consistently rectified.
- free matches (cross-checked, distance < 50, no geometry) and their median
  vertical offset dy and disparity dx: dy far from 0 = rows misaligned (rectification
  or calibration), dx < 0 = left and right swapped.
- the same rate with the right image taken k frames earlier/later: a peak at
  k != 0 means the two videos are offset by k frames.
- image-level shift from phase correlation (whole-image dx, dy) as a cross-check.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

TH_STEREO = 75  # ORB-SLAM3 (ORBmatcher::TH_HIGH + TH_LOW) / 2
TH_FREE = 50


def load_fx(seq: Path) -> float:
    calib = seq / "calibration.json"
    if calib.exists():
        return float(json.loads(calib.read_text())["left"]["fx"])
    text = (seq / "mav0" / "cam0" / "sensor.yaml").read_text()
    return float(re.search(r"intrinsics:\s*\[([^,\]]+)", text).group(1))


def orb(img: np.ndarray, n: int):
    kps, desc = cv2.ORB_create(nfeatures=n, scaleFactor=1.2, nlevels=8, fastThreshold=20).detectAndCompute(img, None)
    if desc is None:
        return np.zeros((0, 3)), np.zeros((0, 256), np.float32)
    pts = np.array([(k.pt[0], k.pt[1], k.octave) for k in kps])
    return pts, np.unpackbits(desc, axis=1).astype(np.float32)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ (1 - b).T + (1 - a) @ b.T


def stereo_rate(pl, dl, pr, dr, fx: float) -> float:
    """ORB-SLAM3-style: per left keypoint, best right candidate in its row band and disparity range."""
    if len(pl) == 0 or len(pr) == 0:
        return 0.0
    d = hamming(dl, dr)
    band = 2.0 * 1.2 ** pl[:, 2:3]
    disp = pl[:, :1] - pr[:, 0][None, :]
    ok = (np.abs(pl[:, 1:2] - pr[:, 1][None, :]) <= band) & (disp >= 0) & (disp <= fx)
    d = np.where(ok, d, np.inf)
    return float((d.min(axis=1) < TH_STEREO).mean())


def free_matches(pl, dl, pr, dr):
    """Cross-checked descriptor matches without geometry: (dx = xL - xR, dy = yR - yL)."""
    if len(pl) == 0 or len(pr) == 0:
        return np.zeros(0), np.zeros(0)
    d = hamming(dl, dr)
    j = d.argmin(axis=1)
    i_back = d.argmin(axis=0)[j]
    keep = (i_back == np.arange(len(pl))) & (d[np.arange(len(pl)), j] < TH_FREE)
    return pl[keep, 0] - pr[j[keep], 0], pr[j[keep], 1] - pl[keep, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seq", type=Path, help="converted sequence (contains mav0/)")
    parser.add_argument("--step", type=int, default=30, help="check every n-th frame")
    parser.add_argument("--max-offset", type=int, default=3, help="frame offsets to try between left and right")
    parser.add_argument("--features", type=int, default=1900)
    args = parser.parse_args()

    left = sorted((args.seq / "mav0" / "cam0" / "data").glob("*.png"))
    right = sorted((args.seq / "mav0" / "cam1" / "data").glob("*.png"))
    if not left or [p.name for p in left] != [p.name for p in right]:
        raise SystemExit(f"cam0/cam1 frame lists differ or are empty: {len(left)} vs {len(right)}")
    fx = load_fx(args.seq)
    offsets = range(-args.max_offset, args.max_offset + 1)
    frames = range(args.max_offset, len(left) - args.max_offset, args.step)

    rates = {k: [] for k in offsets}
    dxs, dys, n_kp, phase = [], [], [], []
    for i in frames:
        imL = cv2.imread(str(left[i]), cv2.IMREAD_GRAYSCALE)
        pl, dl = orb(imL, args.features)
        n_kp.append(len(pl))
        for k in offsets:
            imR = cv2.imread(str(right[i + k]), cv2.IMREAD_GRAYSCALE)
            pr, dr = orb(imR, args.features)
            rates[k].append(stereo_rate(pl, dl, pr, dr, fx))
            if k == 0:
                dx, dy = free_matches(pl, dl, pr, dr)
                dxs.append(dx)
                dys.append(dy)
                (sx, sy), _ = cv2.phaseCorrelate(imR.astype(np.float32), imL.astype(np.float32))
                phase.append((sx, sy))
    dx, dy = np.concatenate(dxs), np.concatenate(dys)
    mean_rate = {k: float(np.mean(v)) for k, v in rates.items()}
    best_k = max(mean_rate, key=mean_rate.get)
    r0 = mean_rate[0]

    print(f"{args.seq.name}: {len(left)} frames, checked {len(frames)} (every {args.step}), fx {fx:.2f}, "
          f"{np.mean(n_kp):.0f} ORB keypoints per left image")
    print(f"stereo match rate (ORB-SLAM3 rule), same frame: {100 * r0:.1f} % "
          f"(min {100 * min(rates[0]):.1f}, max {100 * max(rates[0]):.1f})")
    print("by right-frame offset k: " + ", ".join(f"{k:+d}: {100 * v:.1f} %" for k, v in mean_rate.items()))
    print(f"free matches: {len(dx)}; median dy {np.median(dy):+.2f} px (|dy|<=2 px: {100 * np.mean(np.abs(dy) <= 2):.0f} %), "
          f"median disparity {np.median(dx):+.1f} px (negative: {100 * np.mean(dx < 0):.0f} %)")
    ph = np.array(phase)
    print(f"phase correlation right->left: median dx {np.median(ph[:, 0]):+.1f} px, dy {np.median(ph[:, 1]):+.2f} px")

    findings = []
    swapped = np.mean(dx < 0) > 0.5
    if swapped:  # with a swap no offset matches properly, so the offset scan says nothing
        findings.append("disparities are mostly negative: left and right are swapped")
    elif best_k != 0 and mean_rate[best_k] > 1.5 * r0:
        findings.append(f"left frame i matches right frame i{best_k:+d} best: the videos are offset by "
                        f"{abs(best_k)} frame(s); re-align before converting")
    if abs(np.median(dy)) > 1.0:
        findings.append(f"rows are misaligned by {np.median(dy):+.1f} px: not rectified with these intrinsics "
                        "(check cy of both cameras and whether the videos are the rectified ones)")
    if not findings and r0 < 0.25:
        findings.append("low match rate without a clear geometric cause: check texture/blur, exposure, "
                        "or whether both videos come from the same recording")
    print("verdict: " + ("; ".join(findings) if findings else "consistent rectified stereo pair"))


if __name__ == "__main__":
    main()
