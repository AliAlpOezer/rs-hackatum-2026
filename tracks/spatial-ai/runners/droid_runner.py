"""Run DROID-SLAM (stereo or mono) on a sequence in the shared layout and write a TUM trajectory.

    python tracks/spatial-ai/runners/droid_runner.py data/seq01 --droid-dir /content/DROID-SLAM \
        --out results/seq01/droid_stereo.txt

Needs a CUDA GPU and DROID-SLAM installed (see notebooks/droid_rs_colab.ipynb). Reads
frames/ and, with --stereo (default), frames_right/; both must be rectified, as the R&S
ZED data is. DROID-SLAM hard-codes a 0.1 m stereo baseline (src/droid_kernels.cu), so
stereo positions are rescaled by baseline_m / 0.1 from calib.yaml, the same correction
its EuRoC script applies (x1.10 for 0.11 m). Mono output keeps DROID's arbitrary scale.
Output poses are camera-to-world, timestamps in seconds; a <out>.json sidecar records
runtime and peak GPU memory.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

DROID_BASELINE_M = 0.1
# Defaults of DROID-SLAM's evaluation_scripts/test_euroc.py
DROID_DEFAULTS = {
    "buffer": 512, "beta": 0.3, "filter_thresh": 2.4, "warmup": 15, "keyframe_thresh": 3.0,
    "frontend_thresh": 17.5, "frontend_window": 20, "frontend_radius": 2, "frontend_nms": 1,
    "backend_thresh": 24.0, "backend_radius": 2, "backend_nms": 2, "upsample": False,
    "asynchronous": False, "frontend_device": "cuda", "backend_device": "cuda", "disable_vis": True,
}


def load_calib(seq: Path) -> tuple[list[float], float]:
    calib = yaml.safe_load((seq / "calib.yaml").read_text())
    cam = calib["camera"]
    if any(cam.get("distortion", []) or []):
        raise SystemExit("frames must be rectified (zero distortion); undistort first")
    return list(cam["intrinsics"]), float(calib["baseline_m"])


def list_frames(seq: Path, stereo: bool) -> list[tuple[float, Path, Path | None]]:
    left = sorted((seq / "frames").glob("*.png"))
    if not left:
        raise SystemExit(f"no frames in {seq / 'frames'}")
    out = []
    for p in left:
        r = seq / "frames_right" / p.name
        if stereo and not r.is_file():
            raise SystemExit(f"missing right frame {r}")
        out.append((int(p.stem) * 1e-9, p, r if stereo else None))
    return out


def stream(frames, intrinsics, size, stride=1):
    """(index, images [n_cam, 3, H, W] float32, intrinsics [4]) as DROID-SLAM's demos build them."""
    import torch
    h, w = size
    for i in range(0, len(frames), stride):
        _, left, right = frames[i]
        imgs = []
        for path in (left, right):
            if path is None:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"cannot read {path}")
            h0, w0 = img.shape[:2]
            imgs.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
        image = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()
        fx, fy, cx, cy = intrinsics
        k = torch.tensor([fx * w / w0, fy * h / h0, cx * w / w0, cy * h / h0])
        yield i, image, k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seq", type=Path, help="data/<seq> folder")
    parser.add_argument("--droid-dir", type=Path, required=True, help="DROID-SLAM checkout (droid.pth inside)")
    parser.add_argument("--weights", type=Path, help="default: <droid-dir>/droid.pth")
    parser.add_argument("--out", type=Path, required=True, help="results/<seq>/<method>.txt")
    parser.add_argument("--mono", action="store_true", help="left camera only (scale is arbitrary)")
    parser.add_argument("--stride", type=int, default=2, help="tracked every n-th frame; the rest are filled in")
    parser.add_argument("--image-size", type=int, nargs=2, default=[320, 512], metavar=("H", "W"),
                        help="network input, multiples of 8 (960x600 R&S -> 512x320 keeps the aspect)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override DROID args, e.g. filter_thresh=1.75 upsample=True")
    args = parser.parse_args()

    import torch
    sys.path.insert(0, str(args.droid_dir / "droid_slam"))
    from droid import Droid

    stereo = not args.mono
    intrinsics, baseline = load_calib(args.seq)
    frames = list_frames(args.seq, stereo)
    opts = dict(DROID_DEFAULTS)
    for kv in args.set:
        key, value = kv.split("=", 1)
        opts[key] = type(opts[key])(value == "True" if isinstance(opts[key], bool) else value)
    opts.update(weights=str(args.weights or args.droid_dir / "droid.pth"), stereo=stereo,
                image_size=list(args.image_size))
    opts["buffer"] = max(opts["buffer"], len(frames) // args.stride + 16)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    droid = Droid(SimpleNamespace(**opts))
    for i, image, k in stream(frames, intrinsics, args.image_size, args.stride):
        droid.track(i, image, intrinsics=k)
    traj = droid.terminate(stream(frames, intrinsics, args.image_size))  # N x 7, camera-to-world
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    traj = np.asarray(traj, dtype=float)
    if len(traj) != len(frames):
        raise SystemExit(f"DROID returned {len(traj)} poses for {len(frames)} frames")
    scale = baseline / DROID_BASELINE_M if stereo else 1.0
    traj[:, :3] *= scale
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tstamps = np.array([t for t, _, _ in frames])
    np.savetxt(args.out, np.column_stack([tstamps, traj]), fmt="%.9f",
               header="timestamp_s tx ty tz qx qy qz qw")

    stats = {
        "method": args.out.stem,
        "sequence": args.seq.name,
        "frames": len(frames),
        "stride": args.stride,
        "stereo": stereo,
        "position_scale": scale,
        "image_size": args.image_size,
        "seconds": round(elapsed, 2),
        "fps": round(len(frames) / elapsed, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0),
        "overrides": args.set,
    }
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
