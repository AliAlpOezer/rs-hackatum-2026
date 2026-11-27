"""Run DPVO on a sequence in the shared layout and write a TUM trajectory.

    python tracks/spatial-ai/runners/dpvo_runner.py data/MH_01_easy \
        --network /content/DPVO/dpvo.pth --config /content/DPVO/config/default.yaml \
        --out results/MH_01_easy/dpvo.txt

Needs a CUDA GPU and the dpvo package installed (see notebooks/dpvo_euroc.ipynb).
Output poses are camera-to-world, timestamps in seconds; a <out>.json sidecar
records runtime, frame count and peak GPU memory.
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
RADTAN = {"radtan", "radial-tangential", "none"}  # OpenCV k1 k2 p1 p2 model


def list_frames(seq: Path, stride: int) -> list[tuple[float, Path]]:
    """Frames named <timestamp_ns>.<ext>, sorted, every `stride`-th one."""
    files = sorted(p for p in (seq / "frames").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"no frames in {seq / 'frames'}")
    return [(int(p.stem) * 1e-9, p) for p in files[::stride]]


def load_camera(seq: Path) -> tuple[np.ndarray, np.ndarray]:
    cam = yaml.safe_load((seq / "calib.yaml").read_text())["camera"]
    if cam.get("model", "pinhole") != "pinhole" or cam.get("distortion_model", "radtan") not in RADTAN:
        raise SystemExit(f"unsupported camera {cam.get('model')}/{cam.get('distortion_model')}")
    fx, fy, cx, cy = cam["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    return K, np.array(cam.get("distortion", []), dtype=float)


def prepare(path: Path, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Read, undistort and crop to a multiple of 16 (DPVO's feature stride)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {path}")
    if dist.size and np.any(dist):
        image = cv2.undistort(image, K, dist)
    h, w = image.shape[:2]
    return image[: h - h % 16, : w - w % 16]


def prefetch(frames, K, dist, depth=16):
    """Decode images on a background thread while the GPU tracks."""
    q: queue.Queue = queue.Queue(maxsize=depth)

    def work():
        for t, path in frames:
            q.put((t, prepare(path, K, dist)))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def write_tum(path: Path, tstamps: np.ndarray, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([tstamps, poses]), fmt="%.9f",
               header="timestamp_s tx ty tz qx qy qz qw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seq", type=Path, help="data/<seq> folder")
    parser.add_argument("--network", type=Path, required=True, help="dpvo.pth")
    parser.add_argument("--config", type=Path, required=True, help="DPVO yaml, e.g. config/default.yaml")
    parser.add_argument("--out", type=Path, required=True, help="results/<seq>/<method>.txt")
    parser.add_argument("--stride", type=int, default=2, help="use every n-th frame (DPVO uses 2 on EuRoC)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--opts", nargs="*", default=[], help="DPVO cfg overrides, e.g. LOOP_CLOSURE True")
    args = parser.parse_args()

    import torch
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    cfg.merge_from_file(str(args.config))
    cfg.merge_from_list(args.opts)
    torch.manual_seed(args.seed)

    frames = list_frames(args.seq, args.stride)
    K, dist = load_camera(args.seq)
    intrinsics = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], device="cuda")
    if cfg.BUFFER_SIZE <= len(frames):
        cfg.BUFFER_SIZE = 2 * len(frames)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    slam = None
    with torch.no_grad():
        for t, image in prefetch(frames, K, dist):
            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            if slam is None:
                slam = DPVO(cfg, str(args.network), ht=image.shape[1], wd=image.shape[2])
            slam(t, image, intrinsics)
        poses, tstamps = slam.terminate()  # x y z qx qy qz qw, camera-to-world
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    write_tum(args.out, tstamps, poses)
    stats = {
        "method": "dpvo",
        "sequence": args.seq.name,
        "frames": len(frames),
        "stride": args.stride,
        "seconds": round(elapsed, 2),
        "fps": round(len(frames) / elapsed, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0),
        "opts": args.opts,
    }
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
