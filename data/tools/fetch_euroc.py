"""Download one EuRoC MAV sequence (ASL format) from the ETH Research Collection.

    python data/tools/fetch_euroc.py MH_01_easy /content/euroc [--cache /content/drive/MyDrive/rs-hackatum/euroc]

ETH only hosts whole groups (machine_hall.zip is 12.7 GB). The server honours
HTTP range requests, so remotezip pulls just the one inner <seq>.zip
(MH_01_easy: 1.6 GB) and skips the ROS bags. Result: <dest>/<seq>/mav0/...
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from remotezip import RemoteZip

# ETH Research Collection, DOI 10.3929/ethz-b-000690084 (checked 2026-11-27).
BASE = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/{}/content"
GROUPS = {
    "MH": ("machine_hall", "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe"),
    "V1": ("vicon_room1", "02ecda9a-298f-498b-970c-b7c44334d880"),
    "V2": ("vicon_room2", "ea12bc01-3677-4b4c-853d-87c7870b8c44"),
}
HEADERS = {"User-Agent": "Mozilla/5.0 (rs-hackatum-2026 dataset fetch)"}


def fetch_inner_zip(seq: str, target: Path) -> None:
    group, bitstream = GROUPS[seq[:2]]
    member = f"{group}/{seq}/{seq}.zip"
    print(f"fetching {member} from {group}.zip (range requests)")
    with RemoteZip(BASE.format(bitstream), headers=HEADERS) as z:
        with z.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 << 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seq", help="e.g. MH_01_easy, V1_01_easy")
    parser.add_argument("dest", type=Path, help="extract to <dest>/<seq>/")
    parser.add_argument("--cache", type=Path, help="keep the sequence zip here (e.g. Google Drive)")
    args = parser.parse_args()

    if args.seq[:2] not in GROUPS:
        raise SystemExit(f"unknown sequence {args.seq}; expected MH_*, V1_* or V2_*")
    out = args.dest / args.seq
    if (out / "mav0").is_dir():
        print(f"{out} already extracted")
        return

    zip_dir = args.cache or args.dest
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"{args.seq}.zip"
    if not zip_path.exists():
        partial = zip_path.with_suffix(".zip.part")
        fetch_inner_zip(args.seq, partial)
        partial.rename(zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    # tolerate an extra top-level folder inside the sequence zip
    mav0 = next(out.rglob("mav0"))
    if mav0.parent != out:
        shutil.move(str(mav0), out / "mav0")
    print(f"{out / 'mav0'} ready")


if __name__ == "__main__":
    main()
