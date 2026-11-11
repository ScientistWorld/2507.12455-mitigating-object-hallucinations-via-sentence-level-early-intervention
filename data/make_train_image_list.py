"""Write data/sentinel_train_images.txt: unlabelled COCO train2017 images.

Preference data for hallucination mitigation is built from the model's own
samples on unlabelled images; no annotation of these images is used anywhere.
The list is a seeded hash-ordered sample of the train2017 file names (the
listing can be reproduced from http://images.cocodataset.org/zips/train2017.zip
or from any directory containing those images).  The file is committed, so
this script only needs to be rerun to change the pool.

Usage: python data/make_train_image_list.py <dir with train2017 jpgs> [--n 6000]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

SEED = "sentinel-gym-v2-train-pool"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default=str(Path(__file__).with_name("sentinel_train_images.txt")))
    args = ap.parse_args()
    names = sorted(p.name for p in Path(args.image_dir).iterdir() if p.suffix == ".jpg")
    names.sort(key=lambda n: hashlib.sha256(f"{SEED}:{n}".encode()).hexdigest())
    Path(args.out).write_text("\n".join(names[: args.n]) + "\n", encoding="utf-8")
    print(f"wrote {min(args.n, len(names))} of {len(names)} names to {args.out}")


if __name__ == "__main__":
    main()
