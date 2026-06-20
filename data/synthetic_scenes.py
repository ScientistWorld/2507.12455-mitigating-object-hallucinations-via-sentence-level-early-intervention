"""Synthetic object-hallucination benchmark data.

The original paper uses Visual Genome images and MLLM-generated captions.  This
module provides a small, tracked substitute with the same object-grounding
structure: every example has a visual object set, caption targets, and yes/no
questions.  It is intentionally independent from method/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "triangle"]
OBJECTS = [f"{c} {s}" for c in COLORS for s in SHAPES]
PROMPT = "Describe this image in detail."


def make_scene(rng: random.Random, scene_id: int, min_objects: int = 2, max_objects: int = 4) -> dict:
    n_obj = rng.randint(min_objects, max_objects)
    objects = sorted(rng.sample(OBJECTS, n_obj))
    caption = " ".join(f"There is a {obj}." for obj in objects)
    qa = []
    for obj in objects:
        qa.append({"question": f"Is there a {obj}?", "answer": "yes", "object": obj})
    for obj in rng.sample([o for o in OBJECTS if o not in objects], min(2, len(OBJECTS) - n_obj)):
        qa.append({"question": f"Is there a {obj}?", "answer": "no", "object": obj})
    rng.shuffle(qa)
    return {
        "id": f"scene_{scene_id:05d}",
        "prompt": PROMPT,
        "objects": objects,
        "caption": caption,
        "qa": qa,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--train", type=int, default=900)
    parser.add_argument("--eval", type=int, default=240)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    train = [make_scene(rng, i) for i in range(args.train)]
    eval_rows = [make_scene(rng, i + args.train) for i in range(args.eval)]
    write_jsonl(out / "train_scenes.jsonl", train)
    write_jsonl(out / "eval_scenes.jsonl", eval_rows)
    (out / "objects.json").write_text(json.dumps({"objects": OBJECTS}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
