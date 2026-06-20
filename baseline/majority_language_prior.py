from __future__ import annotations

import argparse
import json
from pathlib import Path


FREQUENT_OBJECTS = ["red circle", "blue square", "green triangle"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", default="data/eval_scenes.jsonl")
    parser.add_argument("--out", default="outputs/language_prior_predictions.jsonl")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.eval_data))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    caption = " ".join(f"There is a {obj}." for obj in FREQUENT_OBJECTS)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in rows:
            qa_predictions = []
            for qa in row["qa"]:
                qa_predictions.append({**qa, "prediction": "yes" if qa["object"] in FREQUENT_OBJECTS else "no"})
            f.write(json.dumps({**row, "prediction": caption, "qa_predictions": qa_predictions}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
