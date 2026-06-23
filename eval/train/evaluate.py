from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OBJECTS = [f"{c} {s}" for c in ["red", "blue", "green", "yellow"] for s in ["circle", "square", "triangle"]]
EXPERIMENT = "synthetic_object_halbench"
TRAIN_SEED = "synthetic-object-halbench-agent-gym-v1-train"
SLICE_SIZE = 100


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_by_id(rows: list[dict], source: str) -> dict[str, dict]:
    keyed = {}
    duplicates = []
    for index, row in enumerate(rows):
        row_id = row_key(row, index)
        if row_id in keyed:
            duplicates.append(row_id)
        keyed[row_id] = row
    if duplicates:
        raise ValueError(f"{source} contains duplicate ids: {', '.join(sorted(set(duplicates))[:10])}")
    return keyed


def attach_predictions(gold_rows: list[dict], prediction_rows: list[dict], source: Path) -> list[dict]:
    predictions = rows_by_id(prediction_rows, str(source))
    scored_rows = []
    missing = []
    for index, gold in enumerate(gold_rows):
        row_id = row_key(gold, index)
        pred = predictions.get(row_id)
        if pred is None:
            missing.append(row_id)
            continue
        scored_rows.append({**gold, "prediction": pred.get("prediction", "")})
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{source} is missing predictions for train rows: {preview}")
    return scored_rows


def extract_objects(text: str) -> list[str]:
    low = text.lower()
    found = []
    for obj in OBJECTS:
        if obj in low:
            found.append(obj)
    return found


def row_key(row: dict, index: int) -> str:
    return str(row.get("id", f"row_{index:08d}"))


def stable_key(row: dict, index: int) -> str:
    row_id = row_key(row, index)
    return hashlib.sha256(f"{TRAIN_SEED}:{row_id}".encode("utf-8")).hexdigest()


def train_rows(rows: list[dict]) -> list[dict]:
    if len(rows) < SLICE_SIZE:
        raise ValueError(f"Train slice requires at least {SLICE_SIZE} rows, found {len(rows)}")
    ranked = sorted(enumerate(rows), key=lambda pair: stable_key(pair[1], pair[0]))
    selected = sorted(ranked[:SLICE_SIZE], key=lambda pair: pair[0])
    return [row for _, row in selected]


def score_caption_rows(rows: list[dict]) -> dict:
    total_mentions = 0
    hallucinated_mentions = 0
    hallucinated_responses = 0
    true_objects_mentioned = 0
    total_true_objects = 0
    lengths = []
    for row in rows:
        gold = set(row["objects"])
        pred = extract_objects(row.get("prediction", ""))
        hallucinated = [obj for obj in pred if obj not in gold]
        total_mentions += len(pred)
        hallucinated_mentions += len(hallucinated)
        hallucinated_responses += int(bool(hallucinated))
        true_objects_mentioned += len(set(pred) & gold)
        total_true_objects += len(gold)
        lengths.append(len(pred))

    n = max(1, len(rows))
    return {
        "response_hallucination_rate": 100.0 * hallucinated_responses / n,
        "mention_hallucination_rate": 100.0 * hallucinated_mentions / max(1, total_mentions),
        "object_recall": 100.0 * true_objects_mentioned / max(1, total_true_objects),
        "avg_object_mentions": sum(lengths) / max(1, len(lengths)),
    }


def score_qa_rows(rows: list[dict]) -> dict:
    correct = 0
    total = 0
    for row in rows:
        mentioned = set(extract_objects(row.get("prediction", "")))
        for qa in row.get("qa", []):
            pred = "yes" if qa.get("object") in mentioned else "no"
            gold = qa.get("answer", "").strip().lower()
            if pred == gold:
                correct += 1
            total += 1
    return {"qa_accuracy": 100.0 * correct / max(1, total)}


def method_name(path: Path) -> str:
    return path.name.removesuffix("_predictions.jsonl")


def score_prediction_file(path: Path, gold_rows: list[dict]) -> dict:
    rows = attach_predictions(train_rows(gold_rows), read_jsonl(path), path)
    if not rows:
        raise ValueError(f"Train slice is empty for {path}")
    metrics = score_caption_rows(rows)
    metrics.update(score_qa_rows(rows))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--gold-data", default="data/eval_scenes.jsonl")
    parser.add_argument("--out", default="scoring/scores_train.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gold_rows = read_jsonl(Path(args.gold_data))
    prediction_paths = sorted(input_dir.glob("*_predictions.jsonl"))
    if not prediction_paths:
        raise FileNotFoundError(f"No '*_predictions.jsonl' artifacts found in {input_dir}")

    experiments: dict[str, dict] = {EXPERIMENT: {"results": {}}}
    for path in prediction_paths:
        experiments[EXPERIMENT]["results"][method_name(path)] = score_prediction_file(path, gold_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiments": experiments}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
