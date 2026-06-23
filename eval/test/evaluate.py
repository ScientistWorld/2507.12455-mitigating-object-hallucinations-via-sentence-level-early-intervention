from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OBJECTS = [f"{c} {s}" for c in ["red", "blue", "green", "yellow"] for s in ["circle", "square", "triangle"]]
EXPERIMENT = "synthetic_object_halbench"
TRAIN_SEED = "synthetic-object-halbench-agent-gym-v1-train"
TEST_SEED = "synthetic-object-halbench-agent-gym-v1-test-heldout"
SLICE_SIZE = 100


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_objects(text: str) -> list[str]:
    low = text.lower()
    found = []
    for obj in OBJECTS:
        if obj in low:
            found.append(obj)
    return found


def row_key(row: dict, index: int) -> str:
    return str(row.get("id", f"row_{index:08d}"))


def stable_key(seed: str, row: dict, index: int) -> str:
    row_id = row_key(row, index)
    return hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()


def test_rows(rows: list[dict]) -> list[dict]:
    if len(rows) < 2 * SLICE_SIZE:
        raise ValueError(f"Test slice requires at least {2 * SLICE_SIZE} rows, found {len(rows)}")
    train_ranked = sorted(enumerate(rows), key=lambda pair: stable_key(TRAIN_SEED, pair[1], pair[0]))
    train_ids = {row_key(row, index) for index, row in train_ranked[:SLICE_SIZE]}
    candidates = [(index, row) for index, row in enumerate(rows) if row_key(row, index) not in train_ids]
    ranked = sorted(candidates, key=lambda pair: stable_key(TEST_SEED, pair[1], pair[0]))
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


def disk_method_names(input_dir: Path) -> list[str]:
    return sorted(path.name.removesuffix("_predictions.jsonl") for path in input_dir.glob("*_predictions.jsonl"))


def train_score_method_names(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        results = data["experiments"][EXPERIMENT]["results"]
    except KeyError as exc:
        raise KeyError(f"{path} does not contain experiments.{EXPERIMENT}.results") from exc
    return sorted(results)


def score_prediction_file(path: Path) -> dict:
    rows = test_rows(read_jsonl(path))
    if not rows:
        raise ValueError(f"Test slice is empty for {path}")
    metrics = score_caption_rows(rows)
    metrics.update(score_qa_rows(rows))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--out", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    methods = train_score_method_names(Path(args.train_scores))
    if methods is None:
        methods = disk_method_names(input_dir)
    if not methods:
        raise FileNotFoundError(f"No methods found from {args.train_scores} or {input_dir}/*_predictions.jsonl")

    experiments: dict[str, dict] = {EXPERIMENT: {"results": {}}}
    missing = []
    for name in methods:
        path = input_dir / f"{name}_predictions.jsonl"
        if not path.exists():
            missing.append(str(path))
            continue
        experiments[EXPERIMENT]["results"][name] = score_prediction_file(path)
    if missing:
        raise FileNotFoundError("Missing prediction artifacts for test scoring: " + ", ".join(missing))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiments": experiments}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
