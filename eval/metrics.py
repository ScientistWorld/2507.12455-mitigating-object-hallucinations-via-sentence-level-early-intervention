from __future__ import annotations

import argparse
import json
from pathlib import Path

OBJECTS = [f"{c} {s}" for c in ["red", "blue", "green", "yellow"] for s in ["circle", "square", "triangle"]]


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
        for qa in row.get("qa_predictions", []):
            pred = qa.get("prediction", "").strip().lower()
            gold = qa.get("answer", "").strip().lower()
            if pred == gold:
                correct += 1
            total += 1
    return {"qa_accuracy": 100.0 * correct / max(1, total)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--out", default="scoring/scores.json")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    experiments: dict[str, dict] = {"synthetic_object_halbench": {"results": {}}}
    for name in ["language_prior", "baseline", "sentinel", "standard_dpo"]:
        path = input_dir / f"{name}_predictions.jsonl"
        if path.exists():
            rows = read_jsonl(path)
            metrics = score_caption_rows(rows)
            metrics.update(score_qa_rows(rows))
            experiments["synthetic_object_halbench"]["results"][name] = metrics

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiments": experiments}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
