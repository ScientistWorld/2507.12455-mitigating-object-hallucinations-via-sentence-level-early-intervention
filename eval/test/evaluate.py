"""Held-out test evaluator.  Writes scoring/scores_test.json.

Scores every method whose raw predictions are on disk
(``<input-dir>/<method>_predictions.jsonl``, rows ``{"id", "prediction"}``),
every method listed in scoring/scores_train.json, and the primary submission
slot ``sentinel``; a method listed but without a usable file is scored as
failed.  The comparison baseline (unaligned LLaVA-v1.5-7B) is a frozen file
owned by this directory; a ``baseline_predictions.jsonl`` on disk is ignored.

Every number is recomputed here from raw text against answers that live only
in this directory.  A run is *failed* -- and receives the worst value of every
metric of every experiment -- when its file is unreadable, malformed, has
duplicate or missing test ids, or violates a capability constraint:
  * object coverage below COVERAGE_FLOOR (0.9) x the frozen baseline's coverage
    (shorter, terser or emptier descriptions cannot buy low hallucination rates);
  * VQAv2 accuracy more than VQA_MAX_DROP points below the frozen baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .chair import score_captions
from .vqa_accuracy import accuracy

HERE = Path(__file__).resolve().parent
GOLD = HERE / "gold.json"
FIXED_BASELINE = HERE / "fixed_baseline_predictions.jsonl"
BASELINE_NAME = "baseline"
PRIMARY_SLOT = "sentinel"
COVERAGE_FLOOR = 0.9
VQA_MAX_DROP = 5.0

CAPTION_EXP = "object_halbench_coco"
VQA_EXP = "vqav2_general"
WORST = {
    CAPTION_EXP: {"response_hallucination_rate": 100.0, "mention_hallucination_rate": 100.0, "object_coverage": 0.0},
    VQA_EXP: {"vqa_accuracy": 0.0},
}
BOUNDS = {"response_hallucination_rate": (0, 100), "mention_hallucination_rate": (0, 100), "object_coverage": (0, 100), "vqa_accuracy": (0, 100)}


class Failed(Exception):
    pass


def load_predictions(path: Path, required: set[str]) -> dict[str, str]:
    if not path.is_file():
        raise Failed(f"missing prediction file {path.name}")
    preds: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise Failed(f"unreadable prediction file: {exc}") from exc
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Failed(f"line {n}: invalid JSON") from exc
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("prediction"), str):
            raise Failed(f"line {n}: each row must be an object with string 'id' and string 'prediction'")
        if row["id"] in preds:
            raise Failed(f"duplicate id {row['id']}")
        preds[row["id"]] = row["prediction"]
    missing = sorted(required - set(preds))
    if missing:
        raise Failed(f"{len(missing)} test inputs without a prediction (e.g. {missing[0]})")
    return preds


def score(preds: dict[str, str], gold: dict) -> dict[str, dict[str, float]]:
    caps = [(preds[cid], set(g["objects"])) for cid, g in sorted(gold["captions"].items())]
    vqa = [accuracy(preds[qid], g["answers"]) for qid, g in sorted(gold["vqa"].items())]
    out = {CAPTION_EXP: score_captions(caps), VQA_EXP: {"vqa_accuracy": 100.0 * sum(vqa) / max(1, len(vqa))}}
    for exp in out.values():
        for k, v in exp.items():
            lo, hi = BOUNDS[k]
            if not (isinstance(v, float) and math.isfinite(v) and lo <= v <= hi):
                raise Failed(f"impossible value {k}={v}")
    return out


def method_names(input_dir: Path, train_scores: Path) -> list[str]:
    names = {p.name[: -len("_predictions.jsonl")] for p in input_dir.glob("*_predictions.jsonl")}
    if train_scores.is_file():
        try:
            data = json.loads(train_scores.read_text())
            for exp in data.get("experiments", {}).values():
                names |= set((exp.get("results") or {}).keys())
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    names.add(PRIMARY_SLOT)
    names.discard(BASELINE_NAME)
    return sorted(n for n in names if n and not n.startswith("_"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs")
    ap.add_argument("--out", default="scoring/scores_test.json")
    ap.add_argument("--train-scores", default="scoring/scores_train.json")
    args = ap.parse_args()

    Path(args.out).unlink(missing_ok=True)  # never leave a stale scoreboard behind
    gold = json.loads(GOLD.read_text())
    required = set(gold["captions"]) | set(gold["vqa"])
    results = {CAPTION_EXP: {}, VQA_EXP: {}}
    status: dict[str, str] = {}

    floors = None
    if FIXED_BASELINE.is_file():
        base = score(load_predictions(FIXED_BASELINE, required), gold)
        for exp, metrics in base.items():
            results[exp][BASELINE_NAME] = metrics
        status[BASELINE_NAME] = "ok (frozen evaluator-owned baseline)"
        floors = {
            "object_coverage_min": COVERAGE_FLOOR * base[CAPTION_EXP]["object_coverage"],
            "vqa_accuracy_min": base[VQA_EXP]["vqa_accuracy"] - VQA_MAX_DROP,
        }

    input_dir = Path(args.input_dir)
    for name in method_names(input_dir, Path(args.train_scores)):
        try:
            metrics = score(load_predictions(input_dir / f"{name}_predictions.jsonl", required), gold)
            if floors is not None:
                cov = metrics[CAPTION_EXP]["object_coverage"]
                acc = metrics[VQA_EXP]["vqa_accuracy"]
                if cov < floors["object_coverage_min"]:
                    raise Failed(f"object coverage {cov:.2f} below the floor {floors['object_coverage_min']:.2f}")
                if acc < floors["vqa_accuracy_min"]:
                    raise Failed(f"VQA accuracy {acc:.2f} below the floor {floors['vqa_accuracy_min']:.2f}")
            status[name] = "ok"
        except Failed as exc:
            metrics = {exp: dict(v) for exp, v in WORST.items()}
            status[name] = f"failed: {exc}"
        for exp, m in metrics.items():
            results[exp][name] = m

    out = {
        "experiments": {exp: {"results": res, "_status": status} for exp, res in results.items()},
        "_meta": {"split": "test", "caption_inputs": len(gold["captions"]), "vqa_inputs": len(gold["vqa"]), "constraint_floors": floors},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, st in sorted(status.items()):
        print(f"{name}: {st}")


if __name__ == "__main__":
    main()
