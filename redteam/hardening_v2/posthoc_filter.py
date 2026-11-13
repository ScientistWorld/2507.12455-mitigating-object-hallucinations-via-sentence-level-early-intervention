"""Brute-force check: inference-time detector filtering (no training).

The paper motivates SENTINEL against inference-time intervention (Appendix
A.2): verify each generated sentence with open-vocabulary detectors and drop
the hallucinated ones.  This script applies that post-hoc to an existing
prediction file: every caption sentence whose extracted objects include one
that BOTH detectors reject is removed.  VQA answers are left unchanged.
It measures how much an add-on with extra inference-time models could gain on
the hidden test (the gym's constraints forbid it for submitted methods).

Usage: python -m redteam.hardening_v2.posthoc_filter --src outputs/x_predictions.jsonl --dst <file> --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from method.sentinel.common import ROOT, load_image, log, read_jsonl
from method.sentinel.detectors import CrossChecker
from method.sentinel.objects import ObjectExtractor

_SPLIT = re.compile(r"(?<=[.!?])\s+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    inputs = {r["id"]: r for r in read_jsonl(ROOT / "data/eval_inputs.jsonl")}
    preds = read_jsonl(Path(args.src))
    done = {r["id"]: r for r in read_jsonl(Path(args.dst))}  # resume
    checker = CrossChecker(args.device)
    extractor = ObjectExtractor()
    out = []
    removed = 0
    total = 0
    for k, row in enumerate(preds):
        item = inputs.get(row["id"])
        if row["id"] in done:
            out.append(done[row["id"]])
            continue
        if item is None or item["task"] != "caption":
            out.append(row)
            continue
        sents = [s for s in _SPLIT.split(row["prediction"].strip()) if s]
        objs = extractor.extract(sents)
        phrases = sorted({o["phrase"] for ob in objs for o in ob})
        verdict = checker.validate(checker.prepare(load_image(item["image"])), phrases) if phrases else {}
        keep = [s for s, ob in zip(sents, objs) if not any(verdict[o["phrase"]]["label"] == "hallucinated" for o in ob)]
        removed += len(sents) - len(keep)
        total += len(sents)
        out.append({"id": row["id"], "prediction": " ".join(keep)})
        if k % 200 == 0:
            log(f"posthoc {k}/{len(preds)} removed {removed}/{total} sentences")
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.dst).open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    log(f"posthoc filter wrote {args.dst}: removed {removed}/{total} sentences")


if __name__ == "__main__":
    main()
