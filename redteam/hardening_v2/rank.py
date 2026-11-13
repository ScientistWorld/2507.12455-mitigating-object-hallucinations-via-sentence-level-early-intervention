"""Analysis only: normalised rank and paired bootstrap on the hidden test.

rank(m) = sum_exp weight_exp * sum_metric |coef| * dir * (value_m - value_ref) / std_metric
with std taken over the disclosed cohort of non-failed runs (future cohorts
are unknown).  Paired bootstrap resamples test images (VQA questions follow
their image) and recomputes each metric difference against the reference.

Usage: python -m redteam.hardening_v2.rank --preds <dir> [--cases <dir>]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from eval.test.chair import caption_objects
from eval.test.vqa_accuracy import accuracy

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def per_image(pred_path: Path, gold: dict, image_of: dict) -> dict:
    preds = {r["id"]: r["prediction"] for r in read_jsonl(pred_path)}
    rows = defaultdict(lambda: {"hal": 0, "ment": 0, "bad": 0, "cov": 0, "gold": 0, "vqa": []})
    for cid, g in gold["captions"].items():
        m = caption_objects(preds.get(cid, ""))
        gs = set(g["objects"])
        r = rows[image_of[cid]]
        r["hal"] = int(bool(m - gs))
        r["ment"] = len(m)
        r["bad"] = len(m - gs)
        r["cov"] = len(m & gs)
        r["gold"] = len(gs)
    for qid, g in gold["vqa"].items():
        rows[image_of[qid]]["vqa"].append(accuracy(preds.get(qid, ""), g["answers"]))
    return dict(rows)


def aggregate(rows: dict, keys: list) -> dict:
    n = len(keys)
    ment = sum(rows[k]["ment"] for k in keys)
    vq = [a for k in keys for a in rows[k]["vqa"]]
    return {
        "response_hallucination_rate": 100 * sum(rows[k]["hal"] for k in keys) / n,
        "mention_hallucination_rate": 100 * sum(rows[k]["bad"] for k in keys) / ment if ment else 0.0,
        "object_coverage": 100 * sum(rows[k]["cov"] for k in keys) / max(1, sum(rows[k]["gold"] for k in keys)),
        "vqa_accuracy": 100 * sum(vq) / max(1, len(vq)),
    }


def bootstrap(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    keys = sorted(a)
    rng = random.Random(seed)
    diffs = defaultdict(list)
    for _ in range(n_boot):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        ma, mb = aggregate(a, sample), aggregate(b, sample)
        for k in ma:
            diffs[k].append(ma[k] - mb[k])
    point = {k: aggregate(a, keys)[k] - aggregate(b, keys)[k] for k in aggregate(a, keys)}
    out = {}
    for k, v in diffs.items():
        v.sort()
        out[k] = {"diff": point[k], "ci95": [v[int(0.025 * n_boot)], v[int(0.975 * n_boot) - 1]]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--cases", default=str(HERE / "cases"))
    ap.add_argument("--reference", default="sentinel")
    args = ap.parse_args()
    ref_cfg = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
    gold = json.loads((ROOT / "eval/test/gold.json").read_text())
    inputs = {r["id"]: r for r in read_jsonl(ROOT / "data/eval_inputs.jsonl")}
    image_of = {i: inputs[i]["image"] for i in list(gold["captions"]) + list(gold["vqa"])}

    # Collect hidden-test metric vectors from every case.
    vectors: dict[str, dict] = {}
    for res_path in sorted(Path(args.cases).glob("*/result.json")):
        res = json.loads(res_path.read_text())
        for m, v in (res.get("methods") or {}).items():
            label = m if res["case"] == "cohort_variants" else f"{res['case']}:{m}"
            if res["case"] != "cohort_variants" and m != "sentinel" and not (res["case"] == "shortcut_manifest_omission"):
                continue
            vectors[label] = v
    ref = vectors.get(args.reference)
    if ref is None:
        raise SystemExit("reference not found in cohort_variants")
    metrics = [(e, m, c) for e, ec in ref_cfg.items() for m, c in ec["metrics"].items()]
    valid = {k: v for k, v in vectors.items() if str(v.get("status", "")).startswith("ok")}
    # Normalising cohort: the honest attempts (frozen baseline + every trained
    # variant).  Shortcut entries are scored against it but do not set the spread.
    honest = {k: v for k, v in valid.items() if ":" not in k and not k.startswith("posthoc")}
    std = {m: statistics.pstdev([v[m] for v in honest.values()]) for _, m, _ in metrics}
    std_all = {m: statistics.pstdev([v[m] for v in valid.values()]) for _, m, _ in metrics}

    def rank(v: dict) -> float:
        total = 0.0
        for e, m, c in metrics:
            if std[m] <= 0:
                continue
            d = (v[m] - ref[m]) * (1 if c["higher_is_better"] else -1)
            total += ref_cfg[e]["weight"] * abs(c["coefficient"]) * d / std[m]
        return total

    ranks = {k: rank(v) for k, v in vectors.items()}
    std_main = std
    std = std_all
    ranks_all_valid_cohort = {k: rank(v) for k, v in vectors.items()}
    std = std_main
    boots = {}
    preds = Path(args.preds)
    ref_rows = per_image(preds / f"{args.reference}_predictions.jsonl", gold, image_of)
    for f in sorted(preds.glob("*_predictions.jsonl")):
        name = f.name[: -len("_predictions.jsonl")]
        if name == args.reference:
            continue
        boots[name] = bootstrap(per_image(f, gold, image_of), ref_rows)
    summary = {"reference": args.reference, "cohort_std": std, "cohort": sorted(honest), "vectors": vectors, "rank_vs_reference": ranks,
               "all_valid_cohort_std": std_all, "rank_vs_reference_all_valid_cohort": ranks_all_valid_cohort,
               "paired_bootstrap_vs_reference": boots}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    for k in sorted(ranks, key=ranks.get, reverse=True):
        v = vectors[k]
        print(f"{k:45s} rank {ranks[k]:+.3f} (all-valid {ranks_all_valid_cohort[k]:+.3f})  R {v['response_hallucination_rate']:.2f} M {v['mention_hallucination_rate']:.2f} C {v['object_coverage']:.2f} V {v['vqa_accuracy']:.2f}  {v.get('status', '')}")
    for k, b in boots.items():
        print(k, json.dumps({m: [round(x["diff"], 2), [round(y, 2) for y in x["ci95"]]] for m, x in b.items()}))


if __name__ == "__main__":
    main()
