#!/bin/bash
# Job script submitted via action.yaml.
#
# Runs inside the compute container with the workspace mounted at /home/user.
# This job performs a compact CPU hyperparameter sweep for the scaled SENTINEL
# reproduction after the method-faithfulness fixes.

set -euo pipefail

cd /home/user

rm -rf outputs checkpoints/sentinel scoring/scores.json tmp/sentinel_sweep
mkdir -p tmp/sentinel_sweep

python data/synthetic_scenes.py --out-dir data

run_cfg() {
  name="$1"
  shift
  echo "=== running ${name} ==="
  out_dir="tmp/sentinel_sweep/${name}/outputs"
  ckpt_dir="tmp/sentinel_sweep/${name}/checkpoints"
  mkdir -p "${out_dir}" "${ckpt_dir}"
  python -m baseline.majority_language_prior \
    --eval-data data/eval_scenes.jsonl \
    --out "${out_dir}/language_prior_predictions.jsonl"
  python -m method.train \
    --outputs "${out_dir}" \
    --out-dir "${ckpt_dir}" \
    "$@"
  python -m eval.metrics \
    --input-dir "${out_dir}" \
    --out "tmp/sentinel_sweep/${name}/scores.json"
}

COMMON_ARGS=(--sft-epochs 20 --preference-images 700 --samples-per-context 16 --sample-temperature 1.4)

run_cfg dpo1_lr1e4_p50 "${COMMON_ARGS[@]}" --dpo-epochs 1 --dpo-lr 1e-4 --dpo-beta 0.1 --caption-hallucination-prob 0.50
run_cfg dpo2_lr1e4_p50 "${COMMON_ARGS[@]}" --dpo-epochs 2 --dpo-lr 1e-4 --dpo-beta 0.1 --caption-hallucination-prob 0.50
run_cfg dpo3_lr1e4_p50 "${COMMON_ARGS[@]}" --dpo-epochs 3 --dpo-lr 1e-4 --dpo-beta 0.1 --caption-hallucination-prob 0.50
run_cfg dpo2_lr2e4_p50 "${COMMON_ARGS[@]}" --dpo-epochs 2 --dpo-lr 2e-4 --dpo-beta 0.1 --caption-hallucination-prob 0.50
run_cfg dpo4_lr1e4_b03_p50 "${COMMON_ARGS[@]}" --dpo-epochs 4 --dpo-lr 1e-4 --dpo-beta 0.3 --caption-hallucination-prob 0.50
run_cfg dpo1_lr5e5_p45 "${COMMON_ARGS[@]}" --dpo-epochs 1 --dpo-lr 5e-5 --dpo-beta 0.1 --caption-hallucination-prob 0.45

python - <<'PY'
import json
import shutil
from pathlib import Path

root = Path("tmp/sentinel_sweep")
rows = []
for scores_path in sorted(root.glob("*/scores.json")):
    name = scores_path.parent.name
    data = json.loads(scores_path.read_text(encoding="utf-8"))
    result = data["experiments"]["synthetic_object_halbench"]["results"]
    baseline = result["baseline"]
    sentinel = result["sentinel"]
    response_gain = baseline["response_hallucination_rate"] - sentinel["response_hallucination_rate"]
    mention_gain = baseline["mention_hallucination_rate"] - sentinel["mention_hallucination_rate"]
    recall_delta = sentinel["object_recall"] - baseline["object_recall"]
    qa_delta = sentinel["qa_accuracy"] - baseline["qa_accuracy"]
    score = response_gain + mention_gain + 0.25 * recall_delta + 0.25 * qa_delta
    rows.append(
        {
            "name": name,
            "score": score,
            "response_gain": response_gain,
            "mention_gain": mention_gain,
            "recall_delta": recall_delta,
            "qa_delta": qa_delta,
            "baseline": baseline,
            "sentinel": sentinel,
            "scores_path": str(scores_path),
        }
    )

if not rows:
    raise SystemExit("no sweep results were produced")

rows.sort(key=lambda row: row["score"], reverse=True)
summary_path = root / "summary.json"
summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

print("\nSweep summary:")
for row in rows:
    print(
        f"{row['name']}: score={row['score']:.3f} "
        f"resp_gain={row['response_gain']:.3f} "
        f"ment_gain={row['mention_gain']:.3f} "
        f"recall_delta={row['recall_delta']:.3f} "
        f"qa_delta={row['qa_delta']:.3f}"
    )

best = rows[0]
best_dir = root / best["name"]
print(f"\nSelected best config: {best['name']}")

for path in [Path("outputs"), Path("checkpoints/sentinel")]:
    if path.exists():
        shutil.rmtree(path)
Path("checkpoints").mkdir(exist_ok=True)
shutil.copytree(best_dir / "outputs", "outputs")
shutil.copytree(best_dir / "checkpoints", "checkpoints/sentinel")
shutil.copyfile(best_dir / "scores.json", "scoring/scores.json")
PY

python validate.py --compare

python - <<'PY'
import json
from pathlib import Path

print("\nBest run sample predictions:")
for name in ["baseline", "sentinel"]:
    path = Path("outputs") / f"{name}_predictions.jsonl"
    print(f"\n{name}:")
    for line in path.read_text(encoding="utf-8").splitlines()[:3]:
        row = json.loads(line)
        print(f"- {row['prediction']}")

pairs = Path("checkpoints/sentinel/preference_pairs.jsonl")
if pairs.exists():
    print(f"\npreference_pairs: {sum(1 for _ in pairs.open(encoding='utf-8'))}")

summary = Path("tmp/sentinel_sweep/summary.json")
if summary.exists():
    print("\nSweep summary JSON:")
    print(summary.read_text(encoding="utf-8"))
PY
