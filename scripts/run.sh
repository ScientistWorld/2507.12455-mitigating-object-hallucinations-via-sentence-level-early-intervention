#!/bin/bash
# Hardening job (see HARDENING.md): rebuild the reference on LLaVA-v1.5-7B and
# run the honest variants, brute-force check and shortcut cases through the
# hidden test.  Every stage resumes from its saved artifacts.
#
#   lane A (cuda:0): baseline predictions -> preference shard 0 -> C-DPO reference
#                    -> predictions -> half-data variant -> post-hoc detector filter
#   lane B (cuda:1): preference shard 1 -> lr 5e-6 variant -> seed-43 replicate
#   then (CPU): freeze evaluator-owned baseline, reference scores, cases, analysis.

set -uo pipefail
cd /home/user
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Jobs get one CPU core: no thread oversubscription; nvidia-smi and torch share PCI ordering.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
L=logs/job
PREDS=checkpoints/hardening/preds
mkdir -p "$L" "$PREDS" outputs
START=$(date +%s)
mins() { echo $(( ($(date +%s) - START) / 60 )); }
budget() { local b=$(( $1 - $(mins) )); [ "$b" -gt 1 ] && echo "$b" || echo 1; }

nvidia-smi || true
python -c "import torch, transformers, peft, spacy; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'spacy', spacy.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" || exit 1
env | grep -i -E "cuda|nvidia|gpu|ray_|slurm" | sort || true
nproc || true
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
# All node GPUs are visible inside the container; use the two least loaded.
read -r G0 G1 < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null |
  sort -t, -k2,2n -k3,3n | head -n 2 | cut -d, -f1 | tr '\n' ' ')
D0=cuda:${G0:-0}; D1=cuda:${G1:-1}
[ "$NGPU" -ge 2 ] || D1=$D0
echo "[run] using $D0 and $D1 of $NGPU visible GPUs"
PREF_STOP=${PREF_STOP:-120}

# ---- 0. smoke test of every stage on a few images --------------------------
if [ ! -f checkpoints/smoke/OK ]; then
  echo "[run] smoke test $(mins) min"
  python -m method.sentinel.build_preferences --out-dir checkpoints/smoke/prefs --num-images 12 --max-steps 3 --batch-images 4 --device $D0 || exit 1
  python -m method.sentinel.build_preferences --out-dir checkpoints/smoke/prefs --merge || exit 1
  python - <<'PY' || exit 1
import json
rows = [json.loads(l) for l in open("checkpoints/smoke/prefs/pairs.jsonl")]
for r in rows[:8]:
    print(round(r["ref_chosen"], 2), round(r["ref_rejected"], 2), len(r["chosen_ids"]), len(r["rejected_ids"]), "|", r["chosen"], "||", r["rejected"], "|", {k: v["label"] for k, v in r["rejected_objects"].items()})
per_tok = [r["ref_chosen"] / len(r["chosen_ids"]) for r in rows]
print("mean ref logp/token (chosen):", sum(per_tok) / len(per_tok))
assert rows and all(-15 < x < 0 for x in per_tok), "implausible reference log-probs"
PY
  python -m method.sentinel.cdpo --pairs checkpoints/smoke/prefs/pairs.jsonl --out checkpoints/smoke/cdpo --max-steps 4 --global-batch 4 --lr 1e-4 --device $D0 || exit 1
  python - <<'PY' || exit 1
import json
h = json.load(open("checkpoints/smoke/cdpo/train_log.json"))["history"]
print([round(x["margin"], 4) for x in h])
assert abs(h[-1]["margin"]) > 1e-3, "C-DPO margin did not move: log-prob plumbing is broken"
PY
  python -m method.sentinel.predict --name smoke --adapter checkpoints/smoke/cdpo/adapter --limit 6 --out-dir checkpoints/smoke/outputs --device $D0 || exit 1
  cat checkpoints/smoke/outputs/smoke_predictions.jsonl
  touch checkpoints/smoke/OK
fi

# ---- 1. baseline predictions + preference construction ---------------------
laneA1() {
  python -m method.sentinel.predict --name baseline --device $D0 > $L/predict_baseline.log 2>&1 || return 1
  python -m method.sentinel.build_preferences --shard 0 --num-shards 2 --device $D0 --time-budget-min "$(budget $PREF_STOP)" > $L/prefs0.log 2>&1
}
laneB1() {
  python -m method.sentinel.build_preferences --shard 1 --num-shards 2 --device $D1 --time-budget-min "$(budget $PREF_STOP)" > $L/prefs1.log 2>&1
}
if [ ! -f checkpoints/preferences/pairs.jsonl ]; then
  echo "[run] stage 1 $(mins) min"
  if [ "$NGPU" -ge 2 ]; then
    laneA1 & a=$!; laneB1 & b=$!
    wait $a; ra=$?; wait $b; rb=$?
  else
    laneA1; ra=$?; laneB1; rb=$?
  fi
  tail -n 3 $L/predict_baseline.log $L/prefs0.log $L/prefs1.log
  [ $ra -eq 0 ] && [ $rb -eq 0 ] || { echo "[run] stage 1 failed ($ra/$rb)"; exit 1; }
  python -m method.sentinel.build_preferences --merge || exit 1
fi
cat checkpoints/preferences/summary.json

# ---- 2. C-DPO reference, variants, brute-force check -----------------------
laneA2() {
  python -m method.sentinel.cdpo --out checkpoints/sentinel --device $D0 > $L/cdpo_sentinel.log 2>&1 || return 1
  python -m method.sentinel.predict --name sentinel --adapter checkpoints/sentinel/adapter --device $D0 > $L/predict_sentinel.log 2>&1 || return 1
  if [ "$(mins)" -lt 255 ]; then
    python -m method.sentinel.cdpo --out checkpoints/sentinel_half_data --data-fraction 0.5 --device $D0 > $L/cdpo_half.log 2>&1 &&
    python -m method.sentinel.predict --name sentinel_half_data --adapter checkpoints/sentinel_half_data/adapter --out-dir $PREDS --device $D0 > $L/predict_half.log 2>&1
  fi
  if [ "$(mins)" -lt 315 ]; then
    [ -s $PREDS/posthoc_sentinel_predictions.jsonl ] || python -m redteam.hardening_v2.posthoc_filter --src outputs/sentinel_predictions.jsonl --dst $PREDS/posthoc_sentinel_predictions.jsonl --device $D0 > $L/posthoc_sentinel.log 2>&1
    [ -s $PREDS/posthoc_baseline_predictions.jsonl ] || python -m redteam.hardening_v2.posthoc_filter --src outputs/baseline_predictions.jsonl --dst $PREDS/posthoc_baseline_predictions.jsonl --device $D0 > $L/posthoc_baseline.log 2>&1
  fi
  return 0
}
laneB2() {
  python -m method.sentinel.cdpo --out checkpoints/sentinel_lr5e6 --lr 5e-6 --device $D1 > $L/cdpo_lr5e6.log 2>&1 &&
  python -m method.sentinel.predict --name sentinel_lr5e6 --adapter checkpoints/sentinel_lr5e6/adapter --out-dir $PREDS --device $D1 > $L/predict_lr5e6.log 2>&1
  if [ "$(mins)" -lt 235 ]; then
    python -m method.sentinel.cdpo --out checkpoints/sentinel_seed43 --seed 43 --device $D1 > $L/cdpo_seed43.log 2>&1 &&
    python -m method.sentinel.predict --name sentinel_seed43 --adapter checkpoints/sentinel_seed43/adapter --out-dir $PREDS --device $D1 > $L/predict_seed43.log 2>&1
  fi
  return 0
}
echo "[run] stage 2 $(mins) min"
if [ "$NGPU" -ge 2 ]; then
  laneA2 & a=$!; laneB2 & b=$!
  wait $a; ra=$?; wait $b
else
  laneA2; ra=$?; laneB2
fi
for f in $L/cdpo_*.log; do echo "== $f"; grep -E "C-DPO on|trainable|saved" "$f"; grep "cdpo {" "$f" | tail -n 2; done
[ $ra -eq 0 ] || { echo "[run] reference training/prediction failed"; tail -n 30 $L/cdpo_sentinel.log $L/predict_sentinel.log; exit 1; }

# ---- 3. freeze evaluator-owned baseline, reference scores, cases -----------
echo "[run] stage 3 $(mins) min"
cp outputs/baseline_predictions.jsonl outputs/sentinel_predictions.jsonl $PREDS/
python - <<'PY'
import json
from pathlib import Path
rows = {json.loads(l)["id"]: l for l in Path("outputs/baseline_predictions.jsonl").read_text().splitlines() if l.strip()}
for split in ("train", "test"):
    gold = json.loads(Path(f"eval/{split}/gold.json").read_text())
    ids = sorted(set(gold["captions"]) | set(gold["vqa"]))
    Path(f"eval/{split}/fixed_baseline_predictions.jsonl").write_text("".join(rows[i] + "\n" for i in ids))
    print(split, len(ids), "frozen baseline rows")
PY
ls outputs
bash scripts/evaluate_train.sh && bash scripts/evaluate_test.sh && bash scripts/evaluate.sh
cat scoring/scores_test.json
python -m redteam.hardening_v2.cases --preds $PREDS > $L/cases.log 2>&1; tail -n 40 $L/cases.log
python -m redteam.hardening_v2.rank --preds $PREDS
echo "[run] done $(mins) min"
