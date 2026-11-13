#!/bin/bash
# Hardening job (see HARDENING.md): rebuild the reference on LLaVA-v1.5-7B and
# run the honest variants, brute-force check and shortcut cases through the
# hidden test.  Every stage resumes from its saved artifacts.
#
#   lane A: baseline predictions -> preference shard 0 -> C-DPO reference
#           -> predictions -> half-data variant -> post-hoc detector filter
#   lane B: preference shard 1 -> lr 5e-6 variant -> re-predict two earlier
#           seed-42/43 adapters (training-seed noise)
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
PREF_STOP=${PREF_STOP:-115}

nvidia-smi || true
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, torch.cuda.device_count(), 'GPUs')" || exit 1
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

# All node GPUs are visible inside the container and are shared with other
# jobs: before each stage take the GPU with the most free memory that the
# other lane is not using.
pick() {  # pick <my-lane> <other-lane>
  local other; other=$(cat "$L/gpu_$2" 2>/dev/null || echo -1)
  local g; g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' -v o="$other" '$1 != o {print $2, $1}' | sort -nr | head -n 1 | awk '{print $2}')
  [ -n "$g" ] || g=0
  [ "$NGPU" -ge 2 ] || g=0
  echo "$g" > "$L/gpu_$1"
  echo "cuda:$g"
}

# ---- 0. sanity: batched greedy generation with the fixed decode path -------
if [ ! -f checkpoints/smoke/OK ]; then
  D=$(pick A B)
  python -m method.sentinel.predict --name smoke --limit 8 --out-dir checkpoints/smoke/outputs --device $D || exit 1
  cut -c1-400 checkpoints/smoke/outputs/smoke_predictions.jsonl
  touch checkpoints/smoke/OK
fi

# ---- 1. baseline predictions + preference construction ---------------------
laneA1() {
  python -m method.sentinel.predict --name baseline --device "$(pick A B)" > $L/predict_baseline.log 2>&1 || return 1
  python -m method.sentinel.build_preferences --shard 0 --num-shards 2 --device "$(pick A B)" --time-budget-min "$(budget $PREF_STOP)" > $L/prefs0.log 2>&1
}
laneB1() {
  sleep 20
  python -m method.sentinel.build_preferences --shard 1 --num-shards 2 --device "$(pick B A)" --time-budget-min "$(budget $PREF_STOP)" > $L/prefs1.log 2>&1
}
if [ ! -f checkpoints/preferences/pairs.jsonl ]; then
  echo "[run] stage 1 $(mins) min"
  laneA1 & a=$!; laneB1 & b=$!
  wait $a; ra=$?; wait $b; rb=$?
  tail -n 3 $L/predict_baseline.log $L/prefs0.log $L/prefs1.log
  [ $ra -eq 0 ] && [ $rb -eq 0 ] || { echo "[run] stage 1 failed ($ra/$rb)"; exit 1; }
  python -m method.sentinel.build_preferences --merge || exit 1
fi
cat checkpoints/preferences/summary.json

# ---- 2. C-DPO reference, variants, brute-force check -----------------------
laneA2() {
  python -m method.sentinel.cdpo --out checkpoints/sentinel --device "$(pick A B)" > $L/cdpo_sentinel.log 2>&1 || return 1
  python -m method.sentinel.predict --name sentinel --adapter checkpoints/sentinel/adapter --device "$(pick A B)" > $L/predict_sentinel.log 2>&1 || return 1
  if [ "$(mins)" -lt 255 ]; then
    python -m method.sentinel.cdpo --out checkpoints/sentinel_half_data --data-fraction 0.5 --steps 0 --epochs 1 --device "$(pick A B)" > $L/cdpo_half.log 2>&1 &&
    python -m method.sentinel.predict --name sentinel_half_data --adapter checkpoints/sentinel_half_data/adapter --out-dir $PREDS --device "$(pick A B)" > $L/predict_half.log 2>&1
  fi
  if [ "$(mins)" -lt 300 ]; then
    [ -s $PREDS/posthoc_sentinel_predictions.jsonl ] || python -m redteam.hardening_v2.posthoc_filter --src outputs/sentinel_predictions.jsonl --dst $PREDS/posthoc_sentinel_predictions.jsonl --device "$(pick A B)" > $L/posthoc_sentinel.log 2>&1
  fi
  if [ "$(mins)" -lt 315 ]; then
    [ -s $PREDS/posthoc_baseline_predictions.jsonl ] || python -m redteam.hardening_v2.posthoc_filter --src outputs/baseline_predictions.jsonl --dst $PREDS/posthoc_baseline_predictions.jsonl --device "$(pick A B)" > $L/posthoc_baseline.log 2>&1
  fi
  return 0
}
laneB2() {
  sleep 30
  python -m method.sentinel.cdpo --out checkpoints/sentinel_lr5e6 --lr 5e-6 --device "$(pick B A)" > $L/cdpo_lr5e6.log 2>&1 &&
  python -m method.sentinel.predict --name sentinel_lr5e6 --adapter checkpoints/sentinel_lr5e6/adapter --out-dir $PREDS --device "$(pick B A)" > $L/predict_lr5e6.log 2>&1
  for v in sentinel_4k sentinel_4k_seed43; do
    if [ "$(mins)" -lt 300 ]; then
      python -m method.sentinel.predict --name $v --adapter checkpoints/v1/$v/adapter --out-dir $PREDS --device "$(pick B A)" > $L/predict_$v.log 2>&1
    fi
  done
  return 0
}
echo "[run] stage 2 $(mins) min"
laneA2 & a=$!; laneB2 & b=$!
wait $a; ra=$?; wait $b
for f in $L/cdpo_*.log; do echo "== $f"; grep -E "C-DPO on|trainable|saved|OOM" "$f"; grep "cdpo {" "$f" | tail -n 2; done
grep -h "OOM\|wrote" $L/predict_*.log $L/posthoc_*.log 2>/dev/null
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
python -m redteam.hardening_v2.cases --preds $PREDS > $L/cases.log 2>&1; tail -n 40 $L/cases.log | cut -c1-600
python -m redteam.hardening_v2.rank --preds $PREDS
echo "[run] done $(mins) min"
