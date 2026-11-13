#!/bin/bash
# Hardening job 3 (see HARDENING.md): the held-out test grew from 700 to 2,000
# images; predictions resume, so every model only generates the new inputs.
# Also trains a seed-43 replicate of the reference (training-seed noise).
#   lane A: baseline, reference, lr3e-6, NLL lr5e-6, NLL lr8e-6 predictions; post-hoc filters
#   lane B: reference seed 43 (train + predict); lr4e-6, lr5e-6, half-data, 4k, 4k-seed43 predictions

set -uo pipefail
cd /home/user
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
L=logs/job3
PREDS=checkpoints/hardening/preds
mkdir -p "$L" "$PREDS"
START=$(date +%s)
mins() { echo $(( ($(date +%s) - START) / 60 )); }
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv || true
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
pick() {
  local other; other=$(cat "$L/gpu_$2" 2>/dev/null || echo -1)
  local g; g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' -v o="$other" '$1 != o {print $2, $1}' | sort -nr | head -n 1 | awk '{print $2}')
  [ -n "$g" ] || g=0
  [ "$NGPU" -ge 2 ] || g=0
  echo "$g" > "$L/gpu_$1"
  echo "cuda:$g"
}
pred() {  # pred <lane> <other> <name> <adapter|-> <out-dir>
  local args=(--name "$3" --out-dir "$5")
  [ "$4" = "-" ] || args+=(--adapter "$4")
  python -m method.sentinel.predict "${args[@]}" --device "$(pick $1 $2)" > $L/predict_$3.log 2>&1
}
laneA() {
  pred A B baseline - outputs
  pred A B sentinel checkpoints/sentinel/adapter outputs
  for v in sentinel_lr3e6 sentinel_nll_lr5e6 sentinel_nll_lr8e6; do pred A B $v checkpoints/$v/adapter $PREDS; done
  cp outputs/sentinel_predictions.jsonl outputs/baseline_predictions.jsonl $PREDS/
  for v in sentinel baseline; do
    python -m redteam.hardening_v2.posthoc_filter --src outputs/${v}_predictions.jsonl --dst $PREDS/posthoc_${v}_predictions.jsonl --device "$(pick A B)" > $L/posthoc_$v.log 2>&1
  done
}
laneB() {
  sleep 30
  python -m method.sentinel.cdpo --out checkpoints/sentinel_seed43 --seed 43 --device "$(pick B A)" > $L/cdpo_sentinel_seed43.log 2>&1 &&
  pred B A sentinel_seed43 checkpoints/sentinel_seed43/adapter $PREDS
  for v in sentinel_lr4e6 sentinel_lr5e6 sentinel_half_data; do pred B A $v checkpoints/$v/adapter $PREDS; done
  for v in sentinel_4k sentinel_4k_seed43; do pred B A $v checkpoints/v1/$v/adapter $PREDS; done
}
laneA & a=$!; laneB & b=$!
wait $a; wait $b
grep -h "C-DPO on\|saved\|OOM" $L/cdpo_*.log; grep "cdpo {" $L/cdpo_*.log | tail -n 1
grep -h "OOM\|wrote\|Error" $L/predict_*.log $L/posthoc_*.log
wc -l outputs/*.jsonl $PREDS/*.jsonl
echo "[run3] done $(mins) min"
