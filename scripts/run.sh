#!/bin/bash
# Hardening job 2 (see HARDENING.md): candidate "better" variants of the
# reference that must keep >= 0.9x baseline object coverage.  Reuses the
# reference preference data (checkpoints/preferences/pairs.jsonl).
#   lane A: plain C-DPO at lr 3e-6, then 4e-6
#   lane B: C-DPO + per-token NLL on the chosen sentence (weight 1) at lr 5e-6, then 8e-6

set -uo pipefail
cd /home/user
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
L=logs/job2
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
variant() {  # variant <lane> <other> <name> <cdpo args...>
  local lane=$1 other=$2 name=$3; shift 3
  python -m method.sentinel.cdpo --out checkpoints/$name --device "$(pick $lane $other)" "$@" > $L/cdpo_$name.log 2>&1 &&
  python -m method.sentinel.predict --name $name --adapter checkpoints/$name/adapter --out-dir $PREDS --device "$(pick $lane $other)" > $L/predict_$name.log 2>&1
}
laneA() { variant A B sentinel_lr3e6 --lr 3e-6; variant A B sentinel_lr4e6 --lr 4e-6; }
laneB() { sleep 30; variant B A sentinel_nll_lr5e6 --lr 5e-6 --sft-weight 1.0; variant B A sentinel_nll_lr8e6 --lr 8e-6 --sft-weight 1.0; }
laneA & a=$!; laneB & b=$!
wait $a; wait $b
for f in $L/cdpo_*.log; do echo "== $f"; grep -E "C-DPO on|OOM|saved" "$f"; grep "cdpo {" "$f" | tail -n 1; done
grep -h "OOM\|wrote" $L/predict_*.log
echo "[run2] done $(mins) min"
