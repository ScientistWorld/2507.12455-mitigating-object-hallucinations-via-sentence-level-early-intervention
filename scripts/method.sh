#!/bin/bash
# Run the paper's method (SENTINEL on LLaVA-v1.5-7B) end to end:
#   1. in-domain preference data construction (one shard per visible GPU)
#   2. C-DPO training with LoRA
#   3. greedy predictions on all evaluation inputs -> outputs/sentinel_predictions.jsonl
# Every stage resumes from its saved artifacts under checkpoints/.
#
# Usage: bash scripts/method.sh   (extra args are passed to the C-DPO trainer)

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# One preference shard per GPU, at most two (compute nodes may expose every
# GPU of the node; the reference used two).  Shards stop at 8,600 pairs total.
NGPU=$(python -c "import torch; print(min(2, max(1, torch.cuda.device_count())))")
pids=()
for ((k = 0; k < NGPU; k++)); do
  python -m method.sentinel.build_preferences --shard "$k" --num-shards "$NGPU" --device "cuda:$k" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python -m method.sentinel.build_preferences --merge
python -m method.sentinel.cdpo --out checkpoints/sentinel --device cuda:0 "$@"
python -m method.sentinel.predict --name sentinel --adapter checkpoints/sentinel/adapter --device cuda:0
