#!/bin/bash
# Baseline: the unaligned LLaVA-v1.5-7B, greedy decoding on all evaluation
# inputs -> outputs/baseline_predictions.jsonl.
#
# The evaluators score an evaluator-owned frozen copy of these predictions
# (eval/{train,test}/fixed_baseline_predictions.jsonl); this script exists so
# the baseline can be regenerated and checked.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -m method.sentinel.predict --name baseline --device cuda:0
