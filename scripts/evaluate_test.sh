#!/bin/bash
# Held-out test-slice evaluator for final anti-shortcut scoring.
#
# Contract mirrors scripts/evaluate.sh: pass the same input directory used for
# train scoring. When scoring/scores_train.json exists, every method listed
# there is scored here as well.

set -e

cd /home/user

mkdir -p scoring

INPUT_DIR="${1:-outputs}"
python -m eval.test.evaluate \
  --input-dir "$INPUT_DIR" \
  --out scoring/scores_test.json \
  --train-scores scoring/scores_train.json
