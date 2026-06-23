#!/bin/bash
# Visible train-slice evaluator for future method development.
#
# Contract mirrors scripts/evaluate.sh: pass an input directory containing
# '*_predictions.jsonl' artifacts, and this writes a score JSON file.

set -e

cd /home/user

mkdir -p scoring

INPUT_DIR="${1:-outputs}"
python -m eval.train.evaluate --input-dir "$INPUT_DIR" --out scoring/scores_train.json
