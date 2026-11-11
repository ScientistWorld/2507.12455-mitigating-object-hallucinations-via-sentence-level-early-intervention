#!/bin/bash
# Visible development-slice evaluator -> scoring/scores_train.json.
# Runs with no arguments and scores every outputs/*_predictions.jsonl.

set -e
cd "$(dirname "$0")/.."
mkdir -p scoring

python -m eval.train.evaluate --input-dir outputs --out scoring/scores_train.json
