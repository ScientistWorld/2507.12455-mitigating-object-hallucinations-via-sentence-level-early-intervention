#!/bin/bash
# Held-out test evaluator -> scoring/scores_test.json.
# Runs with no arguments; scores every outputs/*_predictions.jsonl, every
# method listed in scoring/scores_train.json and the primary slot 'sentinel'
# (missing or invalid submissions receive the worst value of every metric).

set -e
cd "$(dirname "$0")/.."
mkdir -p scoring

python -m eval.test.evaluate --input-dir outputs --out scoring/scores_test.json --train-scores scoring/scores_train.json
