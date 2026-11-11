#!/bin/bash
# Standard evaluation on the visible development slice -> scoring/scores.json.
#
# Self-contained in eval/ (never imports method/).  Scores every
# outputs/<method>_predictions.jsonl (rows {"id", "prediction"}) against the
# development answers; see briefing/evaluation.md for the output contract.
#
# Usage: bash scripts/evaluate.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p scoring

python -m eval.train.evaluate --input-dir outputs --out scoring/scores.json
