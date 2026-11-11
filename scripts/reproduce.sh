#!/bin/bash
# Reproduce the reference end to end (run inside the container, workspace at
# /home/user, after scripts/download.sh).  Needs 1-2 GPUs with >= 40 GB.
#
#   baseline predictions -> SENTINEL preference data -> C-DPO -> predictions
#   -> development and held-out scores.

set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
