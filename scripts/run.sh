#!/bin/bash
# Job script submitted via action.yaml.
#
# Runs the visible train and held-out test scoring slices against the existing
# prediction artifacts produced by the frozen final method.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh outputs
bash scripts/evaluate_test.sh outputs
