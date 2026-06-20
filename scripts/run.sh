#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

rm -rf outputs checkpoints/sentinel scoring/scores.json
bash scripts/reproduce.sh
python validate.py --compare

python - <<'PY'
import json
from pathlib import Path

for name in ["baseline", "sentinel"]:
    path = Path("outputs") / f"{name}_predictions.jsonl"
    print(f"\n{name} sample predictions:")
    for line in path.read_text(encoding="utf-8").splitlines()[:3]:
        row = json.loads(line)
        print(f"- {row['prediction']}")

pairs = Path("checkpoints/sentinel/preference_pairs.jsonl")
if pairs.exists():
    print(f"\npreference_pairs: {sum(1 for _ in pairs.open(encoding='utf-8'))}")
PY
