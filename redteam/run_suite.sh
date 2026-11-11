#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Cache-only checks first, so all shortcut evidence survives a training timeout.
cases=(
  reference
  weaker_no_alignment
  bruteforce_object_vector
  shortcut_copy_labels
  shortcut_rebuild_seed
  shortcut_gold_replacement
  shortcut_manifest_omission
  shortcut_polarity_flip
  shortcut_unscored_hallucinations
  shortcut_empty_caption
  shortcut_all_objects
  shortcut_metric_forgery
  shortcut_testscore_forgery
  shortcut_null_prediction
  shortcut_empty_file
  shortcut_missing_file
  shortcut_baseline_replacement
  weaker_undertrained
  reference_seed7
  reference_seed17
  reference_seed27
  better_more_training
  better_clean_training
)

for case_name in "${cases[@]}"; do
  if [[ -f "redteam/${case_name}/result.json" ]]; then
    echo "Already recorded ${case_name}"
    continue
  fi
  echo "Running ${case_name}"
  # Every run.sh installs a real candidate and invokes both no-argument scripts.
  # case_runner restores data, outputs, and score files even if a case fails.
  if ! bash "redteam/${case_name}/run.sh"; then
    echo "Case failed: ${case_name}; see its error/training logs"
  fi
done
python -m redteam.report > redteam/summary.log
