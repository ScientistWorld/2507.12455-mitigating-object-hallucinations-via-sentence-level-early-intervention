# Train/Test Evaluation Design Notes

## Checked

- Read `eval/train/evaluate.py`, `eval/test/evaluate.py`, `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, and `scoring/reference.json` side by side.
- Confirmed both split scripts enumerate `*_predictions.jsonl` artifacts from disk rather than hardcoding a method name.
- Confirmed `scripts/evaluate_test.sh` invokes `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Grepped for train/test cross-imports; none were found. The test evaluator only reads `scoring/scores_train.json` to keep the method list aligned with the visible train run.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stub exists.
- Ran `bash -n` on both split scripts.

## Changed

- Fixed a label-tampering shortcut in both split evaluators. The first pass scored `objects` and `qa` fields from each method's prediction JSONL, so a shortcut method could edit gold labels inside its own artifact. Train and test now load authoritative labels from `data/eval_scenes.jsonl`, join predictions by `id`, and score only the method-supplied `prediction` text.
- Added duplicate-id and missing-prediction checks. A method can no longer silently change the evaluated slice by omitting hard rows or duplicating easier rows.
- Made `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` pass `--gold-data data/eval_scenes.jsonl` explicitly so the artifact contract is visible from the entry points.
- Added missing metric descriptions in `scoring/reference.json` for all nonzero-coefficient metrics. The descriptions identify hallucination benefit metrics and capability/coverage constraints without introducing paper-agreement metrics.

## Runtime Concerns For Next Iteration

- I did not execute the Python evaluators or produce scores in this design-only pass. The next runner should verify that the existing `outputs/*_predictions.jsonl` artifacts still cover all train and test slice ids after the authoritative-gold join.
- The split is still an IID row split over a compact 240-row synthetic evaluation set. It now catches row memorization and label tampering, but it does not add an OOD subgroup; if future work expands `data/eval_scenes.jsonl`, consider a held-out composition or difficulty-stratified slice.
- `qa_accuracy` remains derived from response text against gold QA labels, matching the existing target description. Existing artifacts also contain `qa_predictions`; a later design pass could add an explicit QA-output metric, but that would require refreshing reference values.
- The legacy unsplit `scripts/evaluate.sh` / `eval/metrics.py` path still exists for the original reproduction workflow and was outside this pass's edit scope. This pass hardened only the new train/test harness.
