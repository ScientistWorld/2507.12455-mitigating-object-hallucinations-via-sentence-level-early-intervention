# Train/Test Evaluation Design Notes

(Rewritten in iteration 11 for the rebuilt LLaVA-v1.5-7B / COCO gym; the notes for the retired synthetic gym are in git history.)

## Splits

- `eval/test/build_splits.py` (the only code that can regenerate answers) selected 2,300 COCO val2017 images with instance annotations and at least 3 VQAv2 questions, hash-ordered: 300 development images (`eval/train/gold.json`) and 2,000 held-out images (`eval/test/gold.json`; enlarged from 700 in iteration 11 so that seed-level differences fall below the test's paired noise, the first 700 unchanged). Each image has one description input (prompt cycles over eight Object HalBench-style prompts) and three VQAv2 questions. `data/eval_inputs.jsonl` holds the inputs of both slices without answers.
- Gold objects follow CHAIR: MSCOCO instance categories plus categories named in the five human captions, mapped through the standard synonym list (`coco_synonyms.txt`).

## Evaluators

- `eval/train/` and `eval/test/` share no code (each has its own CHAIR and VQA scorer copy) and never import `method/`.
- Both score every `outputs/*_predictions.jsonl`, plus the primary slot `sentinel`; the test evaluator also scores every method listed in `scoring/scores_train.json`. Missing or malformed submissions get the worst value of every metric.
- The baseline is an evaluator-owned frozen file (`fixed_baseline_predictions.jsonl`); a `baseline_predictions.jsonl` on disk is ignored.
- Failure floors (relative to the frozen baseline of the same slice): object coverage >= 0.9x, VQA accuracy >= baseline - 5.
- The output score file is deleted before scoring, so a crash never leaves a stale scoreboard.

## Rank calibration

Coverage carries coefficient 1.5 against 1.0 for each hallucination rate so that truncating descriptions (moving along the base model's own length/hallucination trade-off) is roughly rank-neutral; VQA has weight 0.02 because honest variants differ by ~0.1-0.3 points and normalisation by that spread would otherwise amplify answer noise. Evidence in `HARDENING.md` and `redteam/hardening_v2/summary.json`.
