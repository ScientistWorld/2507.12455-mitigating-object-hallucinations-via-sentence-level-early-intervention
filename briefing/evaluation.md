# Evaluation

## Output contract

A method writes `outputs/<method>_predictions.jsonl`, one JSON object per input of `data/eval_inputs.jsonl`:

```json
{"id": "cap_000000000785", "prediction": "The image features ..."}
{"id": "vqa_000000785000", "prediction": "Yes"}
```

`cap_*` inputs ask for a detailed description (eight prompt variants); `vqa_*` inputs are VQAv2 questions with the LLaVA-1.5 short-answer suffix. Predictions are the raw generated text. The evaluator ignores every other field. Every input of the scored split must have exactly one prediction.

`bash scripts/evaluate.sh` (and `scripts/evaluate_train.sh`) score the visible development slice (300 images, 900 questions). The held-out test slice (2,000 different images, 6,000 questions) is scored by a hidden evaluator through the same interface.

## Metrics

Object hallucination follows the Object HalBench / CHAIR protocol over the 80 MSCOCO categories and their standard synonym list. Ground truth for an image is the union of its MSCOCO instance categories and the objects named in its human reference captions. Each category is counted once per description; a mention directly preceded by a negation ("no dog", "without a cup") is not counted as an assertion.

- **response_hallucination_rate** (lower is better): % of descriptions asserting at least one absent object (CHAIR_s / "Resp.").
- **mention_hallucination_rate** (lower is better): % of asserted objects that are absent (CHAIR_i / "Ment.").
- **object_coverage** (higher is better): % of ground-truth objects asserted by the description.
- **vqa_accuracy** (higher is better): VQAv2 accuracy with the official answer normalisation.

## Failure rules

A run is scored as failed, receiving the worst value of every metric (100 for the hallucination rates, 0 for coverage and accuracy), if its prediction file is missing, unreadable or malformed, contains duplicate ids, misses any input of the split, or violates a capability floor:

- object coverage below 0.9 x the frozen baseline's coverage on the same split;
- VQA accuracy more than 5 points below the frozen baseline's.

The comparison baseline (unaligned LLaVA-v1.5-7B, greedy) is a frozen, evaluator-owned prediction file; it is reported as `baseline`.

## Reference points

On the paper's Object HalBench (300 images), unaligned LLaVA-v1.5-7B has 52.7 response / 28.0 mention hallucination and 78.5 VQAv2 accuracy; the paper's best LLaVA-v1.5-7B system reports 4.3 / 2.6 and 78.4. This environment's own baseline and reference values are listed in `scoring/reference.json`.
