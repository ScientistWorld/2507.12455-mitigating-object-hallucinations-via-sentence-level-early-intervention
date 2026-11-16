# Evaluation Targets

All targets are scored by `scripts/evaluate.sh` / `scripts/evaluate_train.sh` on the visible development slice (300 images) and by the hidden evaluator on the held-out test slice (2,000 images), from `outputs/<method>_predictions.jsonl`.

## object_halbench_coco (weight 0.98)

Detailed image descriptions from LLaVA-v1.5-7B (eight prompt phrasings), checked against human MSCOCO object annotations.

- **Primary metric**: `response_hallucination_rate` (lower is better): % of descriptions asserting an absent object.
- **Benefit metric**: `mention_hallucination_rate` (lower is better): % of asserted objects that are absent.
- **Constraint metric**: `object_coverage` (higher is better, coefficient 1.5): % of present objects that are asserted. Runs below 0.9x the frozen baseline's coverage are scored as failed.
- Reference values on the hidden test (`scoring/reference.json`): unaligned baseline 53.45 / 26.50 / 80.58; reference method 42.45 / 20.93 / 77.70 (a seed-43 replicate: 42.70 / 20.72 / 78.10).
- Paper (Object HalBench, LLaVA-v1.5-7B, not on this scale): 52.7 / 28.0 -> 4.3 / 2.6.

## vqav2_general (weight 0.02)

- **Constraint metric**: `vqa_accuracy` (higher is better): VQAv2 accuracy on short answers for the same held-out images. Runs more than 5 points below the frozen baseline are scored as failed.
- Reference values on the hidden test: baseline 77.55; reference method 77.57. Paper (full VQAv2): 78.5 -> 78.4.

## Target summary

| Kind | Experiment / metric | Direction | Entry in `scores.json` |
|---|---|---|---|
| **Primary** | `object_halbench_coco.response_hallucination_rate` (and `mention_hallucination_rate`) | lower | `sentinel` vs `baseline` |
| **Constraint** | `object_halbench_coco.object_coverage` (floor 0.9x baseline, coefficient 1.5) | higher | `sentinel` vs `baseline` |
| **Constraint** | `vqav2_general.vqa_accuracy` (floor baseline - 5) | higher | `sentinel` vs `baseline` |
| **Ablation** | Remove the preference alignment entirely: the unaligned model (`baseline`) hallucinates in 53.45% of descriptions vs 42.45% for the reference | higher is worse | `baseline` |

A method that ignores the constraints by describing less (truncating, going terse or empty) or by giving up question answering falls below a floor and is scored as failed. Further data-scale and training-strength variants (half the preference data, 4.3K pairs, stronger learning rates) were run on the hidden test and are reported in `HARDENING.md`, not in `scores.json`.

## How to run evaluation

```bash
bash scripts/evaluate.sh          # development slice -> scoring/scores.json
bash scripts/evaluate_train.sh    # development slice -> scoring/scores_train.json
bash scripts/evaluate_test.sh     # held-out slice    -> scoring/scores_test.json (hidden evaluator)
```

All three read `outputs/<method>_predictions.jsonl`; see `briefing/evaluation.md` for the format.

## Not measured

The paper's AMBER, HallusionBench, TextVQA, ScienceQA, MM-Vet and GPT-judged language-quality results and the 13B model are not measured here; their reported values are kept for reference under `_paper_table1` in `scoring/reference.json`.
