# Evaluation Targets

## Primary: synthetic_object_halbench

Tests whether a generated response mentions only objects that are present in the image while still naming useful visible objects.
- **Metric**: response_hallucination_rate (lower is better)
- **Constraint metrics**: object_recall and qa_accuracy (higher is better)
- **How to run evaluation**: run `scripts/evaluate.sh outputs` after a method writes prediction JSONL files into `outputs/`.
- **Reference values**: `scoring/reference.json` records the validated scaled synthetic benchmark targets. Full paper Object HalBench numbers are documented in `briefing/evaluation.md` but are not used for automated comparison because this environment uses a smaller substitute benchmark.
