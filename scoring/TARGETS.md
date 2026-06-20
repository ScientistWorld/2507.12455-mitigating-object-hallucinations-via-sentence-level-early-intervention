# Evaluation Targets

## Primary: synthetic_object_halbench

Tests whether a generated response mentions only objects that are present in the image while still naming useful visible objects.
- **Metric**: response_hallucination_rate (lower is better)
- **Constraint metrics**: object_recall and qa_accuracy (higher is better)
- **How to run evaluation**: run `scripts/evaluate.sh outputs` after a method writes prediction JSONL files into `outputs/`.
