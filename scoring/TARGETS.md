# Evaluation Targets

## Paper Targets: paper_llava_v15_7b_main and paper_llava_v15_13b_main

These entries in `scoring/reference.json` preserve the paper-reported Table 1 targets for full-scale LLaVA-v1.5 runs. They document the original benchmark goals: reduce Object HalBench response/mention hallucination and AMBER hallucination metrics while maintaining HallusionBench, VQAv2, TextVQA, ScienceQA, and MM-Vet performance.

The current runnable environment does not reproduce these full-scale targets directly because it uses a compact synthetic substitute instead of the full paper-scale model, image data sources, external object-checking resources, and public hallucination/general benchmark suite.

## Runnable Target: synthetic_object_halbench

Tests whether a generated response mentions only objects that are present in the image while still naming useful visible objects.
- **Metric**: response_hallucination_rate (lower is better)
- **Constraint metrics**: object_recall and qa_accuracy (higher is better)
- **How to run evaluation**: run `scripts/evaluate.sh outputs` after a method writes prediction JSONL files into `outputs/`.
- **Reference values**: `scoring/reference.json` records validated scaled synthetic benchmark targets for automated comparison, and separately records the paper's full LLaVA-v1.5 Table 1 targets.
