# Research Direction

## Research Problem

This benchmark studies object hallucination mitigation for image-conditioned language generation: the system should describe visible objects without inventing absent ones.

## Core Contribution Scope

Work should improve the visual grounding of generated object mentions. A stronger system should mention fewer absent objects while preserving useful coverage of objects that are actually present.

## Approach Scope

Future agents may change the model, training data usage, optimization strategy, decoding policy, or other grounding technique, provided the comparison stays within the provided benchmark resources and the same output format. Improvements should come from better image-conditioned generation, not from using labels from the evaluation split or changing the evaluator.

## Out of Bounds

- Replacing the task with retrieval from evaluation labels.
- Using the evaluation split for training, filtering, or prompt tuning.
- Improving hallucination scores by emitting generic captions with no object content.
- Changing the evaluator or prediction schema to hide hallucinated mentions.
