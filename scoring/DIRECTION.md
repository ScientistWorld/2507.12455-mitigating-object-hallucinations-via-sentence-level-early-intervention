# Research Direction

## Research Problem

This benchmark studies object hallucination mitigation for image-conditioned language generation: the system should describe visible objects without inventing absent ones.

## Core Contribution Scope

Work should stay within training-time or alignment-time methods that improve visual grounding of generated object mentions. Improvements should preserve useful object coverage, not simply suppress mentions.

## Approach Scope

Future agents may change candidate sampling, object validation signals, preference-pair construction, contextual bootstrapping, or the preference objective. They may also improve the compact model so long as comparisons remain within the provided benchmark resources.

## Out of Bounds

- Replacing the task with retrieval from evaluation labels.
- Using the evaluation split for training, filtering, or prompt tuning.
- Improving hallucination scores by emitting generic captions with no object content.
- Changing the evaluator or prediction schema to hide hallucinated mentions.
