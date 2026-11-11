# Research Direction

## Research Problem

Object hallucination in detailed image descriptions from a multimodal LLM: the model should describe what is in the image without asserting absent objects, and stay generally capable.

## Core Contribution Scope

Improve how the fixed base model is adapted so that its own greedy descriptions assert fewer absent objects while still naming the objects that are present and keeping its question answering ability, without human annotation.

## Approach Scope

Any training-time technique is in scope: how training signal is generated from the model's own outputs and the provided detectors, how it is filtered, which objective is optimised, how much data is built, and how the model is adapted. Changes must stay within the constraints in `CONSTRAINTS.md`.

## Out of Bounds

- Using human annotations (MSCOCO, VQAv2, Visual Genome or any other labels) of any image.
- Inference-time machinery: detector filtering, best-of-n selection, rewriting by another model.
- Swapping the base model for a different or larger one.
- Gaining on hallucination by describing less: terse, empty or negated descriptions.
- Editing the evaluator, its answers, the frozen baseline or the evaluation inputs.
