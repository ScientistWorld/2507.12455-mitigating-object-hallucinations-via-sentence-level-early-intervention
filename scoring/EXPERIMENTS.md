# Experiment Overview

## paper_llava_v15_7b_main

Records the paper's full-scale LLaVA-v1.5-7B comparison across hallucination and general-capability benchmarks. This is a reference-only target in the current compact environment.

## paper_llava_v15_13b_main

Records the paper's full-scale LLaVA-v1.5-13B comparison across hallucination and general-capability benchmarks. This is a reference-only target in the current compact environment.

## synthetic_object_halbench

Tests whether the reproduced system reduces object hallucination on image-conditioned object descriptions while preserving useful coverage of visible objects and simple object-presence question answering. This experiment is a compact, method-agnostic analogue of the paper's Object HalBench-style headline claim and is the only experiment currently written by `scripts/evaluate.sh`.

The reference values for this experiment are scaled synthetic benchmark targets from the validated reproduction run. They intentionally do not copy the paper's full Object HalBench values, because those were measured on a different model and benchmark scale; the full paper values are recorded in the `paper_llava_v15_*` entries.
