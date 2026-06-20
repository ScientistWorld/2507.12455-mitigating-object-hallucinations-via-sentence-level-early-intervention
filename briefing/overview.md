# Overview

- **Paper ID:** 2507.12455
- **Title:** Mitigating Object Hallucinations via Sentence-Level Early Intervention
- **Domain:** Multimodal Large Language Models / Vision-Language Hallucination Mitigation
- **TL;DR:** SENTINEL reduces object hallucinations in MLLMs by building in-domain sentence-level preference pairs and training with context-aware preference optimization, while preserving general visual-language ability.

## Short Summary

The paper addresses object hallucination in multimodal large language models, where generated descriptions or answers mention objects not present in the image. The authors argue that hallucinations often appear early in generation and then propagate, so reducing the first occurrence can reduce later errors. SENTINEL constructs preference data from the model's own generated sentences, labels object mentions using open-vocabulary detector cross-checking, and trains on sentence-level positive/negative continuations conditioned on prior context. On LLaVA-v1.5, the reported results show large reductions on Object HalBench and AMBER while maintaining or improving VQAv2, TextVQA, ScienceQA, HallusionBench, and MM-Vet performance.

## Key Results

- On LLaVA-v1.5-7B, Object HalBench response-level hallucination drops from 52.7 to 4.3 and mention-level hallucination drops from 28.0 to 2.6.
- On LLaVA-v1.5-7B, AMBER generative hallucination improves from CHAIR 8.4 / Hal. 35.5 / Cog. 4.0 to 2.9 / 14.6 / 1.2.
- General capability is preserved: LLaVA-v1.5-7B with SENTINEL reports VQAv2 78.4, TextVQA 58.2, ScienceQA image accuracy 69.2, and MM-Vet 32.6.
