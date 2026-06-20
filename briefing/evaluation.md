# Evaluation

## Metrics

- **Object HalBench response-level hallucination rate:** percentage of generated responses containing at least one hallucinated object. Lower is better.
- **Object HalBench mention-level hallucination rate:** percentage of object mentions that are hallucinated. Lower is better.
- **AMBER generative metrics:** CHAIR, hallucination score, and cognitive hallucination score. Lower is better for all three.
- **AMBER discriminative metrics:** F1 scores for existence, attribute, state, number, action, relation, and overall hallucination discrimination. Higher is better.
- **HallusionBench question accuracy:** accuracy on hallucination/illusion-sensitive multimodal questions. Higher is better.
- **General capability metrics:** VQAv2 accuracy, TextVQA accuracy, ScienceQA image-subset accuracy, and MM-Vet overall score. Higher is better.
- **Language quality metrics:** GPT-judge scores for grammatical correctness, fluency, detailedness, and choice of words on image descriptions. Higher is better.

## Baselines and Targets

The main reference point is the unmodified base model under the same evaluation protocol. For LLaVA-v1.5-7B, the paper reports baseline Object HalBench 52.7 response-level and 28.0 mention-level hallucination, AMBER CHAIR 8.4, AMBER Hal. 35.5, AMBER Cog. 4.0, HallusionBench 46.86, VQAv2 78.5, TextVQA 58.2, ScienceQA image accuracy 66.8, and MM-Vet 31.0.

The paper's LLaVA-v1.5-7B target is Object HalBench 4.3 response-level and 2.6 mention-level hallucination, AMBER CHAIR 2.9, AMBER Hal. 14.6, AMBER Cog. 1.2, HallusionBench 47.56, VQAv2 78.4, TextVQA 58.2, ScienceQA image accuracy 69.2, and MM-Vet 32.6.

For LLaVA-v1.5-13B, the paper reports baseline Object HalBench 46.0 / 23.0, AMBER CHAIR 6.9, Hal. 31.9, Cog. 3.3, HallusionBench 46.43, VQAv2 80.0, TextVQA 61.2, ScienceQA image accuracy 71.6, and MM-Vet 36.0. The paper's target is Object HalBench 3.3 / 1.9, AMBER CHAIR 2.7, Hal. 11.7, Cog. 0.9, HallusionBench 46.77, VQAv2 79.9, TextVQA 61.0, ScienceQA image accuracy 72.8, and MM-Vet 36.2.

## Evaluation Protocol

Evaluate all candidate systems with the same base model size, benchmark datasets, prompts, and decoding settings used for the corresponding baseline. The paper follows LLaVA-v1.5 evaluation settings; Object HalBench uses eight diverse prompts for stability, and qualitative comparisons use greedy decoding. Hallucination benchmarks and general benchmarks should both be reported, because a system that reduces hallucination by becoming terse or less capable does not satisfy the paper's claim.

Use the benchmark's standard scoring procedure for each metric. Do not train on or tune against benchmark test answers. For MM-Vet and language-quality judging, use the same judge model and prompt format across compared systems.
