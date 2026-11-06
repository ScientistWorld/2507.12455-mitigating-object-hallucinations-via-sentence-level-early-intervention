# Mitigating Object Hallucinations via Sentence-Level Early Intervention

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2507.12455"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2507.12455-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2507.12455](https://arxiv.org/abs/2507.12455) |
| **Domain** | Multimodal Large Language Models / Vision-Language Hallucination Mitigation |
| **TL;DR** | SENTINEL reduces object hallucinations in MLLMs by building in-domain sentence-level preference pairs and training with context-aware preference optimization, while preserving general visual-language ability. |

## Summary

The paper addresses object hallucination in multimodal large language models, where generated descriptions or answers mention objects not present in the image. The authors argue that hallucinations often appear early in generation and then propagate, so reducing the first occurrence can reduce later errors. SENTINEL constructs preference data from the model's own generated sentences, labels object mentions using open-vocabulary detector cross-checking, and trains on sentence-level positive/negative continuations conditioned on prior context. On LLaVA-v1.5, the reported results show large reductions on Object HalBench and AMBER while maintaining or improving VQAv2, TextVQA, ScienceQA, HallusionBench, and MM-Vet performance.

## Problem

How can a multimodal large language model generate image-grounded descriptions and answers with substantially fewer object hallucinations, without losing its general visual question answering, reasoning, OCR, and language generation capabilities?

Object hallucination means the model mentions an object or visual fact that is not supported by the image. The benchmark problem is not only to reduce these hallucinated mentions, but to do so under the same base-model scale and evaluation conditions used for standard MLLM comparisons.

## The paper's method

SENTINEL is a sentence-level preference learning framework for mitigating object hallucinations in MLLMs. Its central idea is to intervene at the earliest sentence where hallucination can appear, using in-domain model-generated continuations rather than rewritten responses from a proprietary model or human annotation.

## Key results of the paper

- On LLaVA-v1.5-7B, Object HalBench response-level hallucination drops from 52.7 to 4.3 and mention-level hallucination drops from 28.0 to 2.6.
- On LLaVA-v1.5-7B, AMBER generative hallucination improves from CHAIR 8.4 / Hal. 35.5 / Cog. 4.0 to 2.9 / 14.6 / 1.2.
- General capability is preserved: LLaVA-v1.5-7B with SENTINEL reports VQAv2 78.4, TextVQA 58.2, ScienceQA image accuracy 69.2, and MM-Vet 32.6.

## Run it

```bash
pip install sciworld
sw research 2507.12455 --mode innovate --agent claude-code
sw evaluate 2507.12455
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

