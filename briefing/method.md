# Paper's Method

## Key Contribution

SENTINEL is a sentence-level preference learning framework for mitigating object hallucinations in MLLMs. Its central idea is to intervene at the earliest sentence where hallucination can appear, using in-domain model-generated continuations rather than rewritten responses from a proprietary model or human annotation.

## Approach

SENTINEL has three main components:

1. **In-domain candidate bootstrapping:** For each image, prompt, and current context, the current MLLM samples multiple candidate next sentences. Generation stops at sentence completion so each candidate can be judged at sentence granularity.
2. **Object extraction and validation:** Candidate sentences are parsed into scene-graph triplets; noun entities from subjects and objects are filtered with POS and WordNet-style concrete noun checks. Candidate objects are validated with two open-vocabulary detectors, Grounding DINO and YOLO World. Objects are treated as factual if both detectors confirm presence, hallucinated if both reject presence, and uncertain otherwise. Sentences with hallucinated objects become negative samples; sentences containing only factual objects can become positive samples. Uncertain objects are ignored.
3. **Context-aware preference learning:** Preference pairs have the form `(image, prompt, context, positive next sentence, negative next sentence)`. The preferred sentence is selected from non-hallucinated, context-coherent continuations. The context is iteratively extended with a verified non-hallucinated sentence, producing training pairs across progressively longer hallucination-free contexts. Training uses a context-aware DPO objective where the input is image, prompt, and context, but the loss is computed on the positive/negative continuation rather than the context tokens.

The paper trains LLaVA-v1.5-7B and LLaVA-v1.5-13B with LoRA on Visual Genome images. It reports one epoch of training, 8.6K preference samples for 7B and 7.0K samples for 13B, AdamW optimization, LoRA rank 128, LoRA alpha 256, beta 0.1, global batch size 64, and ZeRO stage 2.

## Main Claims

- **Primary claim:** Sentence-level early intervention with in-domain context-aware preference learning sharply reduces object hallucination compared with the original model and recent mitigation methods.
- **Constraint claim:** The hallucination reduction does not sacrifice general multimodal capability; performance is preserved or improved on VQAv2, TextVQA, ScienceQA, HallusionBench, and MM-Vet.
- **Mechanistic claims:** In-domain samples outperform rewritten samples; context-coherent positives outperform context-agnostic positives; hallucination-free iterative context construction improves robustness; detector cross-checking and ignoring uncertain objects improve preference data quality.
- **Generality claim:** The same framework reduces hallucinations on additional MLLM families, including LLaVA-v1.6 and Qwen2/Qwen2.5-VL models.

## This environment's implementation

`method/sentinel/` implements the steps above on LLaVA-v1.5-7B (HF port, LLaVA-1.5 square padding and conversation template):

- `build_preferences.py`: n = 8 candidates per context sampled at temperature 1 and stopped at the first period or `</s>`; objects extracted with spaCy (noun heads + compounds) and the paper's POS and WordNet lexicographer-category filters (`objects.py`); presence cross-checked with GroundingDINO (box threshold 0.35) and OWLv2 (threshold 0.10) standing in for YOLO-World (`detectors.py`); uncertain objects ignored; y_w+ chosen among verified non-hallucinated candidates sharing an object with the context (any verified candidate for the first sentence); y_l among hallucinated candidates; the context is extended with y_w+ until `</s>`. Unlabelled COCO train2017 images; construction stops at 8.6K pairs (8,663 built from 2,400 images).
- `cdpo.py`: C-DPO (context in the input, loss only on y_w+ / y_l tokens) with a frozen reference, LoRA r128 / alpha 256 on the language model's linear layers only, AdamW lr 2e-6, no weight decay, cosine schedule, global batch 64, beta 0.1, one epoch (134 updates).
- `predict.py`: greedy decoding (512 new tokens for descriptions, 16 for VQA).
