# Progress

## Current state (iteration 11, hardening)

The gym was rebuilt on the paper's own setting. The earlier synthetic 12-object task was retired: its model input was the gold object vector, so enumerating its 781 scenes solved the test, and its test answers were shipped in `data/` (see `REDTEAM.md`, `HARDENING.md`).

- **Model**: LLaVA-v1.5-7B (HF port `llava-hf/llava-1.5-7b-hf`, pinned revision), LLaVA-1.5 square padding and vicuna_v1 template.
- **Method** (`method/sentinel/`): SENTINEL's in-domain candidate sampling, object extraction (spaCy + POS + WordNet lexicographer filters), two-detector cross-check (GroundingDINO + OWLv2), uncertain-object ignoring, context-coherent positive selection, iterative contextual bootstrapping, and C-DPO with LoRA at the paper's Table 10 hyper-parameters.
- **Data**: 8,663 preference pairs from 2,400 unlabelled COCO train2017 images (paper: 8.6K pairs from ~4K Visual Genome images); 3.6 pairs per image, 5.0 bootstrapping steps per image.
- **Evaluation**: CHAIR / Object HalBench protocol over the 80 COCO categories on 300 dev + 2,000 held-out val2017 images with eight description prompts; object coverage; VQAv2 accuracy on 3 questions per image. Evaluator-owned frozen baseline, failure floors, worst values for failed runs.

## Results (hidden test, 2,000 images)

| Run | Resp. | Ment. | Coverage | VQA |
|---|---|---|---|---|
| LLaVA-v1.5-7B baseline (paper Object HalBench: 52.7 / 28.0) | 53.45 | 26.50 | 80.58 | 77.55 |
| SENTINEL reference (paper: 4.3 / 2.6) | 42.45 | 20.93 | 77.70 | 77.57 |
| SENTINEL reference, seed 43 | 42.70 | 20.72 | 78.10 | 77.48 |

The reproduction reduces response-level hallucination by 11.0 points (95% paired bootstrap CI 9.2 to 12.9; a second training seed moves it by 0.25) and mention-level hallucination by 5.6 points while keeping VQA accuracy and 96% of object coverage. The direction matches the paper; the size (21% relative) is far below the paper's 92%. Stronger optimisation on the same data reduces hallucination further (see `HARDENING.md`), so the gap is in training strength, not in the procedure.

## Deviations from the paper

- Visual Genome images -> unlabelled COCO train2017 images (available locally; the evaluation uses disjoint COCO val2017 images).
- YOLO-World -> OWLv2 as the second open-vocabulary detector (the paper notes any open-world detector can be used; YOLO-World needs runtime downloads unavailable on compute nodes).
- FACTUAL scene-graph parser -> spaCy dependency parse (noun heads with compounds), followed by the paper's POS and WordNet filters.
- Number of sampled candidates (n = 8), sampling temperature (1.0) and detector thresholds (GroundingDINO 0.35 box threshold, OWLv2 0.10) are not given in the paper; standard defaults were used.
- Bootstrapping is capped at 10 sentences per image (mean 5.0 reached).
- Object HalBench's 300 COCO val2014 images and ChatGPT-assisted object extraction -> 2,000 held-out val2017 images with the original rule-based CHAIR extraction, objects counted once per response, negated mentions ignored.
- AMBER, HallusionBench, TextVQA, ScienceQA, MM-Vet, the GPT language-quality judge and the 13B model are not run.

## Engineering notes

- transformers 4.47.1's LLaVA "legacy" path is triggered at every decode step (one new token holds fewer than 576 image tokens) and then attends to left padding; the runner expands the image placeholder itself and sets `config.image_seq_length = 0`. A first run affected by this was discarded.
- Jobs run with one CPU core on GPUs shared with other jobs: libraries are single-threaded, detector preprocessing runs on the GPU, GPUs are chosen per stage by free memory, and generation/training fall back to smaller batches on OOM.
- Reference artifacts: `method/sentinel/reference_preference_summary.json`, `method/sentinel/reference_train_log.json`; the hardening job scripts are in `redteam/hardening_v2/`.

## Handoff (end of iteration 11; GPU budget spent)

- **Final gym.**
  - Evaluators: `eval/train`, `eval/test`.
  - Frozen baseline: `eval/*/fixed_baseline_predictions.jsonl`.
  - Reference scores: `scoring/scores*.json`, `scoring/reference.json`.
  - Hardening evidence: `HARDENING.md`, `redteam/hardening_v2/`.
- **To reproduce the reference:**
  1. `bash scripts/download.sh` (login node).
  2. `bash scripts/run.sh` (2 GPUs; about 4-5 h on shared A100s).
- **Most useful next steps:**
  - Run the paper's ablations: context-agnostic positives, no ICB, uncertain-as-factual/hallucinated, standard DPO.
  - Add a second general-capability benchmark.
  - Investigate why the paper's learning rate gives a 21% rather than 92% hallucination reduction at 8.6K pairs (training-strength sweep in `HARDENING.md`).
