# Progress

## What Works

Briefing documentation is filled for SENTINEL from `paper/2507.12455.md`. The problem and evaluation briefings are method-agnostic; the method briefing captures the paper's algorithmic steps.

Implemented a scaled, self-contained reproduction path:
- `data/synthetic_scenes.py` creates image-object scenes with captions and object-existence questions.
- `method/sentinel/` implements in-domain candidate sampling, object extraction, two-validator cross-checking, uncertain filtering, iterative context bootstrapping, and C-DPO with context excluded from the target loss.
- `baseline/majority_language_prior.py` provides a language-prior baseline.
- `eval/metrics.py` evaluates generated artifacts without importing from `method/`.
- `scripts/reproduce.sh`, `scripts/method.sh`, `scripts/baseline.sh`, and `scripts/evaluate.sh` wire the environment together.

## Results

No reproduced training result is claimed yet. Local lightweight validation passed for syntax, import separation, and baseline/evaluator JSON output. The actual method training path requires torch and will run in the submitted compute job.

Audit fixes on 2026-06-20:
- Corrected `scoring/reference.json` hallucination/recall/QA values to use the same percentage scale as `eval/metrics.py`.
- Changed QA scoring so `eval/` derives answers independently from generated captions and gold questions rather than trusting method-supplied `qa_predictions`.
- Removed oracle-fabricated fallback preference pairs; SENTINEL preference pairs now come only from sampled model continuations.
- Fixed generation to decode only newly generated continuation tokens, not the prompt prefix.
- Updated the prediction-schema constraints and ignored `tmp/` throwaway audit files.

Job `f9b6e519-0ae` failed before method training. It generated the synthetic data and `outputs/language_prior_predictions.jsonl`, then stopped with `ModuleNotFoundError: No module named 'method'` because `scripts/method.sh` executed `python method/train.py`. Fixed by invoking the method as a module with `python -m method.train`.

## Remaining

Run the compute job, inspect `scoring/scores.json`, compare the baseline and SENTINEL outputs, and update the milestone only if the paper's core claim is supported on the scaled benchmark.

## Issues

The full paper setup uses LLaVA-v1.5, Visual Genome, Scene-GraphParser, GroundingDINO, YOLO-World, and public hallucination benchmarks. That is larger than this initial autonomous run, so the first runnable environment uses a scaled synthetic benchmark while preserving the SENTINEL algorithmic procedure.

The current implementation is therefore not enough to claim the full paper-scale core claim. It can support only a scaled method milestone after the submitted job proves the pipeline runs and produces real scores.

## Deviations from Paper

- Paper: LLaVA-v1.5-7B/13B on Visual Genome images with Scene-GraphParser, GroundingDINO, and YOLO-World. This reproduction: a compact image-conditioned language model on deterministic synthetic object scenes with two independent symbolic visual validators. Reason: keep the environment self-contained and fit a first end-to-end run within the available budget while preserving SENTINEL's computational steps.
- Paper: full Object HalBench, AMBER, HallusionBench, and general VQA benchmarks. This reproduction: a scaled synthetic object hallucination benchmark with response/mention hallucination, object recall, and QA accuracy. Reason: avoid large benchmark/model downloads for the first core-claim test and keep evaluation reusable from generated artifacts.

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

     The test: could someone reading your code identify which algorithm from
     the paper you implemented? If not, you replaced the method. -->
