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

Job `1f76b35f-f0d` completed the scaled SENTINEL path and validates `method_runs`, but not `core_claim`. Reproduced scores were:
- `language_prior`: response hallucination 98.75, mention hallucination 71.667, object recall 27.984, QA accuracy 47.312.
- `baseline`: response hallucination 80.000, mention hallucination 80.000, object recall 6.584, QA accuracy 40.447.
- `sentinel`: response hallucination 77.917, mention hallucination 77.917, object recall 7.270, QA accuracy 40.778.

The failure mode is generation collapse: baseline and SENTINEL mostly emit one repeated object, so the evaluator has little useful object coverage to score. After this job, the method was patched to train an auxiliary visual object-presence head, use the learned object scores as a decoding bias for object tokens, prevent repeated object mentions, and fix padded-prefix scoring in the C-DPO sequence log-probability computation.

Job `83e8b9d2-ea1` completed after those fixes, but still does not support `core_claim`. The shared grounding bias made both trained models non-hallucinatory:
- `baseline`: response hallucination 0.000, mention hallucination 0.000, object recall 45.542, QA accuracy 67.163.
- `sentinel`: response hallucination 0.000, mention hallucination 0.000, object recall 44.307, QA accuracy 66.419.

This run improved over the previous collapse but removed the benefit side of the benchmark: the baseline no longer hallucinated, and only 9 in-domain preference pairs were sampled. The next run lowers the shared object decoding bias, strengthens duplicate-object suppression at both color and shape selection, and increases stochastic sampling to produce a larger C-DPO preference set.

Job `7d02312f-289` was cancelled after 20 minutes with no stage-level output. Partial artifacts showed `baseline.pt`, baseline predictions, language-prior predictions, and 1,799 sampled preference pairs, but no SENTINEL checkpoint, SENTINEL predictions, or `scores.json`. The run had likely entered the unbatched C-DPO loop, where each preference example triggered separate policy/reference forward passes. The implementation now vectorizes padded sequence log-probability scoring across each batch, reduces the default smoke workload to 700 images x 16 samples, and prints flushed progress at each major stage and C-DPO epoch.

## Remaining

Run the corrected compute job, inspect `scoring/scores.json`, compare the baseline and SENTINEL outputs, and update to `core_claim` only if SENTINEL materially reduces hallucination while maintaining non-trivial object recall and QA accuracy on the scaled benchmark.

## Issues

The full paper setup uses LLaVA-v1.5, Visual Genome, Scene-GraphParser, GroundingDINO, YOLO-World, and public hallucination benchmarks. That is larger than this initial autonomous run, so the first runnable environment uses a scaled synthetic benchmark while preserving the SENTINEL algorithmic procedure.

The current implementation is therefore not enough to claim the full paper-scale core claim. It can support only a scaled method milestone after the submitted job proves the pipeline runs and produces real scores.

The current highest milestone is `method_runs`: the paper's algorithmic steps execute end-to-end in the scaled environment, but the latest completed scores do not yet support the core hallucination-mitigation claim.

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
