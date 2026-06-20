# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-20 00:00] - none
- Read the paper and identified SENTINEL's required computational steps: in-domain sentence candidate sampling, object extraction, two-detector validation, uncertain filtering, iterative context bootstrapping, and context-aware DPO.
- Implemented a scaled self-contained benchmark and method pipeline, but have not yet run it inside the compute environment, so no execution milestone is claimed.

### [2026-06-20 04:31] - none
- Audited the inherited workspace and fixed evaluator/reference scale mismatch, independent QA scoring, prompt-prefix decoding during generation, and non-sampled fallback preference pairs.
- Kept milestone at none because no compute job has completed and the implementation remains a scaled synthetic SENTINEL reproduction rather than the full LLaVA/Visual Genome detector stack.

### [2026-06-20 04:37] - none
- Reviewed failed job `f9b6e519-0ae`; it reached data generation and language-prior output but failed before method training due to a top-level package import path issue.
- Fixed `scripts/method.sh` to run `python -m method.train`; no execution milestone is claimed yet because no SENTINEL method output or `scoring/scores.json` exists.

### [2026-06-20 04:44] - method_runs
- Reviewed successful job `1f76b35f-f0d`; the scaled SENTINEL pipeline trained, sampled in-domain preference pairs, ran C-DPO, generated predictions, and produced `scoring/scores.json`.
- Did not claim `core_claim`: both baseline and SENTINEL collapsed to one repeated object, yielding poor recall and only a tiny hallucination-rate difference.
- Patched the compact VLM with a learned visual object-presence head, model-based object-token decoding bias, and corrected variable-length C-DPO log-probability scoring before the next run.

### [2026-06-20 04:51] - method_runs
- Reviewed successful job `83e8b9d2-ea1`; the grounded run completed and improved object grounding, but it made the baseline hallucination rate 0.0, so it no longer tested SENTINEL's hallucination-reduction claim.
- Kept milestone at `method_runs` because SENTINEL did not outperform the baseline on the benefit/constraint trade-off.
- Rebalanced the next run by lowering the shared grounding bias, strengthening repeated-object suppression, and increasing stochastic in-domain candidate sampling for C-DPO.

## Stop Justification
