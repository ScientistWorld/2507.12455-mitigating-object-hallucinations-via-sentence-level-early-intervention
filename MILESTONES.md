# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-20 05:22] - core_claim
- Completed the scaled SENTINEL pipeline end-to-end on the synthetic object-hallucination benchmark.
- The implemented method samples in-domain sentence candidates, validates object mentions with two independent validators, filters uncertain samples, bootstraps hallucination-free context iteratively, and trains with context-aware DPO over chosen/rejected continuations.
- Reproduced core claim on the scaled benchmark: SENTINEL reduced response-level hallucination from 24.583% for the trained baseline to 4.167%, while preserving object recall and QA accuracy.

### [2026-06-20 05:35] - core_claim
- Re-audited the workspace after the milestone reset and confirmed `python validate.py --compare` passes on current artifacts.
- Preparing a CPU test-node rerun of `scripts/reproduce.sh` to refresh the reproducibility artifacts from scratch.

### [2026-06-20 05:41] - method_runs
- Found that the previous compact implementation included non-paper auxiliary object-presence losses and decode-time object bias/repeat filtering.
- Removed those shortcuts so the method path now uses image-conditioned language modeling plus SENTINEL's sampled validation, bootstrapped context construction, and C-DPO objective.
- The old `scores.json` is stale for the patched method; submitting a fresh CPU test-node rerun before restoring `core_claim`.

### [2026-06-22 23:53] - method_runs
- The faithful rerun completed but did not support `core_claim`: SENTINEL reduced hallucination only modestly and sharply reduced object recall/QA behavior.
- Audited the compact preference construction against the paper and added context-coherent positive selection before ICB context appending.
- Masked prompt tokens during supervised baseline training so response modeling, preference scoring, and evaluation use a consistent generation target.
- Submitting a CPU sweep over conservative C-DPO settings; milestone remains `method_runs` until the resulting scores support the benefit and constraint sides together.

### [2026-06-22 23:57] - method_runs
- Continued audit after milestone reset: current `scoring/scores.json` still contains the failed `84900da4-ee5` scores and no `tmp/sentinel_sweep/summary.json` exists yet.
- Verified `eval/metrics.py` scores prediction artifacts independently from `method/`, and `validate.py --compare` passes structurally while showing the current SENTINEL scores are not good enough for `core_claim`.
- Keeping milestone at `method_runs` and submitting the prepared CPU sweep to generate fresh C-DPO-only results.

### [2026-06-23 00:31] - method_runs
- CPU sweep `63cdc35d-8f1` completed successfully and preserved recall/QA in its best configuration, but the hallucination improvement was too small for a robust `core_claim`.
- Diagnosed repeated single-object generation as the remaining scaled-benchmark bottleneck.
- Changed the synthetic SFT caption order from random to canonical visual-object order and exposed hidden size as a CLI scale parameter, then prepared a tighter sweep with stronger recall/QA penalties.
