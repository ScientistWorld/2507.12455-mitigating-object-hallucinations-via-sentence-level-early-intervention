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
