# Hardening report (iteration 11)

**Verdict.** The gym now measures the paper's problem itself, and on its hidden test better results score higher and worse results lower. The redteam showed the old synthetic task could not be fixed by patching. It handed the model the answer as its input (enumerating 781 scenes scored perfectly) and shipped the test answers in `data/`. So the gym was rebuilt on the paper's own setting:
- **Model:** LLaVA-v1.5-7B.
- **Method:** SENTINEL's in-domain sentence sampling, object extraction, two-detector cross-check, context-coherent positives, iterative contextual bootstrapping and C-DPO with LoRA, at the paper's Table 10 settings and its 8.6K-pair data scale.
- **Test:** an Object HalBench-style CHAIR test on 2,000 held-out MSCOCO images, with object coverage and VQAv2 accuracy as constraints.

All ranks below are on the hidden test (`bash scripts/evaluate_test.sh`, no arguments). Every candidate was run through it by `redteam/hardening_v2/cases.py`. Ranks use the stated rule: weight × |coefficient| × improvement in the better direction ÷ the metric's spread. The spread is the population std over the nine honest runs (frozen baseline plus every trained variant that passed). Ranks under the alternative "all non-failed entries" cohort are in `redteam/hardening_v2/summary.json`. Paired 95% bootstrap intervals resample test images, and every method sees the same inputs.

Metric notation: **R** response hallucination % (lower is better), **M** mention hallucination % (lower is better), **C** object coverage % (higher is better), **V** VQAv2 accuracy % (higher is better).

## Do better results score higher?

**Before** (synthetic gym, `REDTEAM.md`):
- More training scored +0.78 and clean training +0.80.
- Removing alignment cost only −0.16, less than the 0.65 seed range.
- One of three reference seeds made hallucination worse than its own baseline.
- Brute-force enumeration and label copying were perfect (+0.81).

**After** (2,000-image hidden test; the reference is the paper's method at the paper's settings):

| Run (all LLaVA-v1.5-7B) | R | M | C | V | Rank vs reference | ΔR vs reference [95% CI] |
|---|---|---|---|---|---|---|
| C-DPO + NLL on chosen, lr 8e-6 *(better variant)* | 28.35 | 13.86 | 73.49 | 77.81 | **+0.695** | −14.1 [−16.1, −12.2] |
| C-DPO + NLL on chosen, lr 5e-6 *(better variant)* | 34.45 | 16.61 | 75.31 | 77.53 | +0.430 | −8.0 [−9.8, −6.3] |
| C-DPO, lr 3e-6 *(stronger optimisation)* | 31.70 | 15.81 | 74.28 | 77.51 | +0.297 | −10.8 [−12.5, −9.0] |
| Reference, training seed 43 | 42.70 | 20.72 | 78.10 | 77.48 | +0.238 | +0.25 [−1.0, 1.6] |
| **Reference** (8,663 pairs, lr 2e-6, 1 epoch) | **42.45** | **20.93** | **77.70** | **77.57** | **0** | – |
| Earlier 4.3K-pair run, seed 43 *(weaker)* | 50.50 | 25.03 | 80.07 | 77.56 | −0.406 | +8.1 [6.3, 9.8] |
| Half the pairs, 1 epoch *(weaker)* | 49.55 | 24.88 | 79.51 | 77.47 | −0.614 | +7.1 [5.4, 8.8] |
| Unaligned LLaVA-v1.5-7B *(frozen baseline)* | 53.45 | 26.50 | 80.58 | 77.55 | −0.760 | +11.0 [9.2, 12.9] |
| Earlier 4.3K-pair run, seed 42 *(weaker)* | 50.90 | 25.31 | 79.38 | 77.49 | −0.932 | +8.5 [6.7, 10.2] |
| C-DPO, lr 4e-6 | 22.90 | 11.30 | 70.10 | 77.65 | **failed** (coverage < 72.52) | −19.6 [−21.6, −17.5] |
| C-DPO, lr 5e-6 | 16.10 | 7.89 | 65.44 | 77.48 | **failed** (coverage < 72.52) | −26.4 [−28.4, −24.3] |

- **Stronger alignment scores higher; less data or no alignment scores lower.**
  - The best variant beats the reference by +0.70, about 3× the seed replicate's +0.24.
  - Its hallucination reduction is about 56× the seed difference in R.
  - All three better variants beat the reference on R and M with intervals far from 0.
- **The two failed variants trade too much coverage.** Their descriptions name 13–19% fewer present objects, and coverage is the paper's "without sacrificing detail" constraint.
- **The 4.3K-pair seed-42 run ranks just below the baseline.** It trades 1.2 coverage points for 2.5 R points, about the same rate as truncating descriptions (see the coefficient section).
- **"Before" vs "after" are different tasks.** The variants before were synthetic-data interventions; the variants after are real LLaVA trainings. The comparison is of the gym's behaviour, not of the numbers.

## The reproduction and the paper's claim

| | R | M | C | V |
|---|---|---|---|---|
| Paper, LLaVA-v1.5-7B baseline → SENTINEL (Object HalBench, VQAv2) | 52.7 → 4.3 | 28.0 → 2.6 | – | 78.5 → 78.4 |
| This gym, baseline → reference (2,000 held-out images) | 53.45 → 42.45 | 26.50 → 20.93 | 80.58 → 77.70 | 77.55 → 77.57 |

- **Matches the paper:** the baseline reproduces the paper's Object HalBench baseline closely, and the direction of the effect is reproduced: fewer hallucinations, VQA unchanged.
- **Above seed noise:** the reference's −11.0 R (CI −12.9 to −9.2) and −5.6 M are far beyond the seed replicate's 0.25 / 0.21.
- **Smaller than the paper:** a 21% relative reduction against the paper's 92%.
  - The same pipeline with stronger optimisation reaches R 16–28. So the gap lies in training strength, not in the procedure.
  - Those stronger runs give up coverage. I did not change the paper's settings to close the gap.
- **Earlier departures restored:**
  - The previous reproduction's simplifications are gone: substring parser, gold-reading validators, rejecting a whole sentence on any uncertain object, novelty-biased positives, a fixed two context steps, full fine-tuning with weight decay.
  - The paper's 8.6K-pair scale and its 134 updates are used.
- **Remaining substitutions:** see `PROGRESS.md` (COCO train images instead of Visual Genome, OWLv2 instead of YOLO-World, spaCy instead of the FACTUAL parser).

## Major shortcuts: fix and score before → after

"Tie" means exactly the reference's metrics. "Failed" means worst values in every metric of every experiment (R = M = 100, C = V = 0), which ranks far below any honest run.

| Shortcut (`redteam/hardening_v2/cases/<name>`) | Before | Fix | After |
|---|---|---|---|
| Copy test answers shipped in `data/` (`copy_labels`) | **+0.81, perfect** | Answers live only in `eval/test/gold.json`. `data/` holds inputs only. | Not possible from `data/`. Copying COCO's public human captions: **failed** (C 65.18 < floor) |
| Regenerate answers from the public generator (`rebuild_seed`) | **+0.81, perfect** | No generator: answers are human annotations, built once by `eval/test/build_splits.py` | Not applicable |
| Replace the gold file (`gold_replacement`) | **+0.95, every bound** | Gold and the baseline are evaluator-owned. `data/eval_inputs.jsonl` is never read for scoring. | Editing the inputs file: no effect (tie, candidate unchanged) |
| Brute-force enumeration of answers from the input | **+0.81, perfect** | Real images; the answer is not in the input | Analogue = inference-time detector filtering (`posthoc_filter.py`): baseline + filter **−0.52**; reference + filter **failed** (C 71.61; R 26.65) |
| Say less: truncate descriptions (new) | – | Coverage floor 0.9× baseline; coverage coefficient 1.5 | First 3 or 4 sentences: **failed**. First 5 sentences of the reference: −0.59; of the baseline: −1.33. First sentence only: **failed**. |
| Empty descriptions | −0.49 (+0.36 in a two-member cohort) | Coverage floor | **failed** |
| Deny every object (`polarity_flip`) | tie | Negated mentions do not count; coverage floor | **failed** (C 2.01) |
| Omit methods from the train list (`manifest_omission`) | tie (failed methods silently dropped) | Test scores every file on disk, every train-listed method and the primary slot `sentinel` | Blank `sentinel`: **failed**; retained copy: tie |
| Missing / empty / null-prediction / duplicate-id / dev-only file | Crash with stale scoreboard, or no entry | Strict schema; every test id required; worst values written | **failed** in all five |
| Pre-written `scores_test.json` | −1.0 (overwritten) | Score file deleted before scoring | Overwritten: −9.79 (all-objects candidate) |
| Method-supplied metrics, labels or flags | −1.0 (ignored) | Only `id` and `prediction` are read | Ignored: −9.79 |
| All-objects caption | −1.0 | – | −9.79 |
| Replace the comparison baseline | Accepted (distorted comparisons) | Frozen evaluator-owned baseline; `baseline_predictions.jsonl` on disk ignored | Ignored (candidate unchanged: tie) |

## Shortcuts left open, and why

1. **Using public human annotations of the test images** (COCO instances and captions, VQAv2 answers).
   - The images are public COCO val2017, and their ids appear in `data/eval_inputs.jsonl`.
   - Naive caption copying fails the coverage floor. Combining captions with instance annotations would pass and score near-perfectly.
   - No evaluator change can hide public labels of public images. `CONSTRAINTS.md` forbids any human label, and the grader must check for it.
2. **Inference-time machinery.** The test scores text, so it cannot verify that each output is one greedy pass of the adapted model. Ruled out by `CONSTRAINTS.md`; left to the grader. This covers:
   - detector filtering or object-level rewriting (Woodpecker-style),
   - best-of-n sampling against the detectors,
   - outputs written by another model.

   With the current floor and coefficients, sentence-dropping detector filters do not pay: −0.52 on the baseline, failed on the reference. A filter that edits only the offending words could. Closing this fully would need the hidden test to regenerate outputs from submitted weights, which the light hidden test excludes.
3. **Hallucinated objects outside the 80 COCO categories** (`unscored_hallucinations`: tie). This is CHAIR's closed vocabulary, shared with the paper's Object HalBench, and it gives no gain. Closing it needs open-vocabulary ground truth.
4. **Small trade-offs along the length curve.** Coverage at coefficient 1.5 makes truncation roughly rank-neutral for this cohort's spreads, and in practice slightly losing. A future cohort with very different spreads could shift that balance. The 0.9× floor bounds it either way.

## Reference score before and after

- **Before (synthetic gym):** R 55 / M 20 / C 89.47 / avg objects 3.40 / QA 91.67, on 100 synthetic scenes. Not comparable.
- **After:** R **42.45** / M **20.93** / C **77.70** / V **77.57** on 2,000 held-out images (`scoring/scores_test.json`, `scoring/reference.json`). On the original 700-image test, before it was enlarged: 41.00 / 20.80 / 77.90 / 77.25.

## Headroom

- **No scored metric is at its bound:** R 42.45 (bound 0), M 20.93 (0), C 77.70 (100), V 77.57 (100).
- **Better variants exist:** three variants beat the reference by more than the seed spread, and the paper itself reports R 4.3.
- **The brute force SENTINEL avoids does not outrank the reference:** inference-time detector filtering scores −0.52 on the baseline and fails on the reference.
- **No simple baseline does either:** the unaligned model and its truncations rank −0.76 to −2.06, or fail.
- **No new experiment was needed.** The rebuilt experiment is the paper's own and leaves wide room.

## Changes to the scoring setup, and why

- **Coverage coefficient 1.5 and floor 0.9× the frozen baseline.** The synthetic gym had recall at 0.7 and no floor; a first version of the new evaluator used 0.5 and 0.8×.
  - Hallucination concentrates late in descriptions, as the paper's own Fig. 2 shows. Under the old setting, the unaligned baseline cut to three sentences (R 22.7) outranked the reference, and the reference cut to four sentences did too.
  - With coverage at 1.5, moving along the base model's own length trade-off is roughly rank-neutral; the floor bounds the rest. Only shifting the trade-off curve pays.
  - The same setting keeps sentence-dropping detector filters from outranking the reference.
- **Experiment weights 0.8 / 0.2 → 0.98 / 0.02 (object hallucination / VQA).**
  - Honest variants differ in VQA by 0.1–0.3 points, and the rank divides by that spread. At weight 0.2, VQA noise outweighed 10-point hallucination gains: half-data and seed replicates ranked above the reference.
  - VQA is the paper's "no loss of general ability" constraint. It is now enforced mainly by the failure floor (baseline − 5 points).
- **Test enlarged from 700 to 2,000 images** (the first 700 unchanged). On 700 images, two runs differing only by seed were 0.6 rank units apart, mostly from coverage sampling noise, comparable to the best variant's margin. On 2,000 images the reference's seed replicate is 0.24 apart, and the best variant leads by 0.70.
- **Removed:** the two reference-only paper experiments (weight 0.70, never written by any run). The paper's Table 1 values are kept under `_paper_table1` in `scoring/reference.json` and the `_paper_*` result keys.

## Could not fix / limitations

- **Effect size:** the reproduction's hallucination reduction (21% relative) is much smaller than the paper's (92%) at the paper's stated hyper-parameters.
- **Benchmarks not run:** AMBER, HallusionBench, TextVQA, ScienceQA, MM-Vet, the GPT language-quality judge and the 13B model. General ability is measured only by VQAv2.
- **Shortcuts left to the grader:** public labels and inference-time machinery (above).
- **Rank calibration:** the coefficients were checked on this iteration's cohort. Ranks under a future cohort depend on its spreads. Under the all-entries cohort reported in `summary.json`, the orderings are the same but the margins shrink: best variant +0.24, seed replicate +0.04.
- **Compute environment:** jobs ran on shared GPUs with one CPU core. A transformers 4.47.1 decoding quirk (padding-blind decoding) invalidated a first full run, which was discarded and redone.
