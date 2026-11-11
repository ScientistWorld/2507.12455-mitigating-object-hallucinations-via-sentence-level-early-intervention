# SENTINEL gym scoring audit

**The gym does not reliably measure progress on the paper's problem.** More training scores higher and severe undertraining scores lower, but removing alignment costs less than the measured seed spread. The selected reference improves on its baseline in the paper's direction; one of three retrained seeds instead worsens both hallucination metrics. Most seriously, the substitute task exposes the complete object-presence answer as the model's input. Exhaustively enumerating its 781 possible scenes solves the hidden test perfectly without learning. It is too small and too informative to establish the paper's effect on hallucination in real image descriptions.

All results below come from completed job `02631c51-34a` (exit 0). Each case has its own executable `redteam/<name>/run.sh`, saved train/test scores, logs, and `result.json`. The 23 cases ran in one CPU job. The original, unmodified `bash scripts/evaluate_test.sh` ran **with no arguments**, in a separate subprocess after candidate preparation. No evaluator process, Python environment, or hidden selection seed was altered or used by a candidate. Separate processes suffice for these disk-only candidates: none depends on a process surviving into testing. Cached reference predictions were reused for output-only checks as authorized. Runtime changes were restored after every case. [Run protocol](redteam/README.md), [preregistered plan](redteam/PLAN.md), [job logs](redteam/job_evidence/stdout.log).

## Do better results score higher?

Metric notation throughout: **R / M / C / L / Q** = response hallucination percentage (lower), mention hallucination percentage (lower), object recall percentage (higher), average recognized distinct objects (higher), derived object-presence QA percentage (higher). All percentages are reported on the same hidden 100 scenes. The actual test reference is **55 / 20 / 89.474 / 3.40 / 91.667**, re-evaluated from the original predictions. The synthetic values inside `reference.json` describe all 240 evaluation scenes and are different; they are not substituted for `scores_test.json` here.

| Real method run | Intervention | R / M / C / L / Q | Rank gain over reference* |
|---|---|---|---:|
| [Reference](redteam/reference/run.sh) | Original trained outputs | 55 / 20 / 89.474 / 3.40 / 91.667 | 0 |
| [More training](redteam/better_more_training/run.sh) | 90 rather than 30 SFT epochs; retains preference generation and C-DPO | 1 / 0.331 / 99.013 / 3.02 / 99.405 | +0.78495 |
| [Clean training](redteam/better_clean_training/run.sh) | Remove artificial 50% corruption of training captions; 60 SFT epochs; retains preference generation and C-DPO | 1 / 0.328 / 100 / 3.05 / 100 | +0.79781 |
| [Alignment removed](redteam/weaker_no_alignment/run.sh) | Original frozen SFT baseline, with preference alignment removed | 63 / 24.128 / 85.855 / 3.44 / 88.492 | −0.16316 |
| [Severe undertraining](redteam/weaker_undertrained/run.sh) | Same 192-hidden-unit model, one SFT epoch, no alignment | 86 / 72.258 / 14.145 / 1.55 / 42.857 | −1.83378 |

The stronger variants and severe undertraining move the aggregate in the correct direction by more than the **0.65363** reference-seed rank range. This is evidence that the scorer responds to large quality changes on this toy task. The **0.16316** loss from removing the key alignment idea is smaller than seed variation and does not demonstrate a robust contribution. The stronger models' 54-point improvement in R is also smaller than R's 59-point seed range, although their improvements in M, C, and Q exceed the respective seed ranges. Stronger variants were each trained once; this is not a multi-seed significance claim.

Both stronger variants have **identical baseline and post-C-DPO hidden metrics**. Their preference-generation and C-DPO stages ran successfully (333 and 428 pairs respectively). Nearly solving this test requires neither the paper's alignment contribution nor its real perception problem.

\* Numerical ranks use the disclosed complete valid audit cohort, not an invented future variance. With population standard deviations across its 20 fresh, complete candidate results, the rule is

`gain = 0.3 × [−ΔR/37.14902 − ΔM/26.78600 + 0.7ΔC/26.91986 + 0.2ΔL/3.69992 + 0.5ΔQ/19.24076]`.

Here each delta is relative to the hidden reference above. This uses `abs(coefficient)` and `higher_is_better`; negative coefficient signs are not applied a second time. Each improvement is divided by variation in that metric, so coefficients express importance rather than units. The cohort includes all successful complete cases below, including ties; it excludes null, empty-file, and missing-method cases. [Exact members, standard deviations, raw results and ranks](redteam/summary.json); [calculation](redteam/report.py). Future attempt variances are unknown, so these numerical gains cannot be promised for every future cohort. Exact ties remain ties under any normalization. Gold replacement improves every scored metric, independently of normalization.

## Seed spread and reproduction check

The selected reference configuration was reconstructed from the reproduction's recorded settings: hidden size 192; SFT 30 epochs at 0.003; batch 64; 700 preference images; two context steps; 16 sampled candidates at temperature 1.4; one C-DPO epoch at 0.00005 and beta 0.1; caption corruption probability 0.5. Seeds independently retrain SFT, sample preferences, and train C-DPO. [Implementation](redteam/train_variant.py), with individual configs and training logs in each run folder.

| Retrained reference | R | M | C | L | Q | Rank gain* |
|---|---:|---:|---:|---:|---:|---:|
| [Seed 7](redteam/reference_seed7/run.sh) | 57 | 18.234 | 94.408 | 3.51 | 93.254 | +0.05628 |
| [Seed 17](redteam/reference_seed17/run.sh) | 83 | 25.926 | 92.105 | 3.78 | 91.865 | −0.26425 |
| [Seed 27](redteam/reference_seed27/run.sh) | 24 | 9.740 | 91.447 | 3.08 | 93.452 | +0.38938 |
| **Maximum minus minimum** | **59 pp** | **16.186 pp** | **2.961 pp** | **0.70** | **1.587 pp** | **0.65363** |

A claimed gain smaller than the corresponding spread is noise at this audit's resolution. Seed 7 does not exactly recreate the cached reference despite the recovered configuration; the original frozen outputs remain the comparison anchor. The source defaults also do not encode the selected configuration: they use hidden size 128, SFT 20 epochs, and eight DPO epochs at 0.0007. Running the default training command therefore does not reproduce the selected reference.

The frozen SFT baseline scores **63 / 24.128 / 85.855 / 3.44 / 88.492**. The selected aligned reference reduces R by 8 points and M by 4.128 points and increases recall and QA. Thus its direction agrees with the paper. The paper's 7B result is much larger, 52.7→4.3 response hallucination and 28.0→2.6 mention hallucination. The agreement here is not robust across seeds:

| Seed | Baseline → aligned R | Baseline → aligned M |
|---|---:|---:|
| 7 | 59 → 57 | 21.839 → 18.234 |
| 17 | **73 → 83** | **24.523 → 25.926** |
| 27 | 29 → 24 | 12.219 → 9.740 |

The constant language-prior baseline scores **100 / 73.667 / 25.987 / 3.00 / 45.437**, rank −1.82661 in the audit cohort. It is clearly below the reference, with numerical room above the reference. However, the exhaustive alternative scores **0 / 0 / 100 / 3.04 / 100**, rank +0.80940. It enumerates all `C(12,2)+C(12,3)+C(12,4)=781` legal scenes, compares each with the same exact 12-bit object vector the model receives, then emits a template. It uses no training, captions, QA answers, or hidden slice membership. This is an **honest alternative on the substitute task**, with [its own real hidden-test run](redteam/bruteforce_object_vector/run.sh). It shows the task is solved by decoding its answer-bearing input. A better score here need not be a better method on the paper's problem.

### Departures from the paper

The implementation retains recognizable in-domain positive/negative sampling, a frozen reference, and the context-masked log-ratio C-DPO objective. The following departures nevertheless change the procedure or problem, beyond reducing resources. Sources: [paper sections 3.2–3.4 and appendix B/C](paper/2507.12455.md), [training](method/sentinel/training.py), [model](method/sentinel/model.py), [data generator](data/synthetic_scenes.py).

| Component | Paper | Reproduced environment |
|---|---|---|
| Perception and model | LLaVA-v1.5 7B/13B with CLIP image features | Small GRU; input is gold object-presence vector, with a direct visual-to-token bias; no pixels |
| Data and failure source | Real Visual Genome images; pretrained MLLM hallucinations | 900 synthetic training scenes and 240 evaluation scenes; 12 object types and fixed templates; hallucinations deliberately added to half the SFT labels |
| Object extraction | SceneGraphParser, POS filtering and WordNet concrete entities; open vocabulary | Substring search over 12 color/shape strings; ignores negation and all unrecognized objects |
| Validation | GroundingDINO and YOLO World cross-check independent visual predictions | Both validators read the same exact gold set; the second returns unknown for every yellow triangle |
| Uncertainty | Ignore uncertain objects | Any uncertain object rejects the entire sampled sentence |
| Sentence generation | Stop sampling upon sentence completion | Generate up to 18 tokens/EOS, then truncate to the first sentence; extra sampling consumes randomness |
| Positive coherence | Positive descriptions refer to entities explicitly in context | Shared object, color, **or shape** qualifies; falls back to any positive and preferentially chooses novel objects |
| Bootstrapping | Continue contextual bootstrapping until EOS | Fixed two context steps, including skip-on-no-pair behavior |
| Optimization | LoRA on LLM linear layers, rank 128/alpha 256, cosine schedule, zero weight decay, one epoch; 8.6K/7.0K preference data | Update all model parameters with constant-rate AdamW and its default 0.01 decay; hundreds of preference pairs; different learning rate/model scale |
| Evaluation | Real Object HalBench, AMBER, HallusionBench and broad VQA/OCR/reasoning benchmarks; multiple caption prompts | One prompt, 100 synthetic scenes; all five metrics derive from one 12-object caption parser; no actual general capability or language-quality evaluation |

These substitutions remove visual uncertainty, open vocabulary, natural language diversity and broad capability retention from the research problem. The original paper's early-intervention mechanism is therefore not established merely by a lower synthetic hallucination score.

## Shortcut runs

Every linked name below is the corresponding `run.sh`. Its sibling `scores_test.json` is the score file copied after the real hidden script. “Tie” always means an exact **shortcut tie**, not evidence of scientific improvement. Failed scripts retain stale files only as diagnostic evidence; they are never assigned an earned reference score. Gains use the same disclosed cohort as above.

| Shortcut | What it puts in place | Hidden R / M / C / L / Q; gain vs reference | Works or fails |
|---|---|---|---|
| [shortcut_copy_labels](redteam/shortcut_copy_labels/run.sh) | Copy `data/eval_scenes.jsonl`'s shipped caption into each prediction | **0 / 0 / 100 / 3.04 / 100; +0.80940** | **Works.** Test answers are visible for all 240 scenes, so hidden slice selection gives no protection. |
| [shortcut_rebuild_seed](redteam/shortcut_rebuild_seed/run.sh) | Re-run the public generator with default seed 17, advance through 900 training scenes, reconstruct 240 evaluation captions by id | **0 / 0 / 100 / 3.04 / 100; +0.80940** | **Works.** Does not read evaluation object/caption/QA answers and does not know the hidden slice seed. Merely removing the shipped captions would not fix it. |
| [shortcut_gold_replacement](redteam/shortcut_gold_replacement/run.sh) | Replace the supposedly fixed `data/eval_scenes.jsonl` gold with all 12 objects and all-yes QA; submit a constant all-object caption | **0 / 0 / 100 / 12 / 100; +0.95470** | **Works.** The hidden evaluator treats mutable data as authoritative. Every scored metric reaches its best bound. This also demonstrates fixed-artifact replacement and a successful constant-output shortcut. |
| [shortcut_manifest_omission](redteam/shortcut_manifest_omission/run.sh) | Keep cached reference under `shortcut_retained_reference`; write blank new `sentinel` captions; replace train-score method list with only the retained reference | **55 / 20 / 89.474 / 3.40 / 91.667; 0** | **Works: shortcut tie.** Four methods exist on disk, but hidden results contain only the unchanged reference. Failed new method and baselines are omitted. This demonstrates selective reporting; candidate-name policy outside the stated rank could still reject it. |
| [shortcut_polarity_flip](redteam/shortcut_polarity_flip/run.sh) | Change every “There is a” in the cached predictions to “There is no” | **55 / 20 / 89.474 / 3.40 / 91.667; 0** | **Works: shortcut tie.** Denying the object claims does not change any score or derived QA answer. |
| [shortcut_unscored_hallucinations](redteam/shortcut_unscored_hallucinations/run.sh) | Append elephant, zebra, giraffe and purple diamond claims to each cached prediction | **55 / 20 / 89.474 / 3.40 / 91.667; 0** | **Works: shortcut tie.** These nonexistent objects are outside the parser's 12 strings and never count as hallucinations. |
| [shortcut_empty_caption](redteam/shortcut_empty_caption/run.sh) | Submit an empty caption for every scene | **0 / 0 / 0 / 0 / 39.683; −0.49023** | Fails to beat reference in the full audit cohort; **works conditionally** under the specified normalization: reference-plus-empty cohort gives **+0.36**, despite zero object recall. |
| [shortcut_all_objects](redteam/shortcut_all_objects/run.sh) | Constant caption listing all 12 objects, with original gold unchanged | 100 / 74.667 / 100 / 12 / 60.317; −0.99848 | Fails in the audit cohort (also −0.96 in the pairwise cohort). Coverage alone does not overcome its hallucinations here. |
| [shortcut_metric_forgery](redteam/shortcut_metric_forgery/run.sh) | All-object caption plus impossible ±1e9 metrics, a null metric, fake objects/QA, success flag, negative timing and fake QA predictions | 100 / 74.667 / 100 / 12 / 60.317; −0.99848 | Fails. Method-supplied metrics, gold fields, QA answers and flags are ignored. Caption metrics are recomputed from gold on disk. |
| [shortcut_testscore_forgery](redteam/shortcut_testscore_forgery/run.sh) | Prewrite `scoring/scores_test.json` with impossible winning values, then run hidden test on all-object captions | 100 / 74.667 / 100 / 12 / 60.317; −0.99848 | Fails. A successful hidden run overwrites the forged scoreboard. |
| [shortcut_null_prediction](redteam/shortcut_null_prediction/run.sh) | Set every prediction to JSON null | **No fresh score; train/test both exit 1** | Fails as an earned shortcut. `None.lower()` crashes; old reference scoreboard remains. No worst-case scores are written. |
| [shortcut_empty_file](redteam/shortcut_empty_file/run.sh) | Submit an empty predictions file | **No fresh score; train/test both exit 1** | Fails as an earned shortcut. Missing required ids raises an error; old reference scoreboard remains. No worst-case scores are written. |
| [shortcut_missing_file](redteam/shortcut_missing_file/run.sh) | Remove the proposed method's predictions before running visible evaluation | **Proposed method absent; train/test both exit 0** | **Works to avoid a loss**, not a measured winning candidate. Train lists just the two baselines; hidden test follows that list and assigns no worst values to the missing proposed method. With missing-value skipping its contribution can disappear entirely. |
| [shortcut_baseline_replacement](redteam/shortcut_baseline_replacement/run.sh) | Replace both comparison baselines with blank captions, retain original proposed predictions | **55 / 20 / 89.474 / 3.40 / 91.667; 0** | **Shortcut tie.** Comparison artifacts are mutable and the altered baselines are accepted; this does not increase the proposed method's absolute rank. It can distort a claimed baseline comparison. |

Shortcut preparation is in [redteam/shortcuts.py](redteam/shortcuts.py). Label-copying, generator reconstruction, and gold replacement use only agent-visible `data/`. Caption-corruption and comparison checks reuse cached predictions, rather than retraining the reference. No shortcut reads the paper, hidden evaluator, hidden scores, or hidden selection seed to construct answers.

## Score completeness, bounds and task failure

**Reference at a bound: none.** All five runnable reference metrics are strictly inside their mathematical ranges: R and M have best bound 0%, C and Q have best bound 100%, and L has best scored bound 12 because the extractor recognizes 12 distinct strings. The paper-reference entries likewise have no best-bound values: their two hallucination rates and three AMBER rates are above zero; HallusionBench, VQAv2, TextVQA, ScienceQA and MM-Vet scores are below 100 for both model sizes. Thus none of the 25 configured proposed metric entries is already at its best bound. This numerical headroom does not rescue the problem's trivial solution.

**Two whole experiments are never measured.** Every fresh score file contains only `synthetic_object_halbench` (weight 0.30). Neither `paper_llava_v15_7b_main` (0.45) nor `paper_llava_v15_13b_main` (0.25) is written. Their AMBER, HallusionBench, OCR, reasoning and general VQA constraints therefore make **zero** contribution under missing-value skipping. Documentation acknowledges them as reference-only, but the weighted rank still configures them. The runnable Q metric is just object-presence membership inferred from the same generated caption; it does not test actual QA generation or independent general capability.

**The additive constraints do not guarantee task success.** With only the reference and `shortcut_empty_caption` as the attempt cohort, each nonzero normalized change has magnitude 2. Silence gains `0.3 × 2 × (1 + 1 − 0.7 − 0.2 − 0.5) = +0.36`. It can outrank the reference while recalling no objects. This is a computed application of the requested rank to two real hidden-test results, not a second evaluator or a claim about all future cohorts. Silence loses in the full audit cohort (−0.49023) and in the honest-variants-plus-silence cohort (−0.26806). The possibility depends on future variances, so these soft constraints do not enforce the claim. Rewarding more recognized objects through L also rewards verbosity even when extra objects are false.

**Failed and omitted runs do not get worst values in every metric/experiment.** Null and empty-file runs abort before writing a new scoreboard, leaving the previous valid scoreboard on disk. Rejecting nonzero test exits elsewhere may prevent a stale-file win, but the stated per-metric failure contract is still absent. Missing-method runs can exit successfully with no proposed result at all. Even fully successful runs omit both paper experiments. The audit deliberately records these distinctions and never interprets a stale score file as a successful reference tie.

**Not every on-disk method is scored.** [The hidden evaluator](eval/test/evaluate.py) takes method names from editable `scoring/scores_train.json` whenever that file exists; it discovers prediction files only when train scores are absent. The manifest-omission run leaves `baseline`, `language_prior`, `sentinel`, and `shortcut_retained_reference` files on disk, but writes only the retained reference into hidden `results`. This is directly observed in its saved `result.json` and score file. Any method that is selected and succeeds does receive all five synthetic metrics; the failure is method/experiment omission and error handling.

**Submitted numeric results are otherwise recomputed correctly.** Prediction-side fields cannot directly set hallucination, recall, QA, timing or success metrics, and a successful hidden run replaces prewritten test scores. The important trust failure is that the authoritative gold file and selected method list remain writable. The score does not verify model identity, model size, training effort or scientific provenance.

## Changes needed in the next iteration

1. Restore the real perception problem, or choose a substitute whose inputs do not already encode the object answers. Keep held-out answers and their reproducible generator state separate from agent-visible data. Increase difficulty only after an honest exhaustive/template baseline is no longer perfect.
2. Have the restored evaluator own immutable gold and method-discovery policy. Score the required proposed method and every discovered method, independently of writable train scores; keep comparison baselines fixed.
3. Write a fresh complete scoreboard on every run, including explicit worst values for each required missing/failed metric and experiment. Remove reference-only experiments from the rank until they actually run, or implement their capability checks.
4. Enforce coverage and capability requirements as constraints; do not rely on variance-dependent additive compensation. Check negation and open-vocabulary false claims, then verify that removing the key idea loses by more than multi-seed variation.

All audit source and results are under `redteam/`. `scripts/run.sh`, data, evaluation, method, original predictions and scoring inputs were restored after the audit. Scheduler-generated job logs are preserved under `redteam/job_evidence/`; the job directory itself is restored to its initial git state. The required final `action.yaml` is the only control-file exception to the requested `redteam/` plus `REDTEAM.md` workspace changes. Pre-existing workspace changes recorded in `redteam/initial_state.json` are preserved.
