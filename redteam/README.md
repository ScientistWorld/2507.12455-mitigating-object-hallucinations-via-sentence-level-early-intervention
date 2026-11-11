# Audit runs

Each named directory contains a `run.sh`. Run it from `/home/user` in the gym's
container, as a scheduled job. It installs the candidate into the default
`outputs/`, runs the original `bash scripts/evaluate_train.sh` and
`bash scripts/evaluate_test.sh` with **no arguments**, saves both score files,
logs and return codes, then restores the original runtime inputs.

`run_suite.sh` bundles the runs into one CPU job. Quality interventions use the
actual method code. Shortcut methods use only files provided to future agents;
their source does not import the hidden evaluator, paper, or reference scores.
Checks that alter only a submitted caption reuse the trained reference outputs,
as explicitly permitted by the audit. These output files and checkpoints are
generated, ignored artifacts. `reference_seed*/run.sh` regenerate the reference
method without relying on checkpoints. `shortcut_rebuild_seed` uses only the
public data generator and its public defaults, not the test slice seed.

`report.py` is audit analysis, separated from candidate preparation. It applies
the stated rank rule using absolute coefficient magnitudes, metric directions,
experiment weights, and per-metric standard deviations across a disclosed
cohort. Future attempt variances are unknowable; every numerical rank is labeled
with its cohort. All raw hidden-test metrics remain available in each folder.
Failed test calls are recorded with `score_is_fresh: false`; any surviving score
file is explicitly stale and is never treated as an earned score.
