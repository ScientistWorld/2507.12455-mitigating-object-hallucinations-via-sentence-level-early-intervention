# Audit plan before observing results

Reference configuration: hidden size 192, 30 SFT epochs, seed 7, 700 preference
images, 2 context steps, 16 candidates per context at temperature 1.4, 1 C-DPO
epoch at learning rate 5e-5 and beta 0.1. This is the configuration that produced
the frozen outputs, recovered from the reproduction's recorded configuration.
Seeds 7, 17, and 27 independently retrain this same procedure.

Stronger interventions: increase SFT to 90 epochs; alternatively remove the
artificial 50% corruption of supervised captions and train for 60 epochs. Both
retain the compact model and the actual SENTINEL preference construction and
C-DPO. Weaker interventions: remove alignment using the frozen baseline output;
train the same model for one SFT epoch with alignment removed.

Checks use the true no-argument hidden script, with original eval files checked
against the starting git commit. Runtime input, score and output mutations are
restored after every case. No method-side test-slice knowledge is used. The data
leakage checks cover all 240 publicly shipped evaluation examples, rather than
choosing examples using the hidden seed.

Main shortcut classes: answer-file copying; public-generator reconstruction;
replacing the fixed gold artifact; manifest omission; ignoring negation or
unrecognized hallucinated objects; empty and all-object captions; method-supplied
metrics and prewritten score files; null/missing/empty submissions; comparison
baseline replacement. Exhaustive search uses the same explicit object vector as
the reference model and all 781 allowed scenes, without learning.

The eventual attempt distribution is not available. Raw metric vectors are the
primary evidence. report.py additionally computes the stated normalized rank
using the complete valid audit cohort and discloses that cohort and its standard
deviations. Reference-versus-candidate and honest-plus-silence cohorts test
whether additive soft constraints can reward a method that fails the task.
