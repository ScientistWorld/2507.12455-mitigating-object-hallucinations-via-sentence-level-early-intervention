> **Superseded (iteration 11):** this report verified the retired synthetic 12-object gym. The gym was rebuilt on LLaVA-v1.5-7B and COCO; see `HARDENING.md` and `PROGRESS.md`.

# Verification Report

## Initial Assessment

The workspace started at `core_claim` with a runnable scaled reproduction of the paper's central object-hallucination claim. The method artifacts, evaluation scripts, train/test scoring split, and documentation were already present. The claimed milestone is justified for the compact synthetic benchmark, not for the full paper-scale LLaVA/Object HalBench reproduction.

I checked the workspace structure, method/evaluation separation, download portability, scoring files, train/test evaluator wiring, and documentation. `eval/` reads prediction artifacts and does not import from `method/`. `scripts/download.sh` is self-contained for this workspace because no external downloads are required; it regenerates the compact synthetic data and contains no `shared/` references.

## Issues Found

- Minor: `scoring/DIRECTION.md` described paper-specific implementation levers in a high-level guidance file that should not steer from-scratch agents toward the reproduced method.
- Minor: `scoring/reference.json` included compact synthetic reference values for the runnable experiment without an explicit note distinguishing them from paper-reported full-scale benchmark values.
- Cosmetic: `scoring/TARGETS.md` named specific full-scale resources in the limitation note. This was clearer than necessary for a method-safe guidance file.

## Fixes Applied

- Rewrote `scoring/DIRECTION.md` to describe the allowed research space in method-agnostic terms: improve image-conditioned object grounding while preserving useful coverage and evaluator compatibility.
- Added a `notes` field to the `synthetic_object_halbench` entry in `scoring/reference.json` explaining that these are validated compact synthetic-environment targets, not paper-reported benchmark numbers.
- Softened the limitation wording in `scoring/TARGETS.md` to refer to full paper-scale models, data, external object-checking resources, and public benchmarks without naming implementation-specific components.
- Re-ran `python validate.py --compare` and `python validate.py --post-design`; both checks pass.

## Final Judgment

The workspace is ready to serve as a compact research gym. Future agents can use it in improve mode by modifying the method and writing the same prediction format, and in from-scratch mode by relying on the method-agnostic problem, data, and evaluation files. The scoring captures both benefit metrics (lower hallucination) and constraints (object coverage and QA accuracy), so improvements should require better grounded generation rather than trivial silence.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: compact synthetic benchmark only; the full paper-scale LLaVA/Object HalBench reproduction is documented but not rerun here.
