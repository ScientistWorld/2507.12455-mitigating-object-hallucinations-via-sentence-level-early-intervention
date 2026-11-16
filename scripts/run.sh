#!/bin/bash
# Job script: reproduce the reference end to end (baseline predictions,
# SENTINEL preference construction, C-DPO, predictions, development and
# held-out scores).  Needs 1-2 GPUs; every stage resumes from its artifacts.
# The hardening experiments that produced HARDENING.md are in
# redteam/hardening_v2/job_*.sh.

set -euo pipefail
cd /home/user
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID

bash scripts/reproduce.sh
