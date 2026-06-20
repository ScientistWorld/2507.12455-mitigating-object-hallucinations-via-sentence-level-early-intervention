#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: < 1 MB
#
# This script only creates reproducible workspace-local synthetic data.

set -e

cd /home/user

# No external downloads are required for the scaled benchmark.  The synthetic
# image-object scenes are generated from tracked code so fresh clones are fully
# reproducible without network access.
python data/synthetic_scenes.py --out-dir data
