#!/bin/bash
set -euo pipefail
cd /home/user
python -m redteam.case_runner --case bruteforce_object_vector
