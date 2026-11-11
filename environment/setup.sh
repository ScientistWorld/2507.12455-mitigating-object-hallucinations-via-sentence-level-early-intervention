#!/bin/bash
# Install Python packages into the container overlay (run by the system with
# internet access).  Install into the SYSTEM Python with uv; no virtualenvs.
#
# Pinned stack (resolved together for Python 3.11 / linux x86_64):
#   torch 2.5.1 (CUDA 12.4 wheels), transformers 4.47.1 (LLaVA-1.5 HF port,
#   GroundingDINO, OWLv2), peft 0.13.2 (LoRA), spaCy 3.8 + en_core_web_sm
#   (object extraction), nltk 3.9.1 (WordNet lexical categories).

set -e

uv pip install --system \
  "torch==2.5.1" "torchvision==0.20.1" \
  "transformers==4.47.1" "tokenizers" "peft==0.13.2" "accelerate==1.2.1" \
  "safetensors" "sentencepiece" "protobuf" \
  "numpy==2.1.3" "scipy==1.14.1" "pillow" \
  "spacy==3.8.2" "nltk==3.9.1" \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
