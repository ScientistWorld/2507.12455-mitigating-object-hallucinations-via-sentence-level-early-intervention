#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE submitting compute jobs.
# Idempotent: every fetch is skipped when its target file already exists.
#
# Estimated total download size: ~16 GB
#   - llava-hf/llava-1.5-7b-hf (pinned revision)        ~14.1 GB
#   - IDEA-Research/grounding-dino-base (pinned)          ~0.9 GB
#   - google/owlv2-base-patch16-ensemble (pinned)         ~0.6 GB
#   - COCO 2017 images listed in data/ (~4,800 jpgs)      ~0.8 GB
#   - NLTK WordNet corpus                                 ~11 MB
# Estimated disk usage after extraction: ~16 GB
#
# Everything lands in data/downloads/ (gitignored).  Python packages
# (including the spaCy English model) are installed by environment/setup.sh.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DL="$ROOT/data/downloads"
mkdir -p "$DL/models" "$DL/coco" "$DL/nltk_data/corpora"

fetch() {  # fetch <url> <dest>
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then return 0; fi
  mkdir -p "$(dirname "$dest")"
  curl -fsSL --retry 5 --retry-delay 5 -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

hf_model() {  # hf_model <repo> <revision> <local-name> <files...>
  local repo="$1" rev="$2" name="$3"; shift 3
  local dir="$DL/models/$name"
  mkdir -p "$dir"
  for f in "$@"; do
    fetch "https://huggingface.co/$repo/resolve/$rev/$f" "$dir/$f"
  done
}

echo "[download] LLaVA-v1.5-7B (HF format)"
hf_model llava-hf/llava-1.5-7b-hf b234b804b114d9e37bb655e11cbbb5f5e971b7a9 llava-1.5-7b-hf \
  added_tokens.json chat_template.json config.json generation_config.json \
  model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors \
  model.safetensors.index.json preprocessor_config.json processor_config.json \
  special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json

echo "[download] GroundingDINO-base (open-vocabulary detector #1)"
hf_model IDEA-Research/grounding-dino-base 12bdfa3120f3e7ec7b434d90674b3396eccf88eb grounding-dino-base \
  config.json model.safetensors preprocessor_config.json special_tokens_map.json \
  tokenizer.json tokenizer_config.json vocab.txt

echo "[download] OWLv2-base (open-vocabulary detector #2)"
hf_model google/owlv2-base-patch16-ensemble cfd3195ba4ea9592eec887ded089f4c08eff231d owlv2-base-patch16-ensemble \
  added_tokens.json config.json merges.txt model.safetensors preprocessor_config.json \
  special_tokens_map.json tokenizer_config.json vocab.json

echo "[download] NLTK WordNet"
if [ ! -d "$DL/nltk_data/corpora/wordnet" ]; then
  fetch "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip" \
        "$DL/nltk_data/corpora/wordnet.zip"
  python3 -m zipfile -e "$DL/nltk_data/corpora/wordnet.zip" "$DL/nltk_data/corpora/"
fi

echo "[download] COCO 2017 images (unlabelled preference images + evaluation inputs)"
# Only image files are fetched; no COCO annotation file is needed by the
# method or shipped to it.  Lists are tracked in data/.
python3 - "$ROOT" > "$DL/coco/_needed.txt" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
seen = set()
for line in (root / "data/sentinel_train_images.txt").read_text().split():
    seen.add(f"train2017/{line}")
for line in (root / "data/eval_inputs.jsonl").read_text().splitlines():
    if line.strip():
        seen.add(json.loads(line)["image"])
for rel in sorted(seen):
    print(rel)
PY
missing="$DL/coco/_missing.txt"
: > "$missing"
while read -r rel; do
  [ -s "$DL/coco/$rel" ] || echo "$rel" >> "$missing"
done < "$DL/coco/_needed.txt"
echo "[download] $(wc -l < "$missing") COCO images missing"
if [ -s "$missing" ]; then
  xargs -P 16 -I{} sh -c 'mkdir -p "$(dirname "'"$DL"'/coco/{}")" && curl -fsSL --retry 5 -o "'"$DL"'/coco/{}.part" "http://images.cocodataset.org/{}" && mv "'"$DL"'/coco/{}.part" "'"$DL"'/coco/{}"' < "$missing"
fi
rm -f "$DL/coco/_needed.txt" "$missing"

echo "[download] done"
