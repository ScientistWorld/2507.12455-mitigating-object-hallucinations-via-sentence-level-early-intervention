"""One-time builder for the evaluation inputs and the dev/test answer files.

This is the only code that can regenerate the held-out answers, so it lives in
eval/test/.  It was run once; its outputs are committed:

  data/eval_inputs.jsonl   inputs only (image path, prompt / question); no labels
  eval/train/gold.json     answers for the visible development slice
  eval/test/gold.json      answers for the held-out test slice

Sources (public): MSCOCO 2017 val instance + caption annotations
(http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and
VQAv2 val questions/annotations
(https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_{Questions,Annotations}_Val_mscoco.zip).

Usage:
  python -m eval.test.build_splits --coco-ann <dir with instances_val2017.json,
      captions_val2017.json> --vqa-questions <...questions.json>
      --vqa-annotations <...annotations.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .chair import INVERSE_SYNONYMS, caption_objects

SEED = "sentinel-gym-v2-coco-val2017"
N_DEV = 300
N_TEST = 2000
VQA_PER_IMAGE = 3

# Eight diverse detailed-description prompts (Object HalBench style).
CAPTION_PROMPTS = [
    "What is this photo about? Please answer in great detail.",
    "Provide a thorough description of the given picture.",
    "Explain the narrative or story that the image seems to convey, detailing each part that contributes to it.",
    "Please provide a detailed description of the image. Describe the visual elements, colors, shapes, textures, and any objects or people present along with the overall mood or atmosphere portrayed in the image.",
    "Please provide a detailed description of the image, including its visual elements, such as colors, shapes, textures, objects, and people.",
    "Provide an intricate description of the image, capturing its visual elements, including colors, shapes, textures, objects, and any people present.",
    "Compose a detailed account of the image, encapsulating its visual elements, colors, shapes, textures, objects, and any human subjects, by paying careful attention to the specifics.",
    "Provide a thorough description of the given image.",
]
VQA_SUFFIX = "\nAnswer the question using a single word or phrase."


def h(*parts: object) -> str:
    return hashlib.sha256(":".join([SEED, *map(str, parts)]).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-ann", required=True)
    ap.add_argument("--vqa-questions", required=True)
    ap.add_argument("--vqa-annotations", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    ann = Path(args.coco_ann)

    inst = json.loads((ann / "instances_val2017.json").read_text())
    caps = json.loads((ann / "captions_val2017.json").read_text())
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    file_name = {im["id"]: im["file_name"] for im in inst["images"]}

    gold_objects: dict[int, set[str]] = defaultdict(set)
    for a in inst["annotations"]:
        gold_objects[a["image_id"]].add(INVERSE_SYNONYMS[cat_name[a["category_id"]]])
    seg_images = set(gold_objects)
    for a in caps["annotations"]:
        if a["image_id"] in seg_images:
            gold_objects[a["image_id"]] |= caption_objects(a["caption"])

    questions = json.loads(Path(args.vqa_questions).read_text())["questions"]
    answers = {a["question_id"]: a for a in json.loads(Path(args.vqa_annotations).read_text())["annotations"]}
    q_by_image: dict[int, list[dict]] = defaultdict(list)
    for q in questions:
        if q["image_id"] in seg_images:
            q_by_image[q["image_id"]].append(q)

    pool = sorted((i for i in seg_images if len(q_by_image[i]) >= VQA_PER_IMAGE), key=lambda i: h("img", i))
    dev_ids, test_ids = pool[:N_DEV], pool[N_DEV : N_DEV + N_TEST]

    inputs: list[dict] = []
    golds = {"dev": {"captions": {}, "vqa": {}}, "test": {"captions": {}, "vqa": {}}}
    for split, ids in (("dev", dev_ids), ("test", test_ids)):
        for k, image_id in enumerate(ids):
            image = f"val2017/{file_name[image_id]}"
            cid = f"cap_{image_id:012d}"
            inputs.append({"id": cid, "task": "caption", "image": image, "prompt": CAPTION_PROMPTS[k % len(CAPTION_PROMPTS)]})
            golds[split]["captions"][cid] = {"objects": sorted(gold_objects[image_id])}
            chosen = sorted(q_by_image[image_id], key=lambda q: h("q", q["question_id"]))[:VQA_PER_IMAGE]
            for q in chosen:
                qid = f"vqa_{q['question_id']:012d}"
                inputs.append({"id": qid, "task": "vqa", "image": image, "question": q["question"], "prompt": q["question"] + VQA_SUFFIX})
                golds[split]["vqa"][qid] = {"answers": [a["answer"] for a in answers[q["question_id"]]["answers"]]}

    inputs.sort(key=lambda r: r["id"])
    with (root / "data/eval_inputs.jsonl").open("w", encoding="utf-8") as f:
        for row in inputs:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (root / "eval/train/gold.json").write_text(json.dumps(golds["dev"], indent=1, sort_keys=True) + "\n")
    (root / "eval/test/gold.json").write_text(json.dumps(golds["test"], indent=1, sort_keys=True) + "\n")
    print(f"dev images={len(dev_ids)} test images={len(test_ids)} inputs={len(inputs)}")


if __name__ == "__main__":
    main()
