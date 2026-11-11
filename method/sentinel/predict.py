"""Greedy predictions for every evaluation input -> outputs/<name>_predictions.jsonl.

Each output row is {"id": ..., "prediction": "<generated text>"}; evaluation
derives every metric from this raw text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import LLAVA_PATH, ROOT, LlavaRunner, load_image, log, read_jsonl

CAPTION_MAX_NEW_TOKENS = 512
VQA_MAX_NEW_TOKENS = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--inputs", default=str(ROOT / "data/eval_inputs.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N inputs of each task")
    ap.add_argument("--caption-batch", type=int, default=48)
    ap.add_argument("--vqa-batch", type=int, default=96)
    args = ap.parse_args()

    out_path = Path(args.out_dir) / f"{args.name}_predictions.jsonl"
    items = read_jsonl(Path(args.inputs))
    if args.limit:
        items = [r for r in items if r["task"] == "caption"][: args.limit] + [r for r in items if r["task"] == "vqa"][: args.limit]
    done = {r["id"]: r for r in read_jsonl(out_path)}
    todo = [r for r in items if r["id"] not in done]
    if not todo:
        log(f"{out_path} complete; skipping")
        return
    from transformers import LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(LLAVA_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(args.device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        log(f"merged adapter {args.adapter}")
    runner = LlavaRunner(args.device, model=model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for task, bs, max_new in (("caption", args.caption_batch, CAPTION_MAX_NEW_TOKENS), ("vqa", args.vqa_batch, VQA_MAX_NEW_TOKENS)):
        rows = sorted((r for r in todo if r["task"] == task), key=lambda r: (len(r["prompt"]), r["id"]))
        log(f"{args.name}: {len(rows)} {task} inputs")
        for start in range(0, len(rows), bs):
            chunk = rows[start : start + bs]
            prefixes = [runner.prompt_ids(r["prompt"]) for r in chunk]
            gens = runner.generate(prefixes, [load_image(r["image"]) for r in chunk], max_new_tokens=max_new, do_sample=False)
            with out_path.open("a", encoding="utf-8") as f:
                for r, g in zip(chunk, gens):
                    f.write(json.dumps({"id": r["id"], "prediction": runner.decode(g)}) + "\n")
            if (start // bs) % 10 == 0:
                log(f"{args.name}: {task} {start + len(chunk)}/{len(rows)}")
    # Rewrite cleanly (drops any truncated line left by an interrupted run).
    final = {r["id"]: r for r in read_jsonl(out_path)}
    with out_path.open("w", encoding="utf-8") as f:
        for r in items:
            if r["id"] in final:
                f.write(json.dumps({"id": r["id"], "prediction": final[r["id"]]["prediction"]}) + "\n")
    log(f"wrote {out_path} ({len(final)} rows)")


if __name__ == "__main__":
    main()
