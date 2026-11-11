"""In-domain preference data construction (paper Algorithm 1).

For every unlabelled training image and the detailed-description prompt:
  while the model has not ended its description:
    1. sample n in-domain sentence candidates given (image, prompt, context c)
    2. extract the mentioned objects of every candidate
    3. validate each object with two detectors (factual / hallucinated / uncertain)
    4. hallucinated sentence  = contains a hallucinated object;
       non-hallucinated       = only factual objects (uncertain ones ignored)
    5. y_w+ = a context-coherent non-hallucinated sentence (shares an object with c),
       y_l  = a hallucinated sentence  ->  preference sample (v, q, c, y_w+, y_l)
    6. iterative contextual bootstrapping: c <- c + y_w+
All text is kept as the model's own token ids, so pairs are exactly in-domain.

Run one process per GPU with --shard/--num-shards; then --merge.  Each shard
also scores its pairs under the frozen reference model (needed by C-DPO).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from . import config as C
from .common import ROOT, TRAIN_PROMPT, LlavaRunner, append_jsonl, load_image, log, read_jsonl


def stable_int(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class ImageState:
    def __init__(self, name: str, image):
        self.name = name
        self.image = image
        self.context_ids: list[int] = []
        self.context_text = ""
        self.context_lemmas: set[str] = set()
        self.cache: dict[str, dict] = {}
        self.det_inputs = None
        self.steps = 0
        self.pairs: list[dict] = []
        self.done = False
        self.reason = ""
        self.counts = {"factual": 0, "hallucinated": 0, "neutral": 0, "eos": 0, "incomplete": 0}


def run_shard(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / f"pairs_shard{args.shard}.jsonl"
    stats_path = out_dir / f"images_shard{args.shard}.jsonl"
    names = (ROOT / C.TRAIN_IMAGES_LIST).read_text().split()[: args.num_images]
    mine = names[args.shard :: args.num_shards]
    done_stats = read_jsonl(stats_path)
    done = {r["image"] for r in done_stats}
    pairs_before = sum(r["pairs"] for r in done_stats)
    todo = [n for n in mine if n not in done]
    log(f"shard {args.shard}: {len(mine)} images assigned, {len(done)} done, {len(todo)} to do")

    from .detectors import CrossChecker
    from .objects import ObjectExtractor

    runner = LlavaRunner(args.device)
    checker = CrossChecker(args.device)
    extractor = ObjectExtractor()
    prompt_ids = runner.prompt_ids(TRAIN_PROMPT)
    rng = random.Random(stable_int(C.SEED, args.shard))
    t0 = time.time()
    queue = list(todo)
    active: list[ImageState] = []
    call = 0
    n_done = 0
    n_pairs = 0
    batch_images = args.batch_images
    while queue or active:
        out_of_time = args.time_budget_min > 0 and (time.time() - t0) > 60 * args.time_budget_min
        enough = args.max_pairs > 0 and pairs_before + n_pairs >= args.max_pairs
        out_of_time = out_of_time or enough
        while queue and len(active) < batch_images and not out_of_time:
            name = queue.pop(0)
            active.append(ImageState(name, load_image(f"train2017/{name}")))
        if not active:
            break
        prefixes = [prompt_ids + st.context_ids for st in active]
        torch.manual_seed(stable_int(C.SEED, args.shard, call))
        call += 1
        try:
            gens = runner.generate(
                prefixes, [st.image for st in active], max_new_tokens=args.max_sentence_tokens,
                do_sample=True, temperature=args.temperature, num_return_sequences=args.samples,
                stop_at_period=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_images == 1:
                raise
            batch_images = max(1, batch_images // 2)
            queue = [st.name for st in active[batch_images:]] + queue
            active = active[:batch_images]
            log(f"OOM: reducing images per sampling call to {batch_images}")
            continue
        # Decode + extract objects for all complete candidates of this call.
        per_image = []
        texts = []
        for j, st in enumerate(active):
            cands = []
            for ids in gens[j * args.samples : (j + 1) * args.samples]:
                body = [t for t in ids if t != runner.eos_id]
                if not body:
                    st.counts["eos"] += 1
                    cands.append({"kind": "eos"})
                    continue
                if ids[-1] not in (runner.period_id, runner.eos_id):
                    st.counts["incomplete"] += 1
                    cands.append({"kind": "incomplete"})
                    continue
                cands.append({"kind": "sent", "ids": ids, "text": runner.decode(ids), "ends": ids[-1] == runner.eos_id})
                texts.append(cands[-1]["text"])
            per_image.append(cands)
        objs_all = extractor.extract(texts)
        k = 0
        for st, cands in zip(active, per_image):
            for c in cands:
                if c["kind"] == "sent":
                    c["objects"] = objs_all[k]
                    k += 1
        # Validate, classify, build pairs, extend contexts.
        for st, cands in zip(active, per_image):
            n_eos = sum(c["kind"] == "eos" for c in cands)
            if n_eos * 2 > len(cands):
                st.done, st.reason = True, "eos"
                continue
            sents = [c for c in cands if c["kind"] == "sent"]
            new = sorted({o["phrase"] for c in sents for o in c["objects"]} - set(st.cache))
            if new:
                if st.det_inputs is None:
                    st.det_inputs = checker.prepare(st.image)
                st.cache.update(checker.validate(st.det_inputs, new))
            for c in sents:
                labels = [st.cache[o["phrase"]]["label"] for o in c["objects"]]
                c["lemmas"] = {o["lemma"] for o in c["objects"]}
                if "hallucinated" in labels:
                    c["label"] = "hallucinated"
                elif "factual" in labels:
                    c["label"] = "factual"
                else:
                    c["label"] = "neutral"
                st.counts[c["label"]] += 1
            positives = [c for c in sents if c["label"] == "factual"]
            negatives = [c for c in sents if c["label"] == "hallucinated"]
            if st.context_lemmas:
                coherent = [c for c in positives if c["lemmas"] & st.context_lemmas]
            else:  # first sentence: no context to be incoherent with
                coherent = positives
            chosen = rng.choice(coherent) if coherent else None
            if chosen is not None and negatives:
                rejected = rng.choice(negatives)
                st.pairs.append({
                    "pair_id": f"{st.name}:{st.steps}",
                    "image": f"train2017/{st.name}",
                    "prompt": TRAIN_PROMPT,
                    "step": st.steps,
                    "context_ids": list(st.context_ids),
                    "context": st.context_text,
                    "chosen_ids": chosen["ids"],
                    "chosen": chosen["text"],
                    "rejected_ids": rejected["ids"],
                    "rejected": rejected["text"],
                    "chosen_objects": {o["phrase"]: st.cache[o["phrase"]] for o in chosen["objects"]},
                    "rejected_objects": {o["phrase"]: st.cache[o["phrase"]] for o in rejected["objects"]},
                })
            # Context extension (ICB): the context-coherent positive; otherwise any
            # verified non-hallucinated sentence; otherwise a sentence without
            # any object mention (nothing to hallucinate).
            object_free = [c for c in sents if not c["objects"]]
            nxt = chosen
            if nxt is None and positives:
                nxt = rng.choice(positives)
            if nxt is None and object_free:
                nxt = rng.choice(object_free)
            if nxt is None:
                st.done, st.reason = True, "no_hallucination_free_continuation"
                continue
            body = [t for t in nxt["ids"] if t != runner.eos_id]
            st.context_ids += body
            st.context_text = (st.context_text + " " + nxt["text"]).strip()
            st.context_lemmas |= nxt["lemmas"]
            st.steps += 1
            if nxt["ends"]:
                st.done, st.reason = True, "eos"
            elif st.steps >= args.max_steps:
                st.done, st.reason = True, "max_steps"
        finished = [st for st in active if st.done]
        for st in finished:
            st.det_inputs = None
        for st in finished:
            append_jsonl(pairs_path, st.pairs)
            append_jsonl(stats_path, [{"image": st.name, "steps": st.steps, "pairs": len(st.pairs), "reason": st.reason, **st.counts}])
            n_done += 1
            n_pairs += len(st.pairs)
        active = [st for st in active if not st.done]
        if finished and (n_done % 25 < len(finished)):
            el = time.time() - t0
            log(f"shard {args.shard}: {n_done}/{len(todo)} images, {n_pairs} pairs, {el / 60:.1f} min, {el / max(1, n_done):.2f} s/image")
    log(f"shard {args.shard}: finished {n_done} images this run, {n_pairs} pairs")
    del checker
    torch.cuda.empty_cache()
    score_reference(runner, out_dir, args.shard)


@torch.no_grad()
def score_reference(runner: LlavaRunner, out_dir: Path, shard: int) -> None:
    """log pi_ref(y | v, q, c) for chosen and rejected of every pair of this shard."""
    from .cdpo import response_logps

    pairs = read_jsonl(out_dir / f"pairs_shard{shard}.jsonl")
    ref_path = out_dir / f"ref_logps_shard{shard}.jsonl"
    have = {r["pair_id"] for r in read_jsonl(ref_path)}
    todo = [p for p in pairs if p["pair_id"] not in have]
    log(f"shard {shard}: scoring {len(todo)} pairs under the reference model")
    prompt_cache: dict[str, list[int]] = {}
    for start in range(0, len(todo), 4):
        chunk = todo[start : start + 4]
        prefixes, responses, images = [], [], []
        for p in chunk:
            if p["prompt"] not in prompt_cache:
                prompt_cache[p["prompt"]] = runner.prompt_ids(p["prompt"])
            pre = prompt_cache[p["prompt"]] + p["context_ids"]
            img = load_image(p["image"])
            prefixes += [pre, pre]
            responses += [p["chosen_ids"], p["rejected_ids"]]
            images += [img, img]
        lp = response_logps(runner.model, runner.pad_id, runner.pixel_values(images), prefixes, responses).float().cpu()
        append_jsonl(ref_path, [{"pair_id": p["pair_id"], "ref_chosen": float(lp[2 * i]), "ref_rejected": float(lp[2 * i + 1])} for i, p in enumerate(chunk)])
    log(f"shard {shard}: reference scoring done")


def merge(args) -> None:
    out_dir = Path(args.out_dir)
    pairs, refs, stats = [], {}, []
    for path in sorted(out_dir.glob("pairs_shard*.jsonl")):
        pairs += read_jsonl(path)
    for path in sorted(out_dir.glob("ref_logps_shard*.jsonl")):
        refs.update({r["pair_id"]: r for r in read_jsonl(path)})
    for path in sorted(out_dir.glob("images_shard*.jsonl")):
        stats += read_jsonl(path)
    merged = []
    for p in pairs:
        r = refs.get(p["pair_id"])
        if r is None:
            raise RuntimeError(f"missing reference log-probs for {p['pair_id']}")
        merged.append({**p, "ref_chosen": r["ref_chosen"], "ref_rejected": r["ref_rejected"]})
    merged.sort(key=lambda p: p["pair_id"])
    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in merged:
            f.write(json.dumps(p) + "\n")
    summary = {
        "images": len(stats),
        "pairs": len(merged),
        "pairs_per_image": len(merged) / max(1, len(stats)),
        "mean_steps": sum(s["steps"] for s in stats) / max(1, len(stats)),
        "finish_reasons": {k: sum(s["reason"] == k for s in stats) for k in sorted({s["reason"] for s in stats})},
        "candidate_labels": {k: sum(s.get(k, 0) for s in stats) for k in ("factual", "hallucinated", "neutral", "eos", "incomplete")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"merged: {json.dumps(summary)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="checkpoints/preferences")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-images", type=int, default=C.NUM_IMAGES)
    ap.add_argument("--samples", type=int, default=C.SAMPLES_PER_STEP)
    ap.add_argument("--temperature", type=float, default=C.SAMPLE_TEMPERATURE)
    ap.add_argument("--max-sentence-tokens", type=int, default=C.MAX_SENTENCE_TOKENS)
    ap.add_argument("--max-steps", type=int, default=C.MAX_ICB_STEPS)
    ap.add_argument("--batch-images", type=int, default=C.BATCH_IMAGES)
    ap.add_argument("--time-budget-min", type=float, default=0.0, help="stop taking new images after this many minutes (0 = no limit)")
    ap.add_argument("--max-pairs", type=int, default=0, help="stop taking new images once this shard holds this many pairs (0 = MAX_PAIRS / num_shards)")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.max_pairs == 0:
        args.max_pairs = -(-C.MAX_PAIRS // args.num_shards)
    if args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
