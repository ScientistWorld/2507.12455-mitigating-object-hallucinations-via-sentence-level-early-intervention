"""Context-aware DPO (C-DPO, paper Eq. 2 and Algorithm 2) with LoRA.

L = -log sigma( beta * [ (log pi(y_w+|v,q,c) - log pi(y_l|v,q,c))
                         - (log ref(y_w+|v,q,c) - log ref(y_l|v,q,c)) ] )
The context c is part of the input but masked out of the log-probabilities;
only the tokens of y_w+ / y_l contribute.  LoRA adapts only the language
model's linear layers; vision tower and projector are frozen (Table 10).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import config as C
from .common import LLAVA_PATH, LlavaRunner, load_image, log, read_jsonl, seed_all

LORA_TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def response_logps(model, pad_id: int, pixel_values: torch.Tensor, prefixes: list[list[int]], responses: list[list[int]]) -> torch.Tensor:
    """Sum of log-probabilities of each response given its prefix (right padded)."""
    device = pixel_values.device
    seqs = [p + r for p, r in zip(prefixes, responses)]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    ids, attn = ids.to(device), attn.to(device)
    logits = model(input_ids=ids, attention_mask=attn, pixel_values=pixel_values).logits
    rmax = max(len(r) for r in responses)
    pos = torch.zeros((len(seqs), rmax), dtype=torch.long)
    tgt = torch.full((len(seqs), rmax), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), rmax), dtype=torch.float32)
    for i, (p, r) in enumerate(zip(prefixes, responses)):
        n = len(r)
        pos[i, :n] = torch.arange(len(p) - 1, len(p) - 1 + n)
        tgt[i, :n] = torch.tensor(r)
        mask[i, :n] = 1.0
    pos, tgt, mask = pos.to(device), tgt.to(device), mask.to(device)
    rows = torch.arange(len(seqs), device=device).unsqueeze(1)
    sel = logits[rows, pos].float()  # (B, R, V)
    lp = F.log_softmax(sel, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(dim=1)


def subset(pairs: list[dict], fraction: float, seed: int) -> list[dict]:
    if fraction >= 1.0:
        return pairs
    ranked = sorted(pairs, key=lambda p: hashlib.sha256(f"{seed}:{p['pair_id']}".encode()).hexdigest())
    return sorted(ranked[: int(round(fraction * len(pairs)))], key=lambda p: p["pair_id"])


def run_step(model, runner, batch, micro, prompt_cache, device, beta) -> dict:
    """Accumulate C-DPO gradients of one global batch; returns summed statistics."""
    stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0}
    for start in range(0, len(batch), micro):
        mb = batch[start : start + micro]
        prefixes, responses, images = [], [], []
        for p in mb:
            if p["prompt"] not in prompt_cache:
                prompt_cache[p["prompt"]] = runner.prompt_ids(p["prompt"])
            pre = prompt_cache[p["prompt"]] + p["context_ids"]
            img = load_image(p["image"])
            prefixes += [pre, pre]
            responses += [p["chosen_ids"], p["rejected_ids"]]
            images += [img, img]
        lp = response_logps(model, runner.pad_id, runner.pixel_values(images), prefixes, responses)
        pc, pr = lp[0::2], lp[1::2]
        rc = torch.tensor([p["ref_chosen"] for p in mb], device=device)
        rr = torch.tensor([p["ref_rejected"] for p in mb], device=device)
        logits = (pc - pr) - (rc - rr)
        loss = -F.logsigmoid(beta * logits)
        (loss.sum() / len(batch)).backward()
        stats["loss"] += float(loss.detach().sum())
        stats["acc"] += float((logits > 0).float().sum())
        stats["margin"] += float(logits.detach().sum())
        stats["chosen_reward"] += float((beta * (pc - rc)).detach().sum())
        stats["rejected_reward"] += float((beta * (pr - rr)).detach().sum())
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="checkpoints/preferences/pairs.jsonl")
    ap.add_argument("--out", default="checkpoints/sentinel")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    ap.add_argument("--epochs", type=float, default=C.EPOCHS)
    ap.add_argument("--beta", type=float, default=C.DPO_BETA)
    ap.add_argument("--global-batch", type=int, default=C.GLOBAL_BATCH)
    ap.add_argument("--micro-batch", type=int, default=C.MICRO_BATCH)
    ap.add_argument("--data-fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many optimizer steps")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import LlavaForConditionalGeneration, get_cosine_schedule_with_warmup

    out = Path(args.out)
    if (out / "adapter" / "adapter_config.json").exists():
        log(f"{out} already trained; skipping")
        return
    seed_all(args.seed)
    pairs = subset(read_jsonl(Path(args.pairs)), args.data_fraction, args.seed)
    if not pairs:
        raise RuntimeError("no preference pairs")
    log(f"C-DPO on {len(pairs)} pairs, lr={args.lr}, epochs={args.epochs}, beta={args.beta}")

    device = torch.device(args.device)
    model = LlavaForConditionalGeneration.from_pretrained(LLAVA_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    runner = LlavaRunner(args.device, model=model)
    total_mem = torch.cuda.get_device_properties(device).total_memory / 2**30
    for p in model.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT, target_modules=LORA_TARGET, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if total_mem < 60:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable LoRA params: {sum(p.numel() for p in trainable) / 1e6:.1f}M (gpu mem {total_mem:.0f} GiB)")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(pairs) / args.global_batch)
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    total_steps = min(total_steps, math.ceil(args.epochs) * steps_per_epoch)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(C.WARMUP_RATIO * total_steps), total_steps)

    prompt_cache: dict[str, list[int]] = {}
    batches: list[list[dict]] = []
    for epoch in range(math.ceil(args.epochs)):
        ep = list(pairs)
        random.Random(args.seed + epoch).shuffle(ep)
        batches += [ep[i : i + args.global_batch] for i in range(0, len(ep), args.global_batch)]
    batches = batches[:total_steps]
    model.train()
    history = []
    t0 = time.time()
    micro = args.micro_batch
    for step, batch in enumerate(batches):
        while True:
            try:
                stats = run_step(model, runner, batch, micro, prompt_cache, device, args.beta)
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro > 1:
                    micro //= 2
                    log(f"OOM: micro-batch -> {micro}")
                elif not getattr(model, "is_gradient_checkpointing", False):
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    log("OOM: enabling gradient checkpointing")
                else:
                    raise
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        row = {k: v / len(batch) for k, v in stats.items()}
        row.update(step=step + 1, lr=sched.get_last_lr()[0], pairs=len(batch), minutes=(time.time() - t0) / 60)
        history.append(row)
        log("cdpo " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "num_pairs": len(pairs), "history": history}, indent=1) + "\n")
    log(f"saved adapter to {out / 'adapter'}")


if __name__ == "__main__":
    main()
