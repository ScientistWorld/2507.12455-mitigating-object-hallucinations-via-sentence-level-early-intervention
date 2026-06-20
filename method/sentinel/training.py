from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import TinyVLM
from .vocab import BOS, EOS, OBJECTS, WORDS, encode, extract_objects, object_vector


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sentence_for_object(obj: str) -> str:
    return f"There is a {obj}."


def noisy_caption(objects: list[str], rng: random.Random, hallucination_prob: float = 0.7) -> str:
    # The base MLLM is intentionally imperfect: it sees real objects but also
    # learns a language-prior bias toward plausible absent objects.
    ordered = list(objects)
    rng.shuffle(ordered)
    if rng.random() < hallucination_prob:
        absent = [o for o in OBJECTS if o not in objects]
        ordered.append(rng.choice(absent))
    return " ".join(sentence_for_object(o) for o in ordered)


def batch_supervised(rows: list[dict], rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, inputs, labels = [], [], []
    for row in rows:
        caption = noisy_caption(row["objects"], rng)
        seq = encode(row["prompt"] + " " + caption, add_bos=True, add_eos=True)
        images.append(torch.tensor(object_vector(row["objects"]), dtype=torch.float32))
        inputs.append(torch.tensor(seq[:-1], dtype=torch.long))
        labels.append(torch.tensor(seq[1:], dtype=torch.long))
    return torch.stack(images), pad_sequence(inputs, batch_first=True), pad_sequence(labels, batch_first=True)


def train_sft(model: TinyVLM, train_rows: list[dict], out_dir: Path, args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.sft_lr)
    for epoch in range(args.sft_epochs):
        rng.shuffle(train_rows)
        for start in range(0, len(train_rows), args.batch_size):
            rows = train_rows[start : start + args.batch_size]
            images, inputs, labels = batch_supervised(rows, rng)
            logits = model(images, inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "baseline.pt")


def detector_color_shape(objects: set[str], candidate: str) -> bool | None:
    return candidate in objects


def detector_shape_color(objects: set[str], candidate: str) -> bool | None:
    # Independent implementation path used as the second validator.  The small
    # ambiguity rate creates the same "uncertain" branch SENTINEL filters out.
    color, shape = candidate.split()
    reconstructed = {f"{obj.split()[0]} {obj.split()[1]}" for obj in objects}
    if color == "yellow" and shape == "triangle":
        return None
    return f"{color} {shape}" in reconstructed


def validate_sentence(sentence: str, gold_objects: list[str]) -> str:
    mentioned = extract_objects(sentence)
    if not mentioned:
        return "uncertain"
    gold = set(gold_objects)
    factual = 0
    hallucinated = 0
    for obj in mentioned:
        a = detector_color_shape(gold, obj)
        b = detector_shape_color(gold, obj)
        if a is True and b is True:
            factual += 1
        elif a is False and b is False:
            hallucinated += 1
        else:
            return "uncertain"
    if hallucinated:
        return "hallucinated"
    if factual:
        return "non_hallucinated"
    return "uncertain"


def first_sentence(text: str) -> str:
    parts = text.split(".")
    return (parts[0].strip() + ".") if parts and parts[0].strip() else text.strip()


def build_preferences(model: TinyVLM, rows: list[dict], out_path: Path, args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed + 9)
    pairs = []
    for row in rows[: args.preference_images]:
        context = ""
        image = torch.tensor(object_vector(row["objects"]), dtype=torch.float32)
        for _ in range(args.icb_steps):
            positives, negatives = [], []
            prefix = encode(row["prompt"] + " " + context, add_bos=True, add_eos=False)
            for _sample in range(args.samples_per_context):
                sent = first_sentence(model.generate(image, prefix, max_new_tokens=18, temperature=args.sample_temperature))
                label = validate_sentence(sent, row["objects"])
                if label == "non_hallucinated":
                    positives.append(sent)
                elif label == "hallucinated":
                    negatives.append(sent)
            if not positives or not negatives:
                # SENTINEL preference pairs must come from in-domain model
                # samples.  If the current sampling batch lacks either class,
                # skip this context rather than fabricating oracle pairs.
                continue
            positive = rng.choice(positives)
            negative = rng.choice(negatives)
            pairs.append({"id": row["id"], "prompt": row["prompt"], "objects": row["objects"], "context": context, "chosen": positive, "rejected": negative})
            context = (context + " " + positive).strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, sort_keys=True) + "\n")
    if not pairs:
        raise RuntimeError("No in-domain SENTINEL preference pairs were sampled; increase samples_per_context or adjust the base model.")
    return pairs


def dpo_batch(pairs: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, prefixes, chosen, rejected = [], [], [], []
    for pair in pairs:
        prefix_text = pair["prompt"] + " " + pair["context"]
        images.append(torch.tensor(object_vector(pair["objects"]), dtype=torch.float32))
        prefixes.append(torch.tensor(encode(prefix_text, add_bos=True, add_eos=False), dtype=torch.long))
        chosen.append(torch.tensor(encode(pair["chosen"], add_bos=False, add_eos=True), dtype=torch.long))
        rejected.append(torch.tensor(encode(pair["rejected"], add_bos=False, add_eos=True), dtype=torch.long))
    return (
        torch.stack(images),
        pad_sequence(prefixes, batch_first=True),
        pad_sequence(chosen, batch_first=True),
        pad_sequence(rejected, batch_first=True),
    )


def train_cdpo(policy: TinyVLM, ref: TinyVLM, pairs: list[dict], out_dir: Path, args: argparse.Namespace) -> None:
    opt = torch.optim.AdamW(policy.parameters(), lr=args.dpo_lr)
    ref.eval()
    for _epoch in range(args.dpo_epochs):
        random.Random(args.seed + _epoch).shuffle(pairs)
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            images, prefix, chosen, rejected = dpo_batch(batch)
            policy_chosen = policy.sequence_logprob(images, prefix, chosen)
            policy_rejected = policy.sequence_logprob(images, prefix, rejected)
            with torch.no_grad():
                ref_chosen = ref.sequence_logprob(images, prefix, chosen)
                ref_rejected = ref.sequence_logprob(images, prefix, rejected)
            logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            loss = -F.logsigmoid(args.dpo_beta * logits).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(policy.state_dict(), out_dir / "sentinel.pt")


def predict(model: TinyVLM, rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            image = torch.tensor(object_vector(row["objects"]), dtype=torch.float32)
            prefix = encode(row["prompt"], add_bos=True)
            pred = model.generate(image, prefix, max_new_tokens=42, temperature=0.0)
            qa_preds = []
            for qa in row["qa"]:
                # Use generated caption as the model's answerable state.
                mentioned = set(extract_objects(pred))
                answer = "yes" if qa["object"] in mentioned else "no"
                qa_preds.append({**qa, "prediction": answer})
            f.write(json.dumps({**row, "prediction": pred, "qa_predictions": qa_preds}, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/train_scenes.jsonl")
    parser.add_argument("--eval-data", default="data/eval_scenes.jsonl")
    parser.add_argument("--out-dir", default="checkpoints/sentinel")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sft-epochs", type=int, default=10)
    parser.add_argument("--dpo-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sft-lr", type=float, default=3e-3)
    parser.add_argument("--dpo-lr", type=float, default=7e-4)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--preference-images", type=int, default=500)
    parser.add_argument("--icb-steps", type=int, default=2)
    parser.add_argument("--samples-per-context", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=1.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = read_jsonl(Path(args.train_data))
    eval_rows = read_jsonl(Path(args.eval_data))
    out_dir = Path(args.out_dir)

    baseline = TinyVLM(len(OBJECTS), len(WORDS))
    train_sft(baseline, train_rows, out_dir, args)
    predict(baseline, eval_rows, Path(args.outputs) / "baseline_predictions.jsonl")

    ref = copy.deepcopy(baseline)
    policy = copy.deepcopy(baseline)
    pairs = build_preferences(baseline, train_rows, out_dir / "preference_pairs.jsonl", args)
    train_cdpo(policy, ref, pairs, out_dir, args)
    predict(policy, eval_rows, Path(args.outputs) / "sentinel_predictions.jsonl")


if __name__ == "__main__":
    main()
