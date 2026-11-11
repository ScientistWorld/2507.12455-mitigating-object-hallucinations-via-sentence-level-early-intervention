"""Shared helpers for the SENTINEL reproduction on LLaVA-v1.5-7B."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image

# Jobs run with a single CPU core: avoid thread oversubscription.
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "data" / "downloads"
LLAVA_PATH = DL / "models" / "llava-1.5-7b-hf"
GDINO_PATH = DL / "models" / "grounding-dino-base"
OWL_PATH = DL / "models" / "owlv2-base-patch16-ensemble"
COCO_DIR = DL / "coco"
NLTK_DIR = DL / "nltk_data"

# LLaVA-v1.5 "vicuna_v1" conversation template (the one used by the official
# LLaVA-1.5 evaluation scripts).
SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
TRAIN_PROMPT = "Describe this image in detail."


def chat_prompt(question: str) -> str:
    return f"{SYSTEM} USER: <image>\n{question} ASSISTANT:"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, tolerating a truncated last line from an interrupted job."""
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_image(rel_path: str) -> Image.Image:
    return Image.open(COCO_DIR / rel_path).convert("RGB")


def expand2square(img: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """LLaVA-1.5 'pad' aspect-ratio handling: pad to a square with the CLIP mean colour."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new(img.mode, (side, side), background)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


class LlavaRunner:
    """LLaVA-v1.5-7B wrapper: prompt construction with raw token ids, batched
    generation with left padding, and response log-probabilities."""

    def __init__(self, device: str, dtype: torch.dtype = torch.bfloat16, model=None):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(LLAVA_PATH)
        self.tok = self.processor.tokenizer
        if model is None:
            model = LlavaForConditionalGeneration.from_pretrained(LLAVA_PATH, torch_dtype=dtype, low_cpu_mem_usage=True)
            model.to(self.device)
        self.model = model
        self.model.eval()
        mean = self.processor.image_processor.image_mean
        self.pad_color = tuple(int(255 * x) for x in mean)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.eos_id = self.tok.eos_token_id
        self.period_id = self.tok.convert_tokens_to_ids(".")
        cfg = self.model.config
        self.image_token_id = cfg.image_token_index
        vc = cfg.vision_config
        self.image_seq_length = (vc.image_size // vc.patch_size) ** 2  # 576 visual tokens
        # Prompts carry all 576 image tokens (expanded in prompt_ids).  transformers
        # 4.47 otherwise switches to its "legacy" path at every decode step (a single
        # new token has fewer than image_seq_length image tokens), which rebuilds the
        # attention mask without the left padding of batched generation.  With 0 the
        # legacy test is never true and the standard masked path is always used.
        cfg.image_seq_length = 0
        self._prompt_cache: dict[str, list[int]] = {}

    # -- inputs -------------------------------------------------------------
    def pixel_values(self, images: list[Image.Image]) -> torch.Tensor:
        sq = [expand2square(im, self.pad_color) for im in images]
        pv = self.processor.image_processor(sq, return_tensors="pt")["pixel_values"]
        return pv.to(self.device, dtype=self.model.dtype)

    def prompt_ids(self, question: str) -> list[int]:
        """Token ids of the full chat prompt, with the <image> placeholder
        expanded to the number of visual tokens (processor convention)."""
        if question not in self._prompt_cache:
            ids = self.tok(chat_prompt(question), add_special_tokens=True)["input_ids"]
            n = ids.count(self.image_token_id)
            if n == 1:  # expand the placeholder to one token per visual feature
                k = ids.index(self.image_token_id)
                ids = ids[:k] + [self.image_token_id] * self.image_seq_length + ids[k + 1 :]
            if ids.count(self.image_token_id) != self.image_seq_length:
                raise RuntimeError(f"prompt has {ids.count(self.image_token_id)} image tokens, expected {self.image_seq_length}")
            self._prompt_cache[question] = ids
        return list(self._prompt_cache[question])

    def text_ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _left_pad(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s) :] = torch.tensor(s, dtype=torch.long)
            mask[i, width - len(s) :] = 1
        return ids.to(self.device), mask.to(self.device)

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prefixes: list[list[int]],
        images: list[Image.Image],
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        stop_at_period: bool = False,
    ) -> list[list[int]]:
        """Return generated token ids (without padding / trailing eos) for each
        prefix x return sequence, in order prefix-major."""
        ids, mask = self._left_pad(prefixes)
        pv = self.pixel_values(images)
        eos = [self.eos_id, self.period_id] if stop_at_period else [self.eos_id]
        kwargs = dict(
            input_ids=ids,
            attention_mask=mask,
            pixel_values=pv,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos,
            pad_token_id=self.pad_id,
            num_return_sequences=num_return_sequences,
            use_cache=True,
        )
        if do_sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            kwargs.update(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**kwargs)
        gen = out[:, ids.shape[1] :].tolist()
        results = []
        for row in gen:
            toks = []
            for t in row:
                if t == self.eos_id:
                    toks.append(t)
                    break
                if t == self.pad_id:
                    break
                toks.append(t)
                if stop_at_period and t == self.period_id:
                    break
            results.append(toks)
        return results

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode([t for t in ids if t != self.eos_id], skip_special_tokens=True).strip()
