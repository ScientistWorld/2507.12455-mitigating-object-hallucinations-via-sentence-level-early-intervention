from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .vocab import BOS, COLORS, EOS, OBJECTS, PAD, SHAPES, WORD_TO_ID, decode, extract_objects


class TinyVLM(nn.Module):
    """A compact image-conditioned language model for SENTINEL experiments."""

    def __init__(self, n_objects: int, vocab_size: int, hidden: int = 96):
        super().__init__()
        self.hidden = hidden
        self.image_proj = nn.Sequential(nn.Linear(n_objects, hidden), nn.Tanh())
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.object_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_objects))

    def forward(self, image_vec: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        h0 = self.image_proj(image_vec).unsqueeze(0)
        emb = self.embed(tokens)
        out, _ = self.gru(emb, h0)
        return self.lm_head(out)

    def object_logits(self, image_vec: torch.Tensor) -> torch.Tensor:
        return self.object_head(self.image_proj(image_vec))

    def sequence_logprob(self, image_vec: torch.Tensor, prefix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = []
        for idx in range(prefix.shape[0]):
            clean_prefix = prefix[idx][prefix[idx] != PAD]
            clean_target = target[idx][target[idx] != PAD]
            seq = torch.cat([clean_prefix, clean_target]).unsqueeze(0)
            logits = self.forward(image_vec[idx : idx + 1], seq[:, :-1])
            labels = seq[:, 1:]
            logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            start = max(clean_prefix.numel() - 1, 0)
            scores.append(logp[:, start:].sum())
        return torch.stack(scores)

    @torch.no_grad()
    def generate(
        self,
        image_vec: torch.Tensor,
        prefix: list[int],
        max_new_tokens: int = 36,
        temperature: float = 0.0,
        object_bias: float = 0.0,
        repeat_penalty: float = 0.0,
    ) -> str:
        device = image_vec.device
        ids = list(prefix) if prefix else [BOS]
        prefix_len = len(ids)
        finished = False
        obj_logits = self.object_logits(image_vec.unsqueeze(0)).squeeze(0)
        object_index = {obj: i for i, obj in enumerate(OBJECTS)}
        for _ in range(max_new_tokens):
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            logits = self.forward(image_vec.unsqueeze(0), inp)[:, -1, :]
            logits[:, PAD] = -1e9
            if object_bias:
                self._apply_object_bias(logits[0], ids[prefix_len:], obj_logits, object_index, object_bias, repeat_penalty)
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
            else:
                nxt = int(torch.argmax(logits, dim=-1).item())
            ids.append(nxt)
            if nxt == EOS:
                finished = True
                break
        if not finished:
            ids.append(EOS)
        return decode(ids[prefix_len:])

    @staticmethod
    def _apply_object_bias(
        logits: torch.Tensor,
        generated_ids: list[int],
        obj_logits: torch.Tensor,
        object_index: dict[str, int],
        object_bias: float,
        repeat_penalty: float,
    ) -> None:
        """Bias only object-word positions using the model's learned visual scores."""
        last = generated_ids[-1] if generated_ids else None
        seen = set(extract_objects(decode(generated_ids)))
        if last == WORD_TO_ID["a"]:
            for color in COLORS:
                color_candidates = []
                for shape in SHAPES:
                    obj = f"{color} {shape}"
                    score = obj_logits[object_index[obj]]
                    if obj in seen:
                        score = score - repeat_penalty
                    color_candidates.append(score)
                color_score = torch.stack(color_candidates).max()
                logits[WORD_TO_ID[color]] += object_bias * color_score
        elif last in [WORD_TO_ID[color] for color in COLORS]:
            color = next(color for color in COLORS if WORD_TO_ID[color] == last)
            for shape in SHAPES:
                obj = f"{color} {shape}"
                score = obj_logits[object_index[obj]]
                if obj in seen:
                    score = score - repeat_penalty
                logits[WORD_TO_ID[shape]] += object_bias * score
