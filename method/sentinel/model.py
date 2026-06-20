from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .vocab import BOS, EOS, PAD, decode


class TinyVLM(nn.Module):
    """A compact image-conditioned language model for SENTINEL experiments."""

    def __init__(self, n_objects: int, vocab_size: int, hidden: int = 96):
        super().__init__()
        self.hidden = hidden
        self.image_proj = nn.Sequential(nn.Linear(n_objects, hidden), nn.Tanh())
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.visual_lm_bias = nn.Linear(n_objects, vocab_size, bias=False)

    def forward(self, image_vec: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        h0 = self.image_proj(image_vec).unsqueeze(0)
        emb = self.embed(tokens)
        out, _ = self.gru(emb, h0)
        return self.lm_head(out) + self.visual_lm_bias(image_vec).unsqueeze(1)

    def sequence_logprob(self, image_vec: torch.Tensor, prefix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seqs = []
        masks = []
        for idx in range(prefix.shape[0]):
            clean_prefix = prefix[idx][prefix[idx] != PAD]
            clean_target = target[idx][target[idx] != PAD]
            seq = torch.cat([clean_prefix, clean_target])
            start = max(clean_prefix.numel() - 1, 0)
            mask = torch.zeros(seq.numel() - 1, dtype=torch.float32, device=seq.device)
            mask[start:] = 1.0
            seqs.append(seq)
            masks.append(mask)
        inputs = pad_sequence([seq[:-1] for seq in seqs], batch_first=True, padding_value=PAD)
        labels = pad_sequence([seq[1:] for seq in seqs], batch_first=True, padding_value=PAD)
        mask = pad_sequence(masks, batch_first=True, padding_value=0.0)
        logits = self.forward(image_vec, inputs)
        logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (logp * mask).sum(dim=1)

    @torch.no_grad()
    def generate(
        self,
        image_vec: torch.Tensor,
        prefix: list[int],
        max_new_tokens: int = 36,
        temperature: float = 0.0,
    ) -> str:
        device = image_vec.device
        ids = list(prefix) if prefix else [BOS]
        prefix_len = len(ids)
        finished = False
        for _ in range(max_new_tokens):
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            logits = self.forward(image_vec.unsqueeze(0), inp)[:, -1, :]
            logits[:, PAD] = -1e9
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
