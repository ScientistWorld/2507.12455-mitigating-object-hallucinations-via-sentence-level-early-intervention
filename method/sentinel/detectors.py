"""Object presence validation by cross-checking two open-vocabulary detectors
(paper Sec. 3.2 "Object presence validation", Appendix B.3/B.4).

Detector A: GroundingDINO (as in the paper).  Detector B: OWLv2, an
open-vocabulary detector standing in for YOLO-World (the paper notes any
open-world detector can be used; Table 8 compares several).  For each
candidate object an image-level presence score is the maximum detection score
of that phrase over all predicted boxes.  Both present -> factual, both absent
-> hallucinated, disagreement -> uncertain.

Image preprocessing follows each detector's processor configuration and is
done once per image on the GPU (jobs have a single CPU core).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .common import GDINO_PATH, OWL_PATH

GDINO_BOX_THRESHOLD = 0.35  # GroundingDINO default box threshold
OWL_THRESHOLD = 0.10        # OWLv2 default detection threshold
OWL_MAX_TEXT = 16           # OWLv2 text encoder length


class CrossChecker:
    def __init__(self, device: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor

        self.device = torch.device(device)
        self.g_proc = AutoProcessor.from_pretrained(GDINO_PATH)
        self.g_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_PATH).to(self.device).eval()
        self.o_proc = Owlv2Processor.from_pretrained(OWL_PATH)
        self.o_model = Owlv2ForObjectDetection.from_pretrained(OWL_PATH).to(self.device).eval()
        gp, op = self.g_proc.image_processor, self.o_proc.image_processor
        self.g_short, self.g_long = gp.size["shortest_edge"], gp.size["longest_edge"]
        self.g_mean = torch.tensor(gp.image_mean, device=self.device).view(3, 1, 1)
        self.g_std = torch.tensor(gp.image_std, device=self.device).view(3, 1, 1)
        self.o_size = (op.size["height"], op.size["width"])
        self.o_mean = torch.tensor(op.image_mean, device=self.device).view(3, 1, 1)
        self.o_std = torch.tensor(op.image_std, device=self.device).view(3, 1, 1)

    @torch.no_grad()
    def prepare(self, image) -> dict:
        """Detector inputs for one PIL image (cached by the caller)."""
        x = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).to(self.device).permute(2, 0, 1).float() / 255.0
        h, w = x.shape[1:]
        # GroundingDINO: shortest edge 800, longest <= 1333, ImageNet normalisation.
        scale = min(self.g_short / min(h, w), self.g_long / max(h, w))
        gh, gw = int(round(h * scale)), int(round(w * scale))
        g = F.interpolate(x[None], size=(gh, gw), mode="bilinear", align_corners=False, antialias=True)[0]
        g = ((g - self.g_mean) / self.g_std)[None]
        # OWLv2: pad bottom/right to a square with 0.5 grey, resize to 960x960, CLIP normalisation.
        side = max(h, w)
        o = torch.full((3, side, side), 0.5, device=self.device)
        o[:, :h, :w] = x
        o = F.interpolate(o[None], size=self.o_size, mode="bilinear", align_corners=False, antialias=True)[0]
        o = ((o - self.o_mean) / self.o_std)[None]
        return {"gdino": g, "gdino_mask": torch.ones((1, gh, gw), dtype=torch.long, device=self.device), "owl": o}

    @torch.no_grad()
    def gdino_scores(self, prep: dict, phrases: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(phrases), 12):  # text is limited to 256 tokens
            chunk = phrases[start : start + 12]
            text = " . ".join(chunk) + " ."
            enc = self.g_proc.tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.g_model(pixel_values=prep["gdino"], pixel_mask=prep["gdino_mask"], **enc).logits.sigmoid()[0]
            pos = 0
            for phrase in chunk:
                s = text.index(phrase, pos)
                e = s + len(phrase)
                pos = e
                toks = [i for i, (a, b) in enumerate(offsets) if b > a and a >= s and b <= e and i < logits.shape[1]]
                scores.append(float(logits[:, toks].max()) if toks else 0.0)
        return scores

    @torch.no_grad()
    def owl_scores(self, prep: dict, phrases: list[str]) -> list[float]:
        queries = [f"a photo of a {p}" for p in phrases]
        enc = self.o_proc.tokenizer(queries, padding="max_length", truncation=True, max_length=OWL_MAX_TEXT, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.o_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], pixel_values=prep["owl"]).logits[0].sigmoid()
        return [float(v) for v in logits.max(dim=0).values]

    def validate(self, prep: dict, phrases: list[str]) -> dict[str, dict]:
        """phrase -> {'gdino': score, 'owl': score, 'label': factual|hallucinated|uncertain}"""
        if not phrases:
            return {}
        g = self.gdino_scores(prep, phrases)
        o = self.owl_scores(prep, phrases)
        out = {}
        for p, gs, os_ in zip(phrases, g, o):
            a = gs >= GDINO_BOX_THRESHOLD
            b = os_ >= OWL_THRESHOLD
            label = "factual" if (a and b) else ("hallucinated" if (not a and not b) else "uncertain")
            out[p] = {"gdino": round(gs, 4), "owl": round(os_, 4), "label": label}
        return out
