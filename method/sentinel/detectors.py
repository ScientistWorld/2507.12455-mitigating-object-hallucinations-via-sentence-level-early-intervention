"""Object presence validation by cross-checking two open-vocabulary detectors
(paper Sec. 3.2 "Object presence validation", Appendix B.3/B.4).

Detector A: GroundingDINO (as in the paper).  Detector B: OWLv2, an
open-vocabulary detector standing in for YOLO-World (the paper notes any
open-world detector can be used; Table 8 compares several).  For each
candidate object an image-level presence score is the maximum detection score
of that phrase over all predicted boxes.  Both present -> factual, both absent
-> hallucinated, disagreement -> uncertain.
"""

from __future__ import annotations

import torch

from .common import GDINO_PATH, OWL_PATH

GDINO_BOX_THRESHOLD = 0.35  # GroundingDINO default box threshold
OWL_THRESHOLD = 0.10        # OWLv2 default detection threshold


class CrossChecker:
    def __init__(self, device: str):
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Owlv2ForObjectDetection,
            Owlv2Processor,
        )

        self.device = torch.device(device)
        self.g_proc = AutoProcessor.from_pretrained(GDINO_PATH)
        self.g_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_PATH).to(self.device).eval()
        self.o_proc = Owlv2Processor.from_pretrained(OWL_PATH)
        self.o_model = Owlv2ForObjectDetection.from_pretrained(OWL_PATH).to(self.device).eval()

    @torch.no_grad()
    def gdino_scores(self, image, phrases: list[str]) -> list[float]:
        scores: list[float] = []
        # GroundingDINO text is limited to 256 tokens; chunk long phrase lists.
        for start in range(0, len(phrases), 12):
            chunk = phrases[start : start + 12]
            text = " . ".join(chunk) + " ."
            inputs = self.g_proc(images=image, text=text, return_tensors="pt").to(self.device)
            logits = self.g_model(**inputs).logits.sigmoid()[0]  # (queries, text tokens)
            enc = self.g_proc.tokenizer(text, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            pos = 0
            for phrase in chunk:
                s = text.index(phrase, pos)
                e = s + len(phrase)
                pos = e
                toks = [i for i, (a, b) in enumerate(offsets) if b > a and a >= s and b <= e and i < logits.shape[1]]
                scores.append(float(logits[:, toks].max()) if toks else 0.0)
        return scores

    @torch.no_grad()
    def owl_scores(self, image, phrases: list[str]) -> list[float]:
        queries = [f"a photo of a {p}" for p in phrases]
        inputs = self.o_proc(text=[queries], images=image, return_tensors="pt").to(self.device)
        logits = self.o_model(**inputs).logits[0].sigmoid()  # (patches, queries)
        return [float(v) for v in logits.max(dim=0).values]

    def validate(self, image, phrases: list[str]) -> dict[str, dict]:
        """phrase -> {'gdino': score, 'owl': score, 'label': factual|hallucinated|uncertain}"""
        if not phrases:
            return {}
        g = self.gdino_scores(image, phrases)
        o = self.owl_scores(image, phrases)
        out = {}
        for p, gs, os_ in zip(phrases, g, o):
            a = gs >= GDINO_BOX_THRESHOLD
            b = os_ >= OWL_THRESHOLD
            label = "factual" if (a and b) else ("hallucinated" if (not a and not b) else "uncertain")
            out[p] = {"gdino": round(gs, 4), "owl": round(os_, 4), "label": label}
        return out
