"""CHAIR-style object hallucination scoring over the 80 MSCOCO categories.

Follows Rohrbach et al. (2018) "Object Hallucination in Image Captioning" (the
metric behind Object HalBench): caption words are mapped through the standard
MSCOCO synonym list (``coco_synonyms.txt``, verbatim from the reference CHAIR
release) and the same double-word / qualifier rules, then compared with the
ground-truth object set of the image (segmentation categories plus objects
named in the human reference captions).

Two deliberate robustness changes relative to the original script:
  * objects are counted once per response (a set), so repeating a correct
    object cannot dilute the mention-level rate;
  * a mention directly preceded (within three words, same sentence) by a
    negation word ("no", "not", "without", "n't", ...) is not a claim that the
    object is present and is ignored for both hallucination and coverage.
Plural handling uses a vocabulary-constrained singulariser instead of the
``pattern`` package so the scorer has no external dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path

_SYNONYM_FILE = Path(__file__).with_name("coco_synonyms.txt")

_COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light", "traffic signal",
    "stop light", "fire hydrant", "stop sign", "parking meter", "suit case", "sports ball",
    "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
    "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie", "laptop computer",
    "stove top oven", "home plate", "train track",
]
_ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
_VEHICLE_WORDS = ["jet", "train"]
_IRREGULAR = {
    "men": "man", "women": "woman", "people": "people", "children": "children", "mice": "mouse",
    "geese": "goose", "oxen": "oxen", "policemen": "policeman", "servicemen": "serviceman",
    "gentlemen": "gentleman", "cowboys": "cowboy", "feet": "foot", "teeth": "tooth",
}
_NEGATORS = {"no", "not", "without", "nor", "neither", "never", "none", "nothing"}
_NEGATION_WINDOW = 3
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SENT_SPLIT_RE = re.compile(r"[.!?;:\n]+")


def _load_synonyms() -> tuple[dict[str, str], set[str]]:
    inverse: dict[str, str] = {}
    for line in _SYNONYM_FILE.read_text(encoding="utf-8").splitlines():
        words = [w.strip().lower() for w in line.split(",") if w.strip()]
        if not words:
            continue
        for w in words:
            inverse[w] = words[0]
    return inverse, set(inverse)


INVERSE_SYNONYMS, COCO_WORDS = _load_synonyms()
CATEGORIES = sorted(set(INVERSE_SYNONYMS.values()))

DOUBLE_WORD_MAP: dict[str, str] = {w: w for w in _COCO_DOUBLE_WORDS}
for _w in _ANIMAL_WORDS:
    DOUBLE_WORD_MAP[f"baby {_w}"] = _w
    DOUBLE_WORD_MAP[f"adult {_w}"] = _w
for _w in _VEHICLE_WORDS:
    DOUBLE_WORD_MAP[f"passenger {_w}"] = _w
DOUBLE_WORD_MAP["bow tie"] = "tie"
DOUBLE_WORD_MAP["toilet seat"] = "toilet"

_VOCAB_TOKENS: set[str] = set()
for _phrase in list(COCO_WORDS) + list(DOUBLE_WORD_MAP):
    _VOCAB_TOKENS.update(_phrase.split())


def singularize(word: str) -> str:
    if word in _VOCAB_TOKENS:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    candidates = []
    if word.endswith("ies"):
        candidates.append(word[:-3] + "y")
    if word.endswith("ves"):
        candidates += [word[:-3] + "f", word[:-3] + "fe"]
    if word.endswith("es"):
        candidates.append(word[:-2])
    if word.endswith("s"):
        candidates.append(word[:-1])
    for cand in candidates:
        if cand in _VOCAB_TOKENS:
            return cand
    return word


def _is_negator(token: str) -> bool:
    return token in _NEGATORS or token.endswith("n't")


def caption_objects(caption: str) -> set[str]:
    """Return the set of MSCOCO categories asserted as present in ``caption``."""
    found: set[str] = set()
    if not isinstance(caption, str):
        return found
    for sentence in _SENT_SPLIT_RE.split(caption.lower()):
        raw = _TOKEN_RE.findall(sentence)
        words = [singularize(w) for w in raw]
        merged: list[tuple[str, int]] = []
        i = 0
        while i < len(words):
            pair = " ".join(words[i : i + 2])
            if pair in DOUBLE_WORD_MAP:
                merged.append((DOUBLE_WORD_MAP[pair], i))
                i += 2
            else:
                merged.append((words[i], i))
                i += 1
        tokens = [w for w, _ in merged]
        if "toilet" in tokens and "seat" in tokens:
            merged = [(w, i) for w, i in merged if w != "seat"]
        for word, start in merged:
            if word not in COCO_WORDS:
                continue
            window = raw[max(0, start - _NEGATION_WINDOW) : start]
            if any(_is_negator(t) for t in window):
                continue
            found.add(INVERSE_SYNONYMS[word])
    return found


def score_captions(items: list[tuple[str, set[str]]]) -> dict[str, float]:
    """items: (generated caption, gold category set).  Percentages."""
    n = len(items)
    hallucinated_responses = 0
    mentions = 0
    hallucinated_mentions = 0
    covered = 0
    gold_total = 0
    for caption, gold in items:
        mentioned = caption_objects(caption)
        bad = mentioned - gold
        hallucinated_responses += int(bool(bad))
        mentions += len(mentioned)
        hallucinated_mentions += len(bad)
        covered += len(mentioned & gold)
        gold_total += len(gold)
    return {
        "response_hallucination_rate": 100.0 * hallucinated_responses / max(1, n),
        "mention_hallucination_rate": 100.0 * hallucinated_mentions / mentions if mentions else 0.0,
        "object_coverage": 100.0 * covered / max(1, gold_total),
    }
