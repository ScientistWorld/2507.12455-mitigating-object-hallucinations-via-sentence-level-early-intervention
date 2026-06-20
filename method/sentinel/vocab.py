from __future__ import annotations

COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "triangle"]
OBJECTS = [f"{c} {s}" for c in COLORS for s in SHAPES]

SPECIAL = ["<pad>", "<bos>", "<eos>"]
WORDS = SPECIAL + [
    "Describe",
    "this",
    "image",
    "in",
    "detail",
    "There",
    "is",
    "a",
    "and",
    "yes",
    "no",
    "?",
    ".",
] + COLORS + SHAPES

PAD, BOS, EOS = 0, 1, 2
WORD_TO_ID = {w: i for i, w in enumerate(WORDS)}
ID_TO_WORD = {i: w for w, i in WORD_TO_ID.items()}


def normalize(text: str) -> list[str]:
    text = text.replace("?", " ? ").replace(".", " . ")
    return [tok for tok in text.split() if tok]


def encode(text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    ids = [WORD_TO_ID[tok] for tok in normalize(text) if tok in WORD_TO_ID]
    if add_bos:
        ids = [BOS] + ids
    if add_eos:
        ids = ids + [EOS]
    return ids


def decode(ids: list[int]) -> str:
    toks = []
    for idx in ids:
        if idx == EOS:
            break
        if idx in (PAD, BOS):
            continue
        tok = ID_TO_WORD.get(int(idx), "")
        if tok == ".":
            if toks:
                toks[-1] = toks[-1] + "."
        elif tok == "?":
            if toks:
                toks[-1] = toks[-1] + "?"
        else:
            toks.append(tok)
    return " ".join(toks)


def extract_objects(text: str) -> list[str]:
    low = text.lower()
    return [obj for obj in OBJECTS if obj in low]


def object_vector(objects: list[str]) -> list[float]:
    present = set(objects)
    return [1.0 if obj in present else 0.0 for obj in OBJECTS]
