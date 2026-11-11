"""Object extraction from generated sentences (paper Sec. 3.2 / Appendix B.1).

The paper parses each sentence into scene-graph triplets, keeps subjects and
objects of relations (only the subject of attribute triplets), keeps nouns and
proper nouns by part of speech (spaCy), and removes non-concrete nouns with
WordNet lexicographer categories.  Here the dependency parse of spaCy supplies
the entities directly: the heads of noun phrases are exactly the subjects and
objects of the scene-graph relations, while attribute words are adjectives and
never become entities.  The same POS and WordNet filters are then applied.
"""

from __future__ import annotations

from functools import lru_cache

from .common import NLTK_DIR

# Non-concrete WordNet lexicographer categories removed by the paper.
NON_CONCRETE = {
    "noun.feeling", "noun.attribute", "noun.state", "noun.shape", "noun.time", "noun.quantity",
    "noun.cognition", "noun.event", "noun.communication", "noun.relation", "noun.act", "noun.location",
}
# Words describing the picture itself or layout rather than depicted objects.
PICTURE_WORDS = {
    "image", "picture", "photo", "photograph", "scene", "view", "shot", "frame", "background",
    "foreground", "side", "front", "back", "top", "bottom", "middle", "center", "centre", "corner",
    "edge", "left", "right", "area", "part", "portion", "piece", "variety", "number", "group",
    "couple", "lot", "pair", "set", "type", "kind", "array", "assortment", "collection", "row",
    "color", "colour", "shape", "atmosphere", "setting", "environment", "space", "moment", "day",
    "time", "detail", "element", "feature", "surface", "sense", "look", "appearance", "presence",
}


class ObjectExtractor:
    def __init__(self):
        import nltk
        import spacy

        nltk.data.path.insert(0, str(NLTK_DIR))
        from nltk.corpus import wordnet as wn

        self.wn = wn
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

    @lru_cache(maxsize=100_000)
    def is_concrete(self, lemma: str) -> bool:
        if lemma in PICTURE_WORDS or len(lemma) < 2:
            return False
        synsets = self.wn.synsets(lemma.replace(" ", "_"), pos="n")
        if not synsets:
            return False
        return synsets[0].lexname() not in NON_CONCRETE

    def extract_docs(self, docs) -> list[list[dict]]:
        out = []
        for doc in docs:
            seen: dict[str, dict] = {}
            for tok in doc:
                if tok.pos_ not in ("NOUN", "PROPN"):
                    continue
                if tok.dep_ == "compound":  # part of a larger noun phrase head
                    continue
                lemma = tok.lemma_.lower().strip()
                if not lemma.isalpha() or not self.is_concrete(lemma):
                    continue
                compounds = [c.lemma_.lower() for c in tok.lefts if c.dep_ == "compound" and c.pos_ in ("NOUN", "PROPN")]
                phrase = " ".join(compounds + [lemma])
                if lemma not in seen:
                    seen[lemma] = {"lemma": lemma, "phrase": phrase}
            out.append(list(seen.values()))
        return out

    def extract(self, sentences: list[str]) -> list[list[dict]]:
        return self.extract_docs(self.nlp.pipe(sentences, batch_size=64))
