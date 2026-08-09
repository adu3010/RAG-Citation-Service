"""Okapi BM25, implemented directly rather than pulled from a library.

Dense retrieval alone loses exact identifiers — drug codes, "SOC 3544", "Band 5",
error strings. Those are precisely the queries users care about, so the lexical
arm is not optional. Writing the scorer out also means the inverted index can be
serialised alongside the vector store instead of rebuilt on every boot.

score(q, d) = Σ_{t∈q} idf(t) · (f(t,d)·(k1+1)) / (f(t,d) + k1·(1 − b + b·|d|/avgdl))
idf(t)     = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
"""

from __future__ import annotations

import math
from collections import defaultdict

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._lengths: list[int] = []
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def average_length(self) -> float:
        return (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    def add(self, chunk_id: str, text: str) -> None:
        index = len(self._ids)
        self._ids.append(chunk_id)
        tokens = tokenize(text)
        self._lengths.append(len(tokens))
        for token in tokens:
            postings = self._postings[token]
            postings[index] = postings.get(index, 0) + 1

    def add_many(self, items: list[tuple[str, str]]) -> None:
        for chunk_id, text in items:
            self.add(chunk_id, text)

    def _idf(self, token: str) -> float:
        n = len(self._ids)
        df = len(self._postings.get(token, ()))
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs, highest score first."""
        if not self._ids:
            return []
        avgdl = self.average_length or 1.0
        scores: dict[int, float] = defaultdict(float)
        for token in set(tokenize(query)):
            postings = self._postings.get(token)
            if not postings:
                continue
            idf = self._idf(token)
            for doc, freq in postings.items():
                length_norm = 1.0 - self.b + self.b * (self._lengths[doc] / avgdl)
                scores[doc] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * length_norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self._ids[doc], score) for doc, score in ranked]

    # --- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "ids": self._ids,
            "lengths": self._lengths,
            # JSON keys must be strings; document indices are restored on load.
            "postings": {
                token: {str(doc): freq for doc, freq in postings.items()}
                for token, postings in self._postings.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict) -> BM25Index:
        index = cls(k1=raw.get("k1", 1.5), b=raw.get("b", 0.75))
        index._ids = list(raw["ids"])
        index._lengths = list(raw["lengths"])
        index._postings = defaultdict(dict)
        for token, postings in raw["postings"].items():
            index._postings[token] = {int(doc): freq for doc, freq in postings.items()}
        return index
