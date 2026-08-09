"""Embedding providers behind a single Protocol.

The default provider is a deterministic feature-hashing embedder. That is an
intentional engineering choice, not a shortcut:

* CI stays hermetic and fast — no model weights to download, no flaky network.
* Eval numbers are bit-for-bit reproducible across machines.
* Swapping in a real sentence-transformer is a one-line config change, which is
  exactly the property you want when a model gets deprecated in production.

``HashingEmbedder`` uses ``hashlib.blake2b`` rather than the builtin ``hash()``
because Python salts string hashing per process, which would make a persisted
index unreadable after a restart.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

import numpy as np

from .text import bigrams, tokenize


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an L2-normalised ``(len(texts), dim)`` float32 matrix."""
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Signed feature hashing over unigrams and bigrams with sublinear term weighting."""

    def __init__(self, dim: int = 512) -> None:
        if dim < 32:
            raise ValueError("dim must be >= 32 to keep collisions tolerable")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        # Low bit picks the sign so collisions cancel rather than compound.
        return value % self.dim, 1.0 if value & 1 else -1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            unigrams = tokenize(text)
            counts: dict[str, int] = {}
            for token in unigrams + bigrams(unigrams):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                index, sign = self._hash(token)
                out[row, index] += sign * (1.0 + math.log(count))
        return _l2_normalise(out)


class SentenceTransformerEmbedder:
    """Production provider. Imported lazily so the base install stays light."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install the extra with `pip install -e '.[dense]'`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs weights
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.astype(np.float32)


def get_embedder(kind: str, *, dim: int = 512, model_name: str = "") -> Embedder:
    kind = kind.lower()
    if kind == "hashing":
        return HashingEmbedder(dim=dim)
    if kind in {"sentence-transformer", "sentence_transformers", "dense"}:
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"unknown embedder: {kind!r} (expected 'hashing' or 'sentence-transformer')")
