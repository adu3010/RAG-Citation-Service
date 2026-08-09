"""In-process vector store.

A flat numpy matrix with exact cosine search. For corpora up to ~10^5 chunks this
beats a vector database on latency and removes an entire piece of infrastructure
from the deployment. The interface is narrow on purpose: ``add`` / ``search`` /
``save`` / ``load`` is all a FAISS or pgvector swap would need to satisfy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def get(self, chunk_id: str) -> Chunk | None:
        index = self._by_id.get(chunk_id)
        return self._chunks[index] if index is not None else None

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        index = self._by_id.get(chunk_id)
        return self._vectors[index] if index is not None else None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"got {len(chunks)} chunks but {vectors.shape[0]} vectors")
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")

        # Re-ingesting a document should update it, not duplicate it.
        fresh_chunks: list[Chunk] = []
        fresh_rows: list[np.ndarray] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            existing = self._by_id.get(chunk.id)
            if existing is not None:
                self._chunks[existing] = chunk
                self._vectors[existing] = vector
                continue
            fresh_chunks.append(chunk)
            fresh_rows.append(vector)

        if fresh_chunks:
            start = len(self._chunks)
            self._chunks.extend(fresh_chunks)
            self._vectors = np.vstack([self._vectors, np.asarray(fresh_rows, dtype=np.float32)])
            for offset, chunk in enumerate(fresh_chunks):
                self._by_id[chunk.id] = start + offset

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if len(self._chunks) == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        scores = self._vectors @ query
        k = min(k, scores.shape[0])
        # argpartition avoids a full sort of the corpus on every query.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[i].id, float(scores[i])) for i in top]

    # --- persistence -----------------------------------------------------
    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        (path / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self._chunks], indent=2), encoding="utf-8"
        )
        (path / "store_meta.json").write_text(json.dumps({"dim": self.dim}), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        path = Path(directory)
        meta = json.loads((path / "store_meta.json").read_text(encoding="utf-8"))
        store = cls(dim=int(meta["dim"]))
        vectors = np.load(path / "vectors.npy")
        chunks = [Chunk.from_dict(raw) for raw in json.loads((path / "chunks.json").read_text())]
        store.add(chunks, vectors)
        return store
