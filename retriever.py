"""Hybrid retrieval: BM25 + dense, fused with RRF, then diversified with MMR.

Why RRF instead of a weighted score blend: BM25 scores are unbounded and corpus
dependent while cosine similarity sits in [-1, 1]. Any fixed alpha in
``alpha·bm25 + (1-alpha)·cosine`` needs retuning whenever the corpus grows.
Reciprocal rank fusion only consumes ranks, so it is scale-free and has one
parameter that rarely needs touching.

    RRF(d) = Σ_r 1 / (k + rank_r(d))

MMR then trades relevance against novelty so the top-k is not five near-duplicate
chunks from the same page — which matters a lot for citation quality, because
duplicate evidence looks like corroboration when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25Index
from .embeddings import Embedder
from .store import VectorStore


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    fused_score: float
    dense_rank: int | None
    lexical_rank: int | None
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "fused_score": round(self.fused_score, 6),
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
            "metadata": self.metadata,
        }


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if len(self.store) == 0:
            return []

        query_vector = self.embedder.encode([question])[0]
        dense = self.store.search(query_vector, k=self.candidate_k)
        lexical = self.bm25.search(question, k=self.candidate_k)

        dense_ranks = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
        lexical_ranks = {cid: i + 1 for i, (cid, _) in enumerate(lexical)}
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in dense], [cid for cid, _ in lexical]], k=self.rrf_k
        )
        if not fused:
            return []

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        selected = self._mmr(query_vector, ordered, top_k)

        results: list[RetrievedChunk] = []
        for chunk_id in selected:
            chunk = self.store.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    fused_score=fused[chunk_id],
                    dense_rank=dense_ranks.get(chunk_id),
                    lexical_rank=lexical_ranks.get(chunk_id),
                    metadata=chunk.metadata,
                )
            )
        return results

    def _mmr(
        self, query_vector: np.ndarray, ordered: list[tuple[str, float]], top_k: int
    ) -> list[str]:
        """Maximal Marginal Relevance over the fused candidate pool."""
        candidates = [cid for cid, _ in ordered]
        if self.mmr_lambda >= 1.0 or top_k >= len(candidates):
            return candidates[:top_k]

        relevance = dict(ordered)
        # Normalise fused scores so lambda has consistent meaning across queries.
        best = max(relevance.values()) or 1.0
        relevance = {cid: score / best for cid, score in relevance.items()}

        selected: list[str] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            best_cid, best_value = remaining[0], -float("inf")
            for cid in remaining:
                penalty = 0.0
                if selected:
                    vector = self.store.vector_for(cid)
                    if vector is not None:
                        similarities = [
                            float(vector @ other)
                            for other in (self.store.vector_for(s) for s in selected)
                            if other is not None
                        ]
                        penalty = max(similarities) if similarities else 0.0
                value = self.mmr_lambda * relevance[cid] - (1.0 - self.mmr_lambda) * penalty
                if value > best_value:
                    best_cid, best_value = cid, value
            selected.append(best_cid)
            remaining.remove(best_cid)
        return selected
