"""Runtime configuration.

Plain dataclass + ``os.environ`` rather than a settings library: one less
dependency, and every knob is visible in one screen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


@dataclass(frozen=True)
class Settings:
    # --- indexing ---
    chunk_tokens: int = field(default_factory=lambda: _env_int("RAG_CHUNK_TOKENS", 180))
    chunk_overlap_tokens: int = field(
        default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP_TOKENS", 40)
    )

    # --- retrieval ---
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("RAG_CANDIDATE_K", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))
    mmr_lambda: float = field(default_factory=lambda: _env_float("RAG_MMR_LAMBDA", 0.7))

    # --- providers ---
    embedder: str = field(default_factory=lambda: os.environ.get("RAG_EMBEDDER", "hashing"))
    embedding_dim: int = field(default_factory=lambda: _env_int("RAG_EMBEDDING_DIM", 512))
    sentence_transformer_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    generator: str = field(default_factory=lambda: os.environ.get("RAG_GENERATOR", "extractive"))
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("RAG_ANTHROPIC_MODEL", "claude-sonnet-4-5")
    )
    max_answer_tokens: int = field(default_factory=lambda: _env_int("RAG_MAX_ANSWER_TOKENS", 700))

    # --- serving ---
    corpus_dir: str = field(default_factory=lambda: os.environ.get("RAG_CORPUS_DIR", "data/corpus"))
    index_dir: str = field(default_factory=lambda: os.environ.get("RAG_INDEX_DIR", "data/index"))
    min_groundedness: float = field(
        default_factory=lambda: _env_float("RAG_MIN_GROUNDEDNESS", 0.25)
    )

    def describe(self) -> dict[str, object]:
        """Serialisable view of the active config, exposed on ``GET /stats``."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


settings = Settings()
