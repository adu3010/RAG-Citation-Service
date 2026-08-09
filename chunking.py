"""Sentence-aware chunking.

Fixed-width character chunking is the usual first attempt and it reliably cuts
sentences in half, which wrecks citation quality: a chunk that ends mid-clause
cannot be quoted back to a user as evidence. So chunks are packed out of whole
sentences up to a token budget, with an overlap tail to preserve context that
straddles a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .text import split_sentences, tokenize


@dataclass(slots=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    ordinal: int
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "ordinal": self.ordinal,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Chunk:
        return cls(
            id=raw["id"],
            doc_id=raw["doc_id"],
            text=raw["text"],
            ordinal=raw["ordinal"],
            metadata=raw.get("metadata", {}),
        )


def chunk_document(
    doc_id: str,
    text: str,
    *,
    chunk_tokens: int = 180,
    overlap_tokens: int = 40,
    metadata: dict[str, object] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping, sentence-aligned chunks.

    A sentence longer than ``chunk_tokens`` is emitted on its own rather than
    truncated — losing content silently is worse than an oversized chunk.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_tokens)")

    sentences = split_sentences(text)
    if not sentences:
        return []

    metadata = dict(metadata or {})
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc_id}::{ordinal}",
                doc_id=doc_id,
                text=" ".join(buffer),
                ordinal=ordinal,
                metadata=metadata,
            )
        )
        # Carry a tail of trailing sentences forward as overlap.
        tail: list[str] = []
        tail_len = 0
        for sentence in reversed(buffer):
            n = len(tokenize(sentence, drop_stopwords=False))
            if tail_len + n > overlap_tokens:
                break
            tail.insert(0, sentence)
            tail_len += n
        buffer = tail
        buffer_len = tail_len

    for sentence in sentences:
        n = len(tokenize(sentence, drop_stopwords=False))
        if buffer and buffer_len + n > chunk_tokens:
            flush()
        buffer.append(sentence)
        buffer_len += n

    # Final flush without overlap bookkeeping.
    if buffer:
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc_id}::{ordinal}",
                doc_id=doc_id,
                text=" ".join(buffer),
                ordinal=ordinal,
                metadata=metadata,
            )
        )

    # Overlap can make the tail chunk a strict subset of its predecessor; drop it.
    if len(chunks) > 1 and chunks[-1].text in chunks[-2].text:
        chunks.pop()
    return chunks
