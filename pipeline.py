"""The pipeline: ingest → retrieve → generate → verify.

The verification step is the part that distinguishes this from a demo. An answer
is not returned to the caller until its citations have been resolved against the
passages that were actually retrieved, and its sentences scored for lexical
support. Hallucinated markers are stripped; poorly supported answers are flagged
with ``grounded=False`` so a caller can degrade gracefully instead of shipping a
confident-sounding fabrication to a clinician or a customer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .bm25 import BM25Index
from .chunking import Chunk, chunk_document
from .config import Settings
from .config import settings as default_settings
from .embeddings import Embedder, get_embedder
from .generation import CITATION_RE, Generator, get_generator
from .retriever import HybridRetriever, RetrievedChunk
from .store import VectorStore
from .text import split_sentences, token_overlap

logger = logging.getLogger("rag.pipeline")

SNIPPET_CHARS = 240


@dataclass(slots=True)
class Citation:
    marker: int
    chunk_id: str
    doc_id: str
    source: str
    snippet: str

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "snippet": self.snippet,
        }


@dataclass(slots=True)
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    retrieval: list[RetrievedChunk]
    groundedness: float
    grounded: bool
    abstained: bool
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieval": [r.to_dict() for r in self.retrieval],
            "groundedness": round(self.groundedness, 4),
            "grounded": self.grounded,
            "abstained": self.abstained,
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
        }


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


def parse_citations(answer: str, contexts: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Resolve ``[n]`` markers against ``contexts``, dropping any that do not exist.

    A model that invents ``[7]`` when five passages were supplied has hallucinated
    a source, and silently rendering that marker is how fake citations reach users.
    """
    valid: dict[int, Citation] = {}
    invalid: list[int] = []
    for match in CITATION_RE.finditer(answer):
        marker = int(match.group(1))
        if 1 <= marker <= len(contexts):
            chunk = contexts[marker - 1]
            valid.setdefault(
                marker,
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source=str(chunk.metadata.get("source", chunk.doc_id)),
                    snippet=chunk.text[:SNIPPET_CHARS].strip(),
                ),
            )
        else:
            invalid.append(marker)

    cleaned = answer
    if invalid:
        logger.warning("dropping %d out-of-range citation marker(s): %s", len(invalid), invalid)
        cleaned = CITATION_RE.sub(
            lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(contexts) else "",
            answer,
        )
        cleaned = " ".join(cleaned.split())

    return cleaned, [valid[m] for m in sorted(valid)]


def score_groundedness(answer: str, contexts: list[RetrievedChunk]) -> float:
    """Mean lexical support of each answer sentence against its cited passages.

    A cheap, deterministic proxy for faithfulness. It cannot catch a fluent
    paraphrase that inverts meaning — an LLM judge is the right tool for that —
    but it runs in microseconds and catches the common failure of an answer
    drifting away from its evidence.
    """
    sentences = split_sentences(answer)
    if not sentences or not contexts:
        return 0.0

    scores: list[float] = []
    for sentence in sentences:
        markers = [int(m) for m in CITATION_RE.findall(sentence)]
        cited = [contexts[m - 1].text for m in markers if 1 <= m <= len(contexts)]
        # An uncited sentence is checked against everything retrieved, so that a
        # missing marker costs recall rather than being scored as free support.
        pool = cited or [c.text for c in contexts]
        body = CITATION_RE.sub("", sentence)
        scores.append(max(token_overlap(body, passage) for passage in pool))
    return sum(scores) / len(scores)


class RAGPipeline:
    def __init__(
        self,
        *,
        config: Settings | None = None,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.config = config or default_settings
        self.embedder = embedder or get_embedder(
            self.config.embedder,
            dim=self.config.embedding_dim,
            model_name=self.config.sentence_transformer_model,
        )
        self.generator = generator or get_generator(
            self.config.generator,
            model=self.config.anthropic_model,
            max_tokens=self.config.max_answer_tokens,
        )
        self.store = VectorStore(dim=self.embedder.dim)
        self.bm25 = BM25Index()
        self.retriever = self._build_retriever()
        self._doc_ids: set[str] = set()

    def _build_retriever(self) -> HybridRetriever:
        return HybridRetriever(
            self.store,
            self.bm25,
            self.embedder,
            candidate_k=self.config.candidate_k,
            rrf_k=self.config.rrf_k,
            mmr_lambda=self.config.mmr_lambda,
        )

    # --- ingestion -------------------------------------------------------
    def ingest(self, documents: list[Document]) -> int:
        """Chunk, embed and index ``documents``. Returns the chunk count added."""
        all_chunks: list[Chunk] = []
        for document in documents:
            metadata = {"source": document.id, **document.metadata}
            all_chunks.extend(
                chunk_document(
                    document.id,
                    document.text,
                    chunk_tokens=self.config.chunk_tokens,
                    overlap_tokens=self.config.chunk_overlap_tokens,
                    metadata=metadata,
                )
            )
            self._doc_ids.add(document.id)

        if not all_chunks:
            return 0

        vectors = self.embedder.encode([c.text for c in all_chunks])
        self.store.add(all_chunks, vectors)
        self.bm25.add_many([(c.id, c.text) for c in all_chunks])
        self.retriever = self._build_retriever()
        logger.info("ingested %d chunks from %d document(s)", len(all_chunks), len(documents))
        return len(all_chunks)

    def ingest_directory(self, directory: str | Path) -> int:
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"corpus directory not found: {path}")
        documents = [
            Document(
                id=file.stem,
                text=file.read_text(encoding="utf-8"),
                metadata={"path": str(file), "source": file.name},
            )
            for file in sorted(path.glob("**/*"))
            if file.suffix.lower() in {".md", ".txt"}
        ]
        if not documents:
            raise ValueError(f"no .md or .txt files found under {path}")
        return self.ingest(documents)

    # --- query -----------------------------------------------------------
    def query(self, question: str, top_k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        top_k = top_k or self.config.top_k

        t0 = time.perf_counter()
        contexts = self.retriever.retrieve(question, top_k=top_k)
        t1 = time.perf_counter()
        raw = self.generator.generate(question, contexts)
        t2 = time.perf_counter()

        abstained = raw.startswith("INSUFFICIENT_CONTEXT")
        if abstained:
            answer, citations, groundedness = raw, [], 0.0
        else:
            answer, citations = parse_citations(raw, contexts)
            groundedness = score_groundedness(answer, contexts)

        return Answer(
            question=question,
            answer=answer,
            citations=citations,
            retrieval=contexts,
            groundedness=groundedness,
            grounded=(not abstained) and groundedness >= self.config.min_groundedness,
            abstained=abstained,
            latency_ms={
                "retrieval": (t1 - t0) * 1000,
                "generation": (t2 - t1) * 1000,
                "total": (t2 - t0) * 1000,
            },
        )

    # --- stats / persistence --------------------------------------------
    def stats(self) -> dict[str, object]:
        return {
            "documents": len(self._doc_ids),
            "chunks": len(self.store),
            "lexical_terms": len(self.bm25.to_dict()["postings"]),
            "embedder": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "generator": self.generator.name,
            "config": self.config.describe(),
        }

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.store.save(path)
        (path / "bm25.json").write_text(json.dumps(self.bm25.to_dict()), encoding="utf-8")
        (path / "docs.json").write_text(json.dumps(sorted(self._doc_ids)), encoding="utf-8")

    def load(self, directory: str | Path) -> None:
        path = Path(directory)
        self.store = VectorStore.load(path)
        if self.store.dim != self.embedder.dim:
            raise ValueError(
                f"index dim {self.store.dim} != embedder dim {self.embedder.dim}; "
                "rebuild the index after changing embedder"
            )
        self.bm25 = BM25Index.from_dict(json.loads((path / "bm25.json").read_text()))
        self._doc_ids = set(json.loads((path / "docs.json").read_text()))
        self.retriever = self._build_retriever()
