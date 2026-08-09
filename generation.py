"""Answer generation behind a Protocol, with a mandatory citation contract.

Two providers ship:

* ``ExtractiveGenerator`` — no LLM, no network, no API key. Selects the highest
  scoring sentences from retrieved context and attaches citation markers. Every
  test and CI run uses this, which is what makes the eval harness deterministic.
* ``AnthropicGenerator`` — the real thing, with a system prompt that forces
  ``[n]`` markers and an explicit refusal path when context is insufficient.

Both are validated by the same downstream citation parser, so switching provider
cannot silently change the answer contract.
"""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable

from .retriever import RetrievedChunk
from .text import split_sentences, tokenize

SYSTEM_PROMPT = """\
You answer questions strictly from the numbered context passages provided.

Rules:
1. Every factual sentence must end with one or more citation markers like [1] or [2][4],
   referring to the numbered passage that supports it.
2. Never state anything the passages do not support. Do not use outside knowledge.
3. If the passages do not contain the answer, reply exactly:
   INSUFFICIENT_CONTEXT: <one sentence describing what is missing>
4. Be concise. Prefer three sentences over ten.
"""

CITATION_RE = re.compile(r"\[(\d+)\]")


def build_context_block(contexts: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(contexts, start=1):
        source = chunk.metadata.get("source", chunk.doc_id)
        parts.append(f"[{i}] (source: {source})\n{chunk.text}")
    return "\n\n".join(parts)


@runtime_checkable
class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str: ...


class ExtractiveGenerator:
    """Ranks candidate sentences by how much of the question they cover.

    ``min_coverage`` is the abstention threshold: if no candidate sentence covers
    at least this fraction of the question's content terms, the generator declines
    rather than returning the least-bad sentence it found. Without it, a single
    incidental token match is enough to produce a confident-looking answer to a
    question the corpus cannot address — the failure mode that
    ``abstention_accuracy`` in the eval harness exists to catch. Raising the value
    trades ``abstention_accuracy`` against ``false_abstention_rate``; both are
    reported side by side so the trade can be made on evidence rather than taste.

    Known weakness: coverage weights every query term equally, so matching the
    rare, decisive term "aggregation" counts the same as matching "use". The
    principled fix is to weight coverage by inverse document frequency from the
    lexical index; the default of 0.3 was calibrated on the eval set as an interim
    measure. See "Limitations" in the README.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_coverage: float = 0.3) -> None:
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return "INSUFFICIENT_CONTEXT: no passages were retrieved for this question."

        query_terms = set(tokenize(question))
        if not query_terms:
            return "INSUFFICIENT_CONTEXT: the question contained no searchable terms."

        best_coverage = 0.0
        scored: list[tuple[float, int, int, str]] = []
        for position, chunk in enumerate(contexts, start=1):
            for order, sentence in enumerate(split_sentences(chunk.text)):
                terms = set(tokenize(sentence))
                if not terms:
                    continue
                hits = len(query_terms & terms)
                if hits == 0:
                    continue
                best_coverage = max(best_coverage, hits / len(query_terms))
                # Coverage of the question, lightly penalising very long sentences
                # and rewarding earlier (higher-ranked) passages.
                coverage = hits / len(query_terms)
                brevity = 1.0 / (1.0 + len(terms) / 40.0)
                rank_bonus = 1.0 / position
                scored.append((coverage * brevity * rank_bonus, position, order, sentence))

        if not scored:
            return (
                "INSUFFICIENT_CONTEXT: retrieved passages do not mention the "
                "terms in this question."
            )
        if best_coverage < self.min_coverage:
            return (
                "INSUFFICIENT_CONTEXT: retrieved passages only match this question "
                f"incidentally (best term coverage {best_coverage:.2f} < "
                f"{self.min_coverage:.2f})."
            )

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen = scored[: self.max_sentences]
        # Restore reading order so the answer flows like prose.
        chosen.sort(key=lambda item: (item[1], item[2]))

        seen: set[str] = set()
        sentences: list[str] = []
        for _, position, _, sentence in chosen:
            body = sentence.rstrip(" .")
            if body.lower() in seen:
                continue
            seen.add(body.lower())
            sentences.append(f"{body} [{position}].")
        return " ".join(sentences)


class AnthropicGenerator:
    """Calls the Messages API. Import and client creation are lazy."""

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 700, api_key: str | None = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or run with RAG_GENERATOR=extractive."
            )
        self._client = None

    def _get_client(self):  # pragma: no cover - needs credentials
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic is not installed. Install with `pip install -e '.[llm]'`."
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:  # pragma: no cover
        if not contexts:
            return "INSUFFICIENT_CONTEXT: no passages were retrieved for this question."
        message = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Context passages:\n\n{build_context_block(contexts)}\n\n"
                        f"Question: {question}"
                    ),
                }
            ],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()


def get_generator(kind: str, *, model: str = "", max_tokens: int = 700) -> Generator:
    kind = kind.lower()
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "anthropic":
        return AnthropicGenerator(model=model, max_tokens=max_tokens)
    raise ValueError(f"unknown generator: {kind!r} (expected 'extractive' or 'anthropic')")
