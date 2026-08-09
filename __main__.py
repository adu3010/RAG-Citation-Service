"""``python -m rag`` builds an index from the corpus and persists it."""

from __future__ import annotations

import argparse

from .config import settings
from .pipeline import RAGPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and persist the RAG index.")
    parser.add_argument("--corpus", default=settings.corpus_dir)
    parser.add_argument("--out", default=settings.index_dir)
    args = parser.parse_args(argv)

    pipeline = RAGPipeline()
    chunks = pipeline.ingest_directory(args.corpus)
    pipeline.save(args.out)
    print(f"indexed {chunks} chunks from {args.corpus} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
