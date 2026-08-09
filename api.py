"""FastAPI service layer.

Design notes worth defending in a review:

* The index is built once in the lifespan handler, not per request. Rebuilding an
  index inside a handler is the single most common way RAG demos die under load.
* ``/query`` returns citations and a groundedness score alongside the answer, so
  the caller can render evidence and apply its own confidence threshold.
* ``/metrics`` emits Prometheus text format by hand rather than adding a client
  library — the exposition format is stable and this keeps the image small.
* ``/readyz`` is separate from ``/healthz``: the process can be alive while the
  index is still loading, and Kubernetes needs to tell those apart.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .config import settings
from .eval.metrics import percentile
from .pipeline import Document, RAGPipeline

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rag.api")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=500)


class ServiceState:
    """Holds the pipeline plus lightweight in-process telemetry."""

    def __init__(self) -> None:
        self.pipeline: RAGPipeline | None = None
        self.ready = False
        self.started_at = time.time()
        self.request_count = 0
        self.error_count = 0
        self.abstain_count = 0
        self.ungrounded_count = 0
        # Bounded so a long-running process cannot leak memory through telemetry.
        self.latencies: deque[float] = deque(maxlen=1000)


state = ServiceState()


def build_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline()
    try:
        pipeline.ingest_directory(settings.corpus_dir)
    except (FileNotFoundError, ValueError) as exc:
        # An empty corpus is a valid cold start: documents can arrive via /ingest.
        logger.warning("starting with an empty index: %s", exc)
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("building index from %s", settings.corpus_dir)
    state.pipeline = build_pipeline()
    state.ready = True
    logger.info("index ready: %s", state.pipeline.stats()["chunks"])
    yield
    state.ready = False


app = FastAPI(
    title="Citation-grounded RAG service",
    version="1.0.0",
    description=(
        "Hybrid (BM25 + dense) retrieval with reciprocal rank fusion, MMR "
        "diversification, verified citations and a groundedness gate."
    ),
    lifespan=lifespan,
)


def get_pipeline() -> RAGPipeline:
    if state.pipeline is None or not state.ready:
        raise HTTPException(status_code=503, detail="index is not ready")
    return state.pipeline


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, object]:
    return {"status": "ok", "uptime_s": round(time.time() - state.started_at, 1)}


@app.get("/readyz", tags=["ops"])
async def readyz() -> dict[str, object]:
    if not state.ready:
        raise HTTPException(status_code=503, detail="index is not ready")
    return {"status": "ready", "chunks": len(get_pipeline().store)}


@app.get("/stats", tags=["ops"])
async def stats() -> dict[str, object]:
    pipeline = get_pipeline()
    return {
        **pipeline.stats(),
        "requests": state.request_count,
        "errors": state.error_count,
        "abstentions": state.abstain_count,
        "ungrounded": state.ungrounded_count,
        "latency_p50_ms": round(percentile(list(state.latencies), 50), 2),
        "latency_p95_ms": round(percentile(list(state.latencies), 95), 2),
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def metrics() -> str:
    latencies = list(state.latencies)
    lines = [
        "# HELP rag_requests_total Total /query requests served.",
        "# TYPE rag_requests_total counter",
        f"rag_requests_total {state.request_count}",
        "# HELP rag_errors_total Total failed /query requests.",
        "# TYPE rag_errors_total counter",
        f"rag_errors_total {state.error_count}",
        "# HELP rag_abstentions_total Answers where the model declined for lack of context.",
        "# TYPE rag_abstentions_total counter",
        f"rag_abstentions_total {state.abstain_count}",
        "# HELP rag_ungrounded_total Answers below the groundedness threshold.",
        "# TYPE rag_ungrounded_total counter",
        f"rag_ungrounded_total {state.ungrounded_count}",
        "# HELP rag_query_latency_ms Query latency quantiles in milliseconds.",
        "# TYPE rag_query_latency_ms summary",
        f'rag_query_latency_ms{{quantile="0.5"}} {percentile(latencies, 50):.3f}',
        f'rag_query_latency_ms{{quantile="0.95"}} {percentile(latencies, 95):.3f}',
        f'rag_query_latency_ms{{quantile="0.99"}} {percentile(latencies, 99):.3f}',
        "# HELP rag_indexed_chunks Number of chunks currently indexed.",
        "# TYPE rag_indexed_chunks gauge",
        f"rag_indexed_chunks {len(state.pipeline.store) if state.pipeline else 0}",
    ]
    return "\n".join(lines) + "\n"


@app.post("/query", tags=["rag"])
async def query(payload: Annotated[QueryRequest, Body()]) -> dict[str, object]:
    pipeline = get_pipeline()
    state.request_count += 1
    try:
        answer = pipeline.query(payload.question, top_k=payload.top_k)
    except ValueError as exc:
        state.error_count += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert to 5xx, keep the trace in logs
        state.error_count += 1
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail="internal error") from exc

    state.latencies.append(answer.latency_ms["total"])
    if answer.abstained:
        state.abstain_count += 1
    elif not answer.grounded:
        state.ungrounded_count += 1
        logger.warning(
            "low groundedness %.3f for question=%r", answer.groundedness, payload.question[:120]
        )
    return answer.to_dict()


@app.post("/ingest", tags=["rag"], status_code=201)
async def ingest(payload: Annotated[IngestRequest, Body()]) -> dict[str, object]:
    pipeline = get_pipeline()
    added = pipeline.ingest(
        [Document(id=d.id, text=d.text, metadata=d.metadata) for d in payload.documents]
    )
    return {
        "documents": len(payload.documents),
        "chunks_added": added,
        "chunks_total": len(pipeline.store),
    }
