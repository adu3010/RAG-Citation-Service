# Multi-stage build. The wheel-building stage carries compilers and caches that
# have no business being in a production image; only the installed site-packages
# and the application source cross into the final layer.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Requirements are copied on their own so this layer caches independently of the
# source tree — editing a Python file must not trigger a dependency reinstall.
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.12-slim AS runtime

# PYTHONDONTWRITEBYTECODE keeps the read-only filesystem clean; PYTHONUNBUFFERED
# makes container logs appear in real time instead of on flush.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    RAG_CORPUS_DIR=/app/data/corpus

# curl is needed by HEALTHCHECK below; nothing else is installed.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/corpus/ ./data/corpus/
COPY --chown=appuser:appuser evals/ ./evals/

# Run unprivileged. A container that does not need root should never have it.
USER appuser

EXPOSE 8000

# Probes /readyz rather than /healthz: the process can be alive while the index
# is still building, and traffic must not be routed to it until it is not.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/readyz || exit 1

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
