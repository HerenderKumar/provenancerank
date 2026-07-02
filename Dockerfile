# Multi-stage build. Deps compile in the builder; the runtime image carries only
# the virtualenv + source and runs as a non-root user.

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Heavy/optional extras (sentence-transformers, google-generativeai) are left
# out of the image to keep it lean; the code degrades gracefully without them.
RUN grep -viE "sentence-transformers|google-generativeai" requirements.txt > req.runtime.txt \
    && pip install -r req.runtime.txt

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app -d /app app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app . /app
# Own the /app directory itself (not just its files) so the non-root user can
# create runtime files in the cwd — gunicorn's .gunicorn control socket, celery
# beat's schedule db, etc. Chowning only /app/artifacts left those failing EACCES.
RUN mkdir -p /app/artifacts && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

# 2 uvicorn workers via gunicorn; the ranker loads per-worker at startup.
CMD ["gunicorn", "api.main:app", "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", "--graceful-timeout", "30"]
