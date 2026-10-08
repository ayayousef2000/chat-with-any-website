# syntax=docker/dockerfile:1

# Both base images are pinned to exact digests, like the actions in .github/workflows. Dependabot proposes updates.

# uv, the tool that installs the locked dependencies (0.12.23)
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv

# --- Build stage: install the locked runtime dependencies and the app into a virtual environment ---------------------
# python 3.14.8
FROM python:3.14.8-slim-trixie@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2 AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so that this layer is reused when only the app code changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then the app itself, installed as a regular package (not editable), so the final image needs no source tree.
COPY README.md LICENSE ./
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- Final stage: only the virtual environment, run as an unprivileged user ----------------------------------------
FROM python:3.14.8-slim-trixie@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2 AS runtime

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

# The usage records of stored pages (STATE_DB_PATH, data/state.db) live here. Mount a volume to keep them.
RUN mkdir /app/data && chown app:app /app/data
VOLUME ["/app/data"]

USER app

EXPOSE 8000

# The page itself is the simplest check that the server answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3)"]

# Settings come from environment variables (see .env.example); no .env file is copied into the image.
# Behind a proxy, set FORWARDED_ALLOW_IPS to the proxy's address so that each visitor's own address is used for the
# limits per visitor; uvicorn reads that variable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
