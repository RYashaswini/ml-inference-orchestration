# ---- Build stage ----
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a project-local venv, no dev deps
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the app (code + trained model artifacts)
COPY app/ ./app/

# ---- Runtime stage ----
FROM python:3.12-slim

WORKDIR /app

# Copy venv and app from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app

# Make sure venv binaries are used
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Basic healthcheck hitting your /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]