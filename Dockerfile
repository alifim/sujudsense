# syntax=docker/dockerfile:1.5

# --- Stage 1: Build with uv ---
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    UV_NO_DEV=1

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

RUN uv run python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# --- Stage 2: Runtime image ---
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    LOG_LEVEL=INFO

COPY --from=builder /app /app
COPY .env .

EXPOSE 8000

CMD ["uv", "run", "--env-file", ".env", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]