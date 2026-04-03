# --- Stage 1: Builder ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-install-project --no-dev

# --- Stage 2: Runtime ---
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Ensure the app uses the virtualenv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the source code
COPY src/ml_project_template/api ./src/ml_project_template/api
COPY src/ml_project_template/inference ./src/ml_project_template/inference
COPY src/ml_project_template/core ./src/ml_project_template/core


ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app/src"

# Expose the API port
EXPOSE 8000

CMD ["gunicorn", "ml_project_template.api.app:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--preload", \
     "--timeout", "120"]