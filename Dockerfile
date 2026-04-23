FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Improve reliability for uv in containerized filesystems.
ENV UV_LINK_MODE=copy

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application code and install the project itself.
COPY . .
RUN uv sync --locked --no-dev

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uv run uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
