# Build from the repository root: docker build -f apps/stormsense/runner/Dockerfile .
FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends rclone ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --group operational --no-group dev
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# Bind /app/downloads/models and /app/var/stormsense from persistent storage.
# Publishing also needs the existing rclone configuration mounted read-only.
# bootstrap requires an externally installed hf CLI; mount already bootstrapped assets.
ENTRYPOINT ["python", "-m", "geo2wf.operational.cli"]
CMD ["--help"]
