# syntax=docker/dockerfile:1
# Build from the repository root. See DOCKER.md for CPU/GPU and Compose commands.
FROM ghcr.io/astral-sh/uv:0.11.24 AS uv
FROM node:22-bookworm-slim AS node

FROM python:3.10-slim-bookworm AS full-stack
ARG TORCH_BACKEND=cpu
ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache \
    MPLBACKEND=Agg NO_MKDOCS_2_WARNING=1 \
    STORMSENSE_ROOT=/app STORMSENSE_DEVICE=cpu \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git libgomp1 libstdc++6 rclone tini util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "$LOCAL_GID" geo2wf \
    && useradd --uid "$LOCAL_UID" --gid "$LOCAL_GID" --create-home geo2wf
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && mkdir /app && chown geo2wf:geo2wf /app
WORKDIR /app
USER geo2wf
COPY --chown=geo2wf:geo2wf pyproject.toml uv.lock ./
# CUDA libraries come from locked PyTorch wheels. GPU hosts additionally need
# a working NVIDIA driver and NVIDIA Container Toolkit.
RUN --mount=type=cache,target=/tmp/uv-cache,uid=${LOCAL_UID},gid=${LOCAL_GID} \
    case "$TORCH_BACKEND" in cpu|cuda) ;; *) echo "Use TORCH_BACKEND=cpu or cuda" >&2; exit 2;; esac \
    && uv sync --frozen --extra "$TORCH_BACKEND" --group operational \
       --group artifacts --group dev --group docs --no-install-project
COPY --chown=geo2wf:geo2wf apps/stormsense/package.json apps/stormsense/package-lock.json ./apps/stormsense/
RUN --mount=type=cache,target=/home/geo2wf/.npm,uid=${LOCAL_UID},gid=${LOCAL_GID} npm ci --prefix apps/stormsense
COPY --chown=geo2wf:geo2wf . .
RUN --mount=type=cache,target=/tmp/uv-cache,uid=${LOCAL_UID},gid=${LOCAL_GID} \
    uv sync --frozen --extra "$TORCH_BACKEND" --group operational \
       --group artifacts --group dev --group docs \
    && mkdir -p downloads data logs inference var/stormsense artifacts \
       apps/stormsense/public/data /home/geo2wf/.cache \
    && chmod +x apps/stormsense/runner/entrypoint.sh apps/stormsense/runner/cycle.sh
ENV PATH="/app/.venv/bin:/app/apps/stormsense/node_modules/.bin:$PATH" \
    UV_CACHE_DIR=/home/geo2wf/.cache/uv
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/app/apps/stormsense/runner/entrypoint.sh"]
CMD ["help"]

FROM full-stack AS browser-tests
USER root
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
RUN cd apps/stormsense && npx playwright install --with-deps chromium
USER geo2wf

# Independent web server for a host-local export. Cloudflare deployment uses
# the full-stack target and the existing Worker/R2 configuration instead.
FROM node AS web-build
WORKDIR /app/apps/stormsense
COPY apps/stormsense/package.json apps/stormsense/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY apps/stormsense/ ./
RUN STORMSENSE_PRODUCTION=1 npm run build

FROM nginxinc/nginx-unprivileged:1.28-alpine AS web
COPY apps/stormsense/runner/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-build /app/apps/stormsense/dist /usr/share/nginx/html
EXPOSE 8080

# Preserve full-stack as the default when building without --target.
FROM full-stack AS default
