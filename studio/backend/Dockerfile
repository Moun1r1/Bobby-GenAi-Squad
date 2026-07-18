# Bobby Studio backend — the whole engine stays in Python behind a FastAPI + SSE API.
# Build context is the repo root (see docker-compose: `context: ..`). The repo is copied to /app, so
# PYTHONPATH=/app makes `import bobby_squad` resolve and the seed vaults land at /app/knowledge/vaults.
FROM python:3.12-slim

WORKDIR /app/studio/backend
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

# docker CLI (talk to the DGX's docker via the mounted socket in DGX_LOCAL mode → drive the ga_worker GPU container
# with no ssh) + openssh-client (fallback for a REMOTE DGX). See dgx_monitor / DgxTools and the DGX_* env.
RUN apt-get update && apt-get install -y --no-install-recommends docker-cli openssh-client \
    && rm -rf /var/lib/apt/lists/*

# deps first for layer caching
COPY studio/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# the whole repo (context = repo root) → /app  ⇒  /app/bobby_squad, /app/studio, /app/knowledge/vaults
COPY . /app

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
