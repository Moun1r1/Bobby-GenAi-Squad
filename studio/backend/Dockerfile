# Bobby Studio backend — the whole engine stays in Python behind a FastAPI + SSE API.
# Build context is the generative_agentic package dir (see docker-compose): the package is copied to
# /app/generative_agentic and PYTHONPATH=/app makes `import generative_agentic` resolve.
FROM python:3.12-slim

WORKDIR /app/generative_agentic/studio/backend
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

# deps first for layer caching
COPY studio/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# the engine package (context = the generative_agentic dir) → /app/generative_agentic
COPY . /app/generative_agentic

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
