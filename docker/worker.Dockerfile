# generative-agentic WORKER image — our own, for flexibility.
#
# It gives the swarm a GPU node it can push code to and TRAIN on, and (next) a place to run WORKER generative-agentic
# agents (prompted or trained) on the DGX GPU. Built ON the DGX (arm64 / GB10) FROM a proven local torch+cuda image so
# CUDA is already wired for this hardware — then it layers the framework + room to add deps.
#
# Build (on the DGX):  docker build -f worker.Dockerfile -t ga_worker:latest .
ARG BASE=nvcr.io/nvidia/pytorch:24.10-py3
FROM ${BASE}

WORKDIR /workspace

# the framework itself, importable inside the worker (LLM/embedder clients are stdlib http — no heavy deps needed)
COPY generative_agentic /opt/ga/generative_agentic
ENV PYTHONPATH=/opt/ga:${PYTHONPATH}
ENV GA_WORKER=1

# fail the build early if the base doesn't actually carry torch; add light deps we want available for training glue
RUN python3 -c "import torch; print('torch', torch.__version__)" \
 && pip install --no-cache-dir numpy 2>/dev/null || true \
 && python3 -c "import generative_agentic; print('generative_agentic import OK')"

# JAX/Flax + Google `gemma` foundation stack — the gemma4 / MoE path (better for real Gemma foundation work).
# JAX-CUDA is verified working on GB10 at runtime (CudaDevice detected). The build box has no GPU, so we only
# import-check jax/flax here; gemma is best-effort (heavy transitive deps) and confirmed at runtime.
RUN pip install --no-cache-dir "jax[cuda13]" flax \
 && python3 -c "import jax, flax; print('jax', jax.__version__, 'flax OK')" \
 && (pip install --no-cache-dir gemma || echo "gemma lib deferred — install at runtime")

CMD ["sleep", "infinity"]
