ARG BASE=nvcr.io/nvidia/pytorch:24.10-py3
FROM ${BASE}

WORKDIR /workspace

# the framework + seed vaults, importable inside the worker (LLM/embedder clients are stdlib http — no heavy deps).
# torch lives in the base image's site-packages (already on sys.path); we only add /opt/ga for bobby_squad.
COPY bobby_squad /opt/ga/bobby_squad
COPY knowledge   /opt/ga/knowledge
ENV PYTHONPATH=/opt/ga
ENV GA_WORKER=1

# fail the build early if the base doesn't actually carry torch; add light deps the engine/training glue needs.
RUN python3 -c "import torch; print('torch', torch.__version__)" \
 && pip install --no-cache-dir numpy \
 && python3 -c "import bobby_squad; print('bobby_squad import OK')"

# training stack — the LoRA / DPO libraries the training pipes push to the worker (torch comes from the base above).
# These are pure-Python / manylinux wheels (arch-independent), so this installs the same on any CUDA GPU host.
RUN pip install --no-cache-dir transformers peft trl datasets accelerate \
 && python3 -c "import transformers, peft, trl; print('transformers', transformers.__version__, '· peft', peft.__version__, '· trl OK')"

CMD ["sleep", "infinity"]
