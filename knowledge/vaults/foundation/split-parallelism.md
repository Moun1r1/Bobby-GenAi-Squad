---
title: split-parallelism
tags: parallelism, sharding, jax, fsdp, gb10
source: seed:memory
links: [[gemma-foundation-native]], [[perf-memory]]
---

# Split & parallelism — how a model is divided across compute

## The four splits
- **DP (data parallel)** — replicate the model, split the BATCH. Needs the whole model on one device.
- **FSDP / ZeRO** — shard params + optimizer + grads across devices, gather per-layer. Fits models bigger than one
  device with near-DP simplicity. Default when a model doesn't fit but you have several devices.
- **TP (tensor parallel)** — split weight matrices (heads / MLP columns); every token touches every device. High
  comms — keep within one node / fast interconnect.
- **PP (pipeline parallel)** — split by LAYER into stages; micro-batch to hide the bubble.

Compose them (3D parallelism): FSDP across nodes × TP within a node × DP over the rest.

## The JAX-native way (use this for [[gemma-foundation-native]])
- `jax.sharding.Mesh(devices, axis_names=("data","model"))` — name the device grid.
- `NamedSharding(mesh, PartitionSpec("data","model"))` — declare each array's split (params → model axis, batch →
  data axis).
- `jax.jit(in_shardings=…, out_shardings=…)` or `shard_map` for explicit per-shard programs (`pjit` = older
  spelling). You declare layout; XLA inserts the collectives. MoE routing shards experts across the model axis.


## Rule
Pick the smallest split that FITS with headroom: single-device + LoRA first; FSDP when params don't fit; TP/PP only
across a real fast interconnect. On GB10, "fit the unified pool + place MoE experts" is the whole game.

## code — declare a mesh + shard params (JAX native)
```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
devices = jax.devices()                                       # GB10: one CudaDevice
mesh = Mesh(mesh_utils.create_device_mesh((1, len(devices)), devices), ("data", "model"))
# params sharded on the MODEL axis, batches on the DATA axis
p_shard = NamedSharding(mesh, P(None, "model"))
b_shard = NamedSharding(mesh, P("data", None))
params = jax.device_put(params, p_shard)
train_step = jax.jit(step_fn, in_shardings=(p_shard, b_shard), out_shardings=p_shard)
```

## read further
- JAX distributed arrays + auto-parallelism: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- `shard_map` (explicit per-shard programs): https://jax.readthedocs.io/en/latest/notebooks/shard_map.html
- FSDP/ZeRO concepts: https://engineering.fb.com/2021/07/15/open-source/fsdp/  ·  ZeRO paper arXiv:1910.02054
