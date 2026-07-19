# SOMA flywheel — distill → fine-tune (reproducible pipeline)

The compounding turn of the flywheel: turn Bobby's **verified** solved tasks into
a training corpus, LoRA-fine-tune a base model on it, and gate the result on a
CI-separated gain before you keep it. This is the runnable version of
[`docs/EXTENSIONS.md` §2](../../docs/EXTENSIONS.md).

**Needs a GPU** for the fine-tune (a small base like `qwen3-4b` is enough; the
loop is model-agnostic). Everything is parameterized — no hardcoded paths, models,
or endpoints.

## What this is (SFT) vs what it is not (yet) (DPO)

- **Today — SFT on verified gold (shipped, measured).** The corpus is
  `(prompt → verified output)` pairs where the label is exact-set-equality gold
  (plugin-served or graded-correct). Standard supervised fine-tuning. The +16.5 %
  (CI-separated) result below is *this*.
- **Future — `SelfDPO+` (planned, NOT in this folder).** Preference pairs mined
  from the agent's own critiques, trained with **KL regularization + safety anchors
  + a human-in-loop fallback**. Self-generated preference data frequently
  underperforms human data and can mode-collapse, so it will ship **only** if it
  clears the gain-gate against a human-data / no-DPO control. See [ROADMAP.md](../../ROADMAP.md).

## Dependencies

Core `bobby_squad` is stdlib-only; this pipeline is an **optional** ML add-on:

```bash
pip install -r examples/soma_flywheel/requirements.txt   # torch + transformers + peft + numpy (GPU)
```

`emit_corpus.py` needs only the core package (+ numpy); steps 2–4 need the full stack above.

## Pipeline

```bash
# 1. Emit a verified SFT corpus from Bobby's burn-in task families (labels are
#    exact-set-equality gold, so every training pair is trustworthy).
python emit_corpus.py                      # writes soma_train.jsonl + soma_test.jsonl

# 2. LoRA fine-tune a base model on the corpus (pure transformers + peft, no trl).
python train_lora.py --model /path/to/base-hf-model --train soma_train.jsonl --out ./adapter

# 3. Evaluate base vs base+adapter with a paired bootstrap CI + gain gate.
python eval_stats.py --model /path/to/base-hf-model --adapter ./adapter \
                     --test soma_test.jsonl --gain-min 0.02 --out verdict.json

# 4. If the gate ACCEPTs (CI excludes the null), merge the adapter into standalone
#    weights you can serve.
python merge_adapter.py --model /path/to/base-hf-model --adapter ./adapter --out ./merged
```

`eval_stats.py` prints `ACCEPT` only when the lower bound of the 95 % CI on the
accuracy delta clears `--gain-min` (mirrors `bobby_squad/proving.confirm_gain`).

## Measured (qwen3-4b stand-in)

```
BASE 71.8 %  [67.2 %, 76.0 %]   ->   FINETUNED 88.2 %  [85.0 %, 91.2 %]
Δ +16.5 %   95 % CI [+12.2 %, +21.0 %]   (CI excludes 0 => ACCEPT; McNemar +79 / -13)
per-family: image 0->100 · math 47->71 · algo 64->75 · extract 85->91 · code 95->100
```
