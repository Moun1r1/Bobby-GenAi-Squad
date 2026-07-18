# Primitives & plugins

The engine distills three kinds of artifact, in rising generality. All are executable and gated by a proof; none are
prompts.

1. **Domain plugin** — a bound rule for one task family (e.g. a regex extractor). Proven on that family, served by the
   router. `bobby_squad/burn_in.py`.
2. **LLM-authored algorithm** — the model writes `def solve(text): ...`; the engine sandbox-compiles it (restricted
   builtins), gain-proofs it on held-out inputs, and freezes the code. Verified live: a working Roman-numeral parser
   and a Luhn checksum — logic regex cannot express. `burn_in.py:make_codeplugin`, `_distill(kind="algo")`.
3. **Cross-domain primitive** — a domain-free skeleton bound per domain by a parameter (`extract_matching(text,
   pattern)`, `reduce_integers(text, op)`, `find_analogous_case(query, store)`). Promoted only through a **cross-domain
   gain-proof**: the same code must clear the gate on ≥ N domains it was not co-fit on. `primitive_intel.py`,
   `primitive_lib.py`.

## The library

`PrimitiveLibrary` (`primitive_lib.py`) is a directory-backed, category-organized store with a persistent semantic
memory index:

```
lib/
├── index.json          # {name: {path, category, signature, passed_domains, fingerprint, ...}}
├── embeddings.jsonl     # description → vector (semantic memory)
└── core/<category>/<name>.py
```

Before distilling, the engine **finds a capability back** two ways, so it never re-learns the same thing:

- **Semantic** — embed the task description, cosine-search the index (`"sum the integers"` → `reduce_integers`).
- **Structural** — AST functional fingerprint (alpha-renamed, docstring-stripped): the same loop under different
  variable names is recognized as an existing primitive, not added as a twin.

`recall_or_distill(...)` consults structural then semantic memory first; only genuinely new capability is distilled.
Gate-passed primitives persist and auto-load on the next run, so the bank compounds across runs.

## Absorbing open-source capability

The same surfaces take external capability and make it a cheap local call:

- **Skills / tools** register as plugins behind the router + proof gate.
- **Memory** is the knowledge vault (`vault.py`, linked notes read + written by agents) plus the primitive index and
  correction memory.
- **Compute** is sandbox + GPU-worker tools (`agent_tools.py`) — an agent writes, runs, and trains; verdicts come from
  execution.

Irreducible steps (open judgment — `self_critique`, `break_down_goal`, `merge_conflicting_views`) are flagged and
stay on the LLM; the library holds only what provably generalizes as code.

## Reproduce

```bash
python wiki/proofs/test_primitives.py           # cross-domain generalization + the gate rejecting a non-generalizer
python wiki/proofs/test_primitive_lib.py        # organized library: semantic + structural recall, persistent restart
python wiki/proofs/test_primitive_autoload.py   # gate-passed primitives self-persist and auto-load
python wiki/proofs/test_algo_distill.py         # LLM-authored code frozen as a plugin (deterministic mock)
```
