---
title: Bobby GenAi Squad
---

# Bobby GenAi Squad

**A self-organizing generative-agent squad.** Persistent-self agents that coordinate on a recursive shared board,
verify by outcome (not by prose), and prove their gains — because **organization beats raw intelligence**.

A solo one-pass LLM call forgets by design, even at frontier scale. Wins come from *organization*. Measured on the
same model, varying only how the agents were organized:

| organization | function-coverage |
|---|---|
| raw 1-pass | **21%** |
| uncoordinated squad + shared memory | 56% |
| flat coordination | 76% |
| **recursive coordination** | **96%** |

Pure Python stdlib. Talks to any OpenAI-compatible `/v1/chat/completions` endpoint (local or hosted).
**[Code on GitHub »](https://github.com/Moun1r1/Bobby-GenAi-Squad)**

```bash
pip install -e .
export BOBBY_LLM_URL="http://localhost:8000/v1/chat/completions"
export BOBBY_LLM_MODEL="your-served-model-id"
export BOBBY_EMBED_URL="http://localhost:11434/api/embed"   # optional
```

---

## Features

| primitive | what it does |
|---|---|
| **Agent + SelfCore** | persistent-self: identity/goal/progress in a **pinned tier** compaction never touches → flat prompt across a long horizon |
| **squad_solve** | recursive coverage — a squad drains a shared board; `verify` (run-don't-ask) decides done-vs-split; plateau = board drains |
| **prove** | enforced test **validity** — headroom + negative-control + CI; verdicts WIRE / MARGINAL / DELETE / INCONCLUSIVE / INVALID |
| **IdeaLedger** | identity floor (no regeneration) + **emergent agent-assigned states** + **active-repulsion** frontier |
| **BoardTools** | the squad organizes its own board (`board` / `set_state` / `merge`) as self-selected moves |
| **BehaviorTrace + MetaTools** | metacognition — an agent detects a peer's bias & frontier from its real trace |
| **WorldSense** | sense many worlds (peers, files, frontier, affect, self-model) as data, never a directive |
| **SemanticMemory** | self-governs retention by learned usage; deterministic recall floor |
| **SandboxTools** | full sandbox dev loop (write / edit / run / test) — verdicts from execution |

---

## Gains (proven, with honest kills)

Every gain went through `prove` (headroom + negative control + CI). Most proposals **fail** a fair A/B — that's the point.

| mechanism | verdict | result |
|---|---|---|
| Organization (recursive vs solo) | **WIRE** | 21% → 96% coverage |
| Memory-Gate (importance-gated consolidation) | **WIRE** | +191% |
| Active repulsion (farthest-point frontier) | **WIRE** | +74% concept coverage |
| Self-evolving memory policy | **WIRE** | +25% retention / +12.5% generation (neg-control passed) |
| Idea-space novelty gate | **WIRE** | +22.5% diversity vs lexical gate |
| Active-Design (info-gain probing) | **WIRE** | −81% probes |
| Long-horizon (25 OSS codebases) | **WIRE** | prompt bounded ≤ 4689 tok vs ~40k naive (8.5×) |
| CWBU · Synapse · Dialectic · Drift-audit · Tool-gating | **DELETE** | honest kills — no gain for a capable model |

---

## Proofs — run them yourself

**Deterministic** proofs recompute their verdict with just Python. **Endpoint** proofs run the real agents; the
samples are captured from actual runs (server token counts, section counters, specifics quoted from real sources —
the unfakeable parts).

| proof | proves | kind |
|---|---|---|
| [proposals_gain.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/proposals_gain.py) | Memory-Gate WIRE +191%, Active-Design WIRE, CWBU DELETE | deterministic |
| [memory_policy_gain.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/memory_policy_gain.py) | self-evolving memory WIRE; non-predictive control DELETE | deterministic |
| [organization_recursive.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/organization_recursive.py) | organization beats intelligence | endpoint |
| [cross_domain.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/cross_domain.py) | one engine, any behavior the request asks | endpoint |
| [self_review.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/self_review.py) | metacognition: detect a peer's bias & frontier | endpoint |
| [self_development.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/self_development.py) | full dev loop: discover → build+verify → prove | endpoint |
| [squad_reads_code.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/squad_reads_code.py) | long-horizon: read whole codebases end to end, bounded prompt | endpoint |
| [squad_reads_pdfs.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/squad_reads_pdfs.py) | arXiv knowledge farm | endpoint + `[papers]` |
| [transfer_knowledge.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/transfer_knowledge.py) | transferable knowledge across agents & domains | endpoint + embedder |

### Real captured results

**Long-horizon over 25 large OSS codebases** (hermes, redis, django, langchain, llama.cpp, tokio, polars…):
```
pinned prompt tokens: min=1681 max=4689 (bounded) | naive-counterfactual end=40075 → 8.5x
```

**Read code end to end:**
```
Ada   vue-core   sec 13/13  ✅ END-TO-END
Boole django     sec 15/15  ✅ END-TO-END
Ada → "Flask architecture specialist … lifecycle of the app object, remove_ctx/add_ctx decorators"
```

**arXiv knowledge farm** — 5 agents read real papers, evolved into specialists citing the papers' actual cores:
```
Boole → asymptotic holography … AdS4 Coulomb seed via Liénard–Wiechert fields … antipodal matching
Dirac → higher category theory … Morita invariance of full centers … Drinfeld centers within Gray monoids
```

**Transferable knowledge** — Cantor (read only a *logic* paper) correctly explained a *physics* paper it never read,
from the shared store (grounded=True); and an agent bridged *"antipodal matching"* from hep-th into number theory.

---

## Design rules

1. **Organization beats raw intelligence.** 2. **No static prompts / hardcoded roles** — self + tools + move-space.
3. **Verify by outcome.** 4. **Prove, don't claim.** 5. **Guard-first** — guardable mistakes live in code.

_MIT licensed._
