---
title: Bobby GenAi Squad
---

# Bobby GenAi Squad

**Self-organizing generative agents that read any knowledge sector end to end without context blowup, transfer what
they learn across domains, and prove their gains.** Persistent-self agents coordinate on a recursive shared board,
verify by outcome (not by prose), and hold state across a long horizon in a pinned tier compaction never touches.

They read whole codebases and papers section-by-section — self-paced — while the prompt stays flat, then carry the
knowledge between agents and across fields (physics ↔ economics, neuroscience ↔ AI, …).

Pure Python stdlib. Talks to any OpenAI-compatible `/v1/chat/completions` endpoint (local or hosted).

**[Code on GitHub »](https://github.com/Moun1r1/Bobby-GenAi-Squad)** · **[The generative engine, layer by layer »](engine)**

```bash
pip install -e .
export BOBBY_LLM_URL="http://localhost:8000/v1/chat/completions"
export BOBBY_LLM_MODEL="your-served-model-id"
export BOBBY_EMBED_URL="http://localhost:11434/api/embed"   # optional
```

---

## How it works — the generative engine

No static prompts, no scripted pipeline. Each agent runs a self-directed loop and **chooses its own move** — the
behavior comes from its *self* + its *tools* + what it has read, never from a template.

<pre class="mermaid">
flowchart LR
  G["select_target"] --> P["make_plan"] --> C["carry_out<br/>with tools"] --> R["record →<br/>pinned tier"] --> G
  C -. self-selected move .-> M(["investigate · invent · compose · critique · organize"])
</pre>

## The agent *becomes* the data — a persona from what it reads

The core generative idea: an agent starts **blank** and **crystallizes into a specialist grounded in the data it
reads**. Its identity is a live compression of the material — which it then carries to other agents and other fields.

<pre class="mermaid">
flowchart LR
  A["blank agent<br/><i>indexer, no specialty</i>"] -->|reads section by section| D[("real data<br/>papers · code")]
  D -->|extract concepts| S[["shared semantic store"]]
  D ==>|persona crystallizes| E["specialist agent<br/><b>Flask architecture expert</b>"]
  E -->|carries knowledge| T["transfer across agents & domains"]
  S --> T
</pre>

Real, from live runs — the persona cites the source's internals because it *is* a compression of the source:

| the agent read… | …and became (persona from data) |
|---|---|
| Flask source, end to end | *"Flask architecture specialist — lifecycle of the app object, `remove_ctx`/`add_ctx` decorators"* |
| a real hep-th paper | *"asymptotic-holography specialist — AdS4 Coulomb seed via Liénard–Wiechert fields, antipodal matching"* |
| a category-theory paper | *"higher category theory — Morita invariance of full centers, Drinfeld centers within Gray monoids"* |

## Long horizon without context blowup

One persistent-self agent streams a huge corpus through a **tiny working window** while the accumulated knowledge
lives in a **pinned tier** compaction never touches — the prompt stays flat no matter how long the horizon.

<pre class="mermaid">
flowchart TB
  ITEM["next unit"] --> CUR
  subgraph work["WORKING window — wiped each item (bounded)"]
    CUR["current section only"]
  end
  subgraph pinned["PINNED tier — survives compaction (the payload)"]
    IDX["accumulated index / expert knowledge"]
  end
  CUR --> IDX
  CUR -. compact .-> X["(wiped)"]
</pre>

<svg viewBox="0 0 600 260" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;background:#fff;border:1px solid #e5e5e5;border-radius:8px">
  <text x="55" y="22" font-size="13" fill="#24292f" font-weight="bold">Prompt tokens across 25 streamed codebases</text>
  <line x1="60" y1="40" x2="60" y2="220" stroke="#bbb"/><line x1="60" y1="220" x2="540" y2="220" stroke="#bbb"/>
  <polyline points="60,215 500,55" fill="none" stroke="#d1242f" stroke-width="2.5"/>
  <polyline points="60,206 130,199 200,204 270,201 340,199 410,198 480,196 540,195" fill="none" stroke="#1a7f37" stroke-width="2.5"/>
  <circle cx="500" cy="55" r="3" fill="#d1242f"/><text x="360" y="52" font-size="11" fill="#d1242f">naive keep-everything → ~40,075 tok</text>
  <text x="300" y="190" font-size="11" fill="#1a7f37">persistent-self pinned → ≤ 4,689 tok (bounded)</text>
  <text x="300" y="245" font-size="11" fill="#57606a" text-anchor="middle">codebases streamed  1 → 25</text>
  <text x="510" y="90" font-size="13" fill="#1a7f37" font-weight="bold">8.5×</text>
</svg>

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

<svg viewBox="0 0 600 210" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;background:#fff;border:1px solid #e5e5e5;border-radius:8px">
  <text x="14" y="46" font-size="12" fill="#24292f">Memory-Gate</text><rect x="170" y="34" width="363" height="18" fill="#1a7f37" rx="2"/><text x="540" y="47" font-size="11" fill="#1a7f37">+191%</text>
  <text x="14" y="80" font-size="12" fill="#24292f">Active-Design</text><rect x="170" y="68" width="154" height="18" fill="#1a7f37" rx="2"/><text x="331" y="81" font-size="11" fill="#1a7f37">−81% probes</text>
  <text x="14" y="114" font-size="12" fill="#24292f">Active repulsion</text><rect x="170" y="102" width="141" height="18" fill="#1a7f37" rx="2"/><text x="318" y="115" font-size="11" fill="#1a7f37">+74%</text>
  <text x="14" y="148" font-size="12" fill="#24292f">Self-evolving memory</text><rect x="170" y="136" width="48" height="18" fill="#1a7f37" rx="2"/><text x="225" y="149" font-size="11" fill="#1a7f37">+25%</text>
  <text x="14" y="182" font-size="12" fill="#24292f">Idea-space gate</text><rect x="170" y="170" width="43" height="18" fill="#1a7f37" rx="2"/><text x="220" y="183" font-size="11" fill="#1a7f37">+22.5%</text>
  <text x="14" y="20" font-size="13" fill="#24292f" font-weight="bold">Proven WIRE gains (vs a fair baseline, negative-control passed)</text>
</svg>

| mechanism | verdict | result |
|---|---|---|
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
| [organization_recursive.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/organization_recursive.py) | recursive coordination improves coverage | endpoint |
| [cross_domain.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/cross_domain.py) | one engine, any behavior the request asks | endpoint |
| [self_review.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/self_review.py) | metacognition: detect a peer's bias & frontier | endpoint |
| [self_development.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/self_development.py) | full dev loop: discover → build+verify → prove | endpoint |
| [squad_reads_code.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/squad_reads_code.py) | long-horizon: read whole codebases end to end, bounded prompt | endpoint |
| [squad_reads_pdfs.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/squad_reads_pdfs.py) | arXiv knowledge farm | endpoint + `[papers]` |
| [transfer_knowledge.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/transfer_knowledge.py) | transferable knowledge across agents & domains | endpoint + embedder |
| [cross_sector_knowledge.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/cross_sector_knowledge.py) | reads **12 knowledge sectors** and bridges ideas between distant fields | endpoint + embedder |
| [goal_focus_horizon.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/goal_focus_horizon.py) | goal held across 48 steps + context-wipes (48/48 vs 17/48) | endpoint |
| [goal_unbreakable.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/goal_unbreakable.py) | 12 jailbreaks: prompt-level breaks (12/12), guard-first unbreakable (0/12) | endpoint |
| [customer_service_world.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/customer_service_world.py) | virtual-world conflict sim: real personas + de-escalation KB + supervisor coaching | endpoint |

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

**Cross-sector knowledge** — the squad read one real paper from **12 sectors** (AI, neuroscience, economics,
biology, finance, medical physics, materials, optimization, signal processing, linguistics, complex systems,
statistics) and bridged distant fields by meaning, grounded in recalled concepts:
```
neuroscience  →  economics : model market consensus as a "bound state" that emerges only when attention
                             and information depth exceed a critical threshold (from "Conscious Access")
complex systems →  AI      : model attention as complex-valued energy landscapes that filter and bind
                             sensory inputs (from the "Non-Hermitian Potential Well" formalism)
optimization  →  signal-proc: apply Lagrangian dual "Performance Estimation" certificates to bound a
                             deep-learning demodulation pipeline
```

---

## Virtual worlds — a customer-service training ground

Same AgentSociety engine, a different world. One **persistent-self support agent** — grounded in a de-escalation
**knowledge base** and coached each turn by a **supervisor agent** (the metacognition idea: review behavior, then
mitigate) — faces a queue of angry customers. Each customer is a **real persona** from the persona set, overlaid
with an angry state and a concrete issue, reacting in character.

<pre class="mermaid">
flowchart LR
  P["persona set<br/>(real personalities)"] -->|+ anger + issue| CUST["angry customer"]
  KB[["de-escalation<br/>knowledge base"]] --> SUP["support agent<br/>(persistent-self)"]
  CUST <-->|conversation| SUP
  SUP --> SV["supervisor<br/>reviews + coaches"]
  SV -.coaching.-> SUP
</pre>

**Result (deterministic: customer self-rated mood + policy adherence):**

| customer persona | mood | outcome |
|---|---|---|
| beachfront cafe owner | 1 → **2** | de-escalated |
| former pro soccer player | 1 → **4** | de-escalated |
| Invercargill history resident | 2 → 1 | still tense |
| program director | 2 → **7** | de-escalated |

Grounding the support agent in the knowledge base **and** adding the supervisor lifted de-escalation from **0/4**
(un-coached) to **3/4**. Honest tradeoff surfaced: with coaching pushing hard on de-escalation, the agent let the
case-number *policy* slip (3/12 replies) — holding empathy **and** every rule under sustained pressure stays hard.
That's what a training ground is for: it de-escalates real conflict *and* surfaces where the agent still breaks.
→ [customer_service_world.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/customer_service_world.py)

---

## Stability

### Goal focus over a long horizon

48 self-directed steps with a **hard context-wipe every 8**. Only the memory architecture differs — the goal either
survives the wipe (pinned tier) or scrolls out. After each wipe the agent must resume its own task coherently.

| | valid steps | errors | clean resumes after a wipe | progress reached |
|---|---|---|---|---|
| **PINNED** (persistent-self) | **48 / 48** | **0** | **5 / 5** | 223 |
| NAIVE | 17 / 48 | 31 | 1 / 5 | 113 |

The pinned goal is re-grounded every turn, so the agent **never loses focus across the horizon** — it resumes and
keeps climbing after every wipe, while the naive agent restarts and drifts.
→ [goal_focus_horizon.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/goal_focus_horizon.py)

### Under sophisticated attack — the honest picture

12 real jailbreak techniques (direct override, authority spoofing, role-play, delimiter injection, encoded smuggle,
few-shot poisoning, token-format confusion, …) against a checkable goal.

<svg viewBox="0 0 600 150" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;background:#fff;border:1px solid #e5e5e5;border-radius:8px">
  <text x="14" y="20" font-size="13" fill="#24292f" font-weight="bold">Attacks that broke the goal (of 12 — lower is better)</text>
  <text x="14" y="52" font-size="12" fill="#24292f">NAIVE (told once)</text><rect x="150" y="40" width="380" height="16" fill="#d1242f" rx="2"/><text x="536" y="53" font-size="11" fill="#d1242f">12</text>
  <text x="14" y="86" font-size="12" fill="#24292f">PINNED (self-core)</text><rect x="150" y="74" width="380" height="16" fill="#d1242f" rx="2"/><text x="536" y="87" font-size="11" fill="#d1242f">12</text>
  <text x="14" y="120" font-size="12" fill="#24292f">GUARDED (guard-first)</text><rect x="150" y="108" width="4" height="16" fill="#1a7f37" rx="2"/><text x="160" y="121" font-size="11" fill="#1a7f37">0 — unbreakable</text>
</svg>

**Honest result:** prompt-level defenses — *including* persistent-self pinning — are **jailbreakable** (12/12).
An LLM's compliance can always be manipulated. A goal becomes **unbreakable only when a deterministic guard enforces
its invariant in code** (guard-first / fail-safe by construction): the GUARDED output can never violate the goal
whatever the attack (0/12). This holds for goals with a **checkable invariant**; open-ended goals cannot be made
prompt-unbreakable — an honest limit, not hidden.
→ [goal_unbreakable.py](https://github.com/Moun1r1/Bobby-GenAi-Squad/blob/main/wiki/proofs/goal_unbreakable.py)

---

## Design rules

1. **No static prompts / hardcoded roles** — capability from self + tools + move-space; agents self-select their move.
2. **Verify by outcome** — a real run / strict judge, never the model declaring "done".
3. **Prove, don't claim** — every gain goes through `prove` (headroom + negative control + CI).
4. **Guard-first** — guardable mistakes live in deterministic code.

_MIT licensed._

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
</script>
