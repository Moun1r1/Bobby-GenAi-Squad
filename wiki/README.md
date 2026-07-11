# Bobby GenAi Squad — Wiki & Proofs

Every claim in the [README](../README.md) has a **runnable proof** here. Two kinds:

- **Deterministic** — run with just Python, no model, no network. The verdict is computed live.
- **Endpoint-gated** — point at an OpenAI-compatible LLM (`BOBBY_LLM_URL`) and, for some, an embedder
  (`BOBBY_EMBED_URL`). These run the real agents; the output samples in [`samples/`](samples/) are captured from
  actual runs (real server token counts, real transcripts — not hand-written).

Run a proof from the repo root, e.g. `python wiki/proofs/organization_recursive.py`.

## Proofs

| pipeline | proves | kind | how to run |
|---|---|---|---|
| `proofs/proposals_gain.py` | fair A/Bs: Memory-Gate **WIRE +191%**, Active-Design **WIRE**, CWBU **DELETE** | deterministic | `python wiki/proofs/proposals_gain.py` |
| `proofs/memory_policy_gain.py` | self-evolving memory retention **WIRE**, non-predictive negative control **DELETE** | deterministic | `python wiki/proofs/memory_policy_gain.py` |
| `proofs/organization_recursive.py` | organization beats intelligence: recursive coordination coverage | endpoint | `python wiki/proofs/organization_recursive.py` |
| `proofs/cross_domain.py` | one engine, any behavior the request asks — strict-judge graded | endpoint | `python wiki/proofs/cross_domain.py` |
| `proofs/self_review.py` | metacognition: an agent detects a peer's bias & frontier from its real trace | endpoint | `python wiki/proofs/self_review.py` |
| `proofs/self_development.py` | full dev loop: discover → build+verify → prove | endpoint | `python wiki/proofs/self_development.py` |
| `proofs/self_improve_connectivity.py` | the squad invents + builds + proves its own inter-agent connectivity | endpoint | `python wiki/proofs/self_improve_connectivity.py` |
| `proofs/squad_reads_code.py` | **long-horizon:** a self-organizing squad reads whole codebases **end to end**, section-by-section, self-paced, with a **bounded prompt** (no context blowup) | endpoint + a corpus dir | `HORIZON_APPS=/path/to/repos python wiki/proofs/squad_reads_code.py` |
| `proofs/squad_reads_pdfs.py` | **arXiv knowledge farm:** the squad downloads real arXiv papers, reads them section-by-section, and **builds transferable expert knowledge** — same long-horizon mechanism, on papers instead of code | endpoint + `pip install bobby-genai-squad[papers]` | `python wiki/proofs/squad_reads_pdfs.py` |
| `proofs/transfer_knowledge.py` | **transferable knowledge:** a concept one agent read from a paper is recalled by *another* agent who never read it, and **carried across domains** (physics → number theory) via the shared semantic store | endpoint + embedder | `python wiki/proofs/transfer_knowledge.py` |

## Samples (captured from real runs)

- [`samples/squad_reads_code_output.txt`](samples/squad_reads_code_output.txt) — two agents read `vue-core` and
  `django` **END-TO-END** (13/13, 15/15 sections) and evolved into codebase experts, while the prompt held only the
  current sections. Impossible to fake: the specialist identities cite exact internals of the real source.
- [`samples/arxiv_squad_output.txt`](samples/arxiv_squad_output.txt) — 5 agents each read a **real arXiv paper**
  (IDs verifiable on arxiv.org) section-by-section and evolved into specialists whose expertise cites the papers'
  actual technical cores (Liénard–Wiechert fields / AdS4 Coulomb seed, Morita invariance of Drinfeld centers, the
  quopit Pauli group). A model can't produce those specifics without reading the paper.
- [`samples/transfer_knowledge_output.txt`](samples/transfer_knowledge_output.txt) — Cantor (read only a logic
  paper) correctly explains a *physics* paper it never read, from the shared store (grounded=True), and an agent
  bridges "antipodal matching" from hep-th into number theory. Knowledge transferred across agents and domains.
- [`samples/KNOWLEDGE_MAP_25_oss_repos.md`](samples/KNOWLEDGE_MAP_25_oss_repos.md) — one persistent-self agent
  streamed **25 large OSS codebases** (hermes, redis, django, langchain, llama.cpp, tokio, polars, …) into a pinned
  index. Real, specific findings per repo.
- [`samples/long_horizon_flatness.txt`](samples/long_horizon_flatness.txt) — the mechanical proof for that run:
  pinned prompt stayed **≤ 4689 tokens** across all 25 while the naive counterfactual reached **40075** (~8.5×).
  These are real served `prompt_tokens` — the unfakeable part.

## How to trust these

The deterministic proofs recompute their verdicts every run. The endpoint-gated samples were produced by the
agents on a served model; the numbers that can't be faked (server token counts, section-reached counters,
grounded specifics quoted from real source) are the evidence. Re-run any of them and you get the same shape.
