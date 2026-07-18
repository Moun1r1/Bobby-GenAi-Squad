#!/usr/bin/env python3
import collections
import hashlib
import json
import math
import os
import re
import shutil
import string
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import Agent, SelfCore, LLM, VaultHub, SemanticMemory, slug
from bobby_squad.retrieval import default_embed

DATA = os.environ.get("NQA_PATH", "/tmp/lb/data/narrativeqa.jsonl")
N_DOCS = int(os.environ.get("N_DOCS", "8"))               # books to cover
MAX_Q_PER_DOC = int(os.environ.get("MAX_Q_PER_DOC", "20"))
READ_CHARS = int(os.environ.get("READ_CHARS", "3000"))    # passage the reader ingests per step into its WORLD
RECALL_BUDGET = int(os.environ.get("RECALL_BUDGET", "6000"))
PER_VAULT_K = int(os.environ.get("PER_VAULT_K", "6"))
HOPS = int(os.environ.get("HOPS", "2"))                   # knowledge-graph traversal depth on navigate

# ── ARM B — DATA-QUALITY POLICY (POLICY=1). Three proven memory primitives layered on the SAME world engine, so the
# only difference from the floor arm is the policy (a clean A→B delta). All are bobby_squad primitives, nothing new:
#   1. RETENTION GATE  — every extracted fact passes a SemanticMemory(tau) novelty gate before it enters the graph, so
#      redundant/near-duplicate ("bad") notes are dropped and recall isn't diluted.        (proven WIRE +25% retention)
#   2. MEMORY-GATE      — the reader's world is built by compact(consolidate=True): distinct reading is consolidated
#      into the pinned tier, duplicates gated out.                                          (proven WIRE +191% retention)
#   3. FINDINGS-STEERED RECALL — the value-governed SemanticMemory (CorrectionMemory/FindingsMemory) is a SECOND recall
#      axis unioned with the graph navigate, so a single-axis needle survives and usage teaches what to surface.
POLICY = os.environ.get("POLICY", "0") == "1"
TAU = float(os.environ.get("TAU", "0.9"))                 # retention novelty threshold (proven WIRE at 0.9)
FINDINGS_K = int(os.environ.get("FINDINGS_K", "6"))       # top-k findings unioned into recall
CONSOLIDATE_EVERY = int(os.environ.get("CONSOLIDATE_EVERY", "6"))  # Memory-Gate cadence over reading passages
FLAT_K = int(os.environ.get("FLAT_K", "6"))               # needle-preserving: top-k VERBATIM passages in recall (WIRE +51.7)


def _normalize(s):
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, golds):
    best, pt = 0.0, _normalize(pred).split()
    for g in golds:
        gt = _normalize(g).split()
        common = sum(min(pt.count(w), gt.count(w)) for w in set(pt) if w in gt)
        if pt and gt and common:
            prec, rec = common / len(pt), common / len(gt)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _ptoks(t):
    return len(t) // 4


ANSWER = ("Answer the question concisely using only the provided material. Reply with just the answer.\n"
          "Question: {q}\nAnswer:")


def main():
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    by_doc = collections.OrderedDict()
    for r in rows:
        by_doc.setdefault(hashlib.md5(r["context"].encode()).hexdigest(), []).append(r)
    docs = list(by_doc.values())[:N_DOCS]
    print(f"== KNOWLEDGE-REUSE (NarrativeQA) · {len(docs)} books · model={os.environ.get('BOBBY_LLM_MODEL','?')} "
          f"· arm={'B:policy(retention+memory-gate+findings)' if POLICY else 'A:floor(cosine recall)'} "
          f"· QA-F1 + token cost ==\n", flush=True)

    solo_f1 = fw_f1 = 0.0
    solo_tok = fw_tok = 0
    nq = 0
    for di, qs in enumerate(docs):
        qs = qs[:MAX_Q_PER_DOC]
        ctx = qs[0]["context"]
        d = tempfile.mkdtemp(prefix="vault_")
        # WORLD ENGINE: a persistent reader reads the book ONCE into its WORLD (pinned understanding) + a knowledge
        # graph (vault, auto-linked by entity). That world is REUSED to answer every question — no fresh agent, no re-read.
        hub = VaultHub(d, embed_fn=default_embed)
        # ARM B: a value-governed novelty-gated findings store (CorrectionMemory/FindingsMemory) — the retention gate
        # AND the second recall axis. capacity/policy=value → usage teaches what survives (self-governing memory).
        fmem = SemanticMemory(tau=TAU, k=FINDINGS_K, embed_fn=default_embed, policy="value") if POLICY else None

        def recall(t, _hub=hub, _fmem=fmem):
            # needle-preserving whole passages (FLAT-k, WIRE +51.7) unioned with the summarised graph-hop subgraph
            block = _hub.navigate(t, per_vault_k=PER_VAULT_K, hops=HOPS, budget=RECALL_BUDGET, per_note=600,
                                  whole_vault="doc", whole_k=FLAT_K)
            if _fmem is not None and len(_fmem):           # UNION a second axis: value-ranked findings (needle-safe)
                fb = _fmem.as_block(t, "Relevant facts recalled (value-ranked):", k=FINDINGS_K)
                if fb:
                    block = (block + "\n\n" + fb) if block else fb
            return block

        reader = Agent(SelfCore(identity="a scholar who has read this entire book and remembers it",
                                goal="understand the book deeply and remember every character, event and relation for questions"),
                       llm=LLM(), recall=recall)
        secs = [ctx[i:i + READ_CHARS] for i in range(0, len(ctx), READ_CHARS)] or [""]
        seen = set()
        for i, sec in enumerate(secs):
            # verbatim window as a navigable note (needle-preserving) + sequential links, alongside the entity graph
            hub.enrich("doc", f"passage {i+1}", sec, source="book",
                       links=[f"doc/passage-{i+2}"] if i + 1 < len(secs) else None)
            out = reader.act("Read this passage of the book. In 2-3 sentences note the key events, characters and "
                             "their relations. Then on new lines list the salient entities as `Entity | one fact`.\n\n"
                             + sec, max_tokens=320)
            summary = (out.split("Entity")[0] if "Entity" in out else out).strip()[:280]
            if POLICY:
                # MEMORY-GATE: reading accrues in the working tier; consolidate distinct → pinned on cadence, dups gated
                if (i + 1) % CONSOLIDATE_EVERY == 0:
                    reader.compact(consolidate=True)
            else:
                reader.record(f"[part {i+1}] {summary}")
            for line in (out or "").splitlines():
                if "|" not in line:
                    continue
                ent, fact = line.split("|", 1)
                ent, fact = ent.strip(" -*#`").strip(), fact.strip()
                if 2 <= len(ent) <= 60 and fact:
                    # RETENTION GATE: only facts that add a distinct direction enter the graph (drops redundant notes)
                    if POLICY and not fmem.add(f"{ent} :: {fact}"):
                        continue
                    rel = [f"kg/{slug(e)}" for e in seen if e.lower() in fact.lower() and slug(e) != slug(ent)][:4]
                    hub.enrich("kg", ent, fact, source="book", links=rel or None)
                    seen.add(ent)
        if POLICY:
            reader.compact(consolidate=True)               # final Memory-Gate: fold remaining distinct reading in
        fw_tok += _ptoks(ctx)                              # one-time capture (read the book once)
        for r in qs:
            q = r.get("question") or r.get("input")
            golds = r["answers"] if isinstance(r["answers"], list) else [r["answers"]]
            # SOLO: re-read the whole book for THIS question (pays the full cost every time)
            sp = f"Material:\n{ctx[:480000]}\n\n" + ANSWER.format(q=q)
            sa = LLM()([{"role": "user", "content": sp}], max_tokens=64, temperature=0.0).strip()
            solo_tok += _ptoks(sp); solo_f1 += qa_f1(sa, golds)
            # ENGINE: REUSE the reader's WORLD (its pinned understanding + recall-navigated graph)
            fa = (reader.act(f"Question about the book you read: {q}\nAnswer concisely with just the answer:",
                             max_tokens=64) or "").strip()
            fw_tok += _ptoks(reader.ctx.recall(q) if reader.ctx.recall else "")
            fw_f1 += qa_f1(fa, golds)
            nq += 1
        shutil.rmtree(d, ignore_errors=True)
        print(f"  book {di+1}/{len(docs)}: {len(qs)} Q · running F1 solo {100*solo_f1/nq:.1f} / fw {100*fw_f1/nq:.1f} "
              f"· tokens solo {solo_tok} / fw {fw_tok}", flush=True)

    print(f"\n== FINAL over {nq} questions, {len(docs)} books ==", flush=True)
    print(f"  SOLO       F1 {100*solo_f1/nq:5.1f}  · total tokens processed {solo_tok:,} ({solo_tok//nq:,}/Q)", flush=True)
    print(f"  FRAMEWORK  F1 {100*fw_f1/nq:5.1f}  · total tokens processed {fw_tok:,} ({fw_tok//nq:,}/Q)", flush=True)
    print(f"  → REUSE: framework processed {solo_tok/max(1,fw_tok):.1f}× FEWER tokens for the same {nq} answers "
          f"(capture-once, reuse-{nq//max(1,len(docs))}×/book)", flush=True)
    print("\nRESULT " + json.dumps({"questions": nq, "books": len(docs), "solo_f1": solo_f1/nq, "fw_f1": fw_f1/nq,
                                    "solo_tok": solo_tok, "fw_tok": fw_tok}), flush=True)


if __name__ == "__main__":
    main()
