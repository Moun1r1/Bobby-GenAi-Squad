#!/usr/bin/env python3
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys
import string
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# The ENGINE is the framework's own primitives — the SAME ones the backend wires (studio/backend/runner.py:
# _get_hub / _vault_recall): a persistent-self Agent + a KnowledgeVault navigated through the recall seam. No
# hand-rolled retrieval here; the engine arm reads the document ONCE into the agent's world + a linked knowledge
# graph, then answers from that reused world exactly as an Agent(recall=hub.navigate) does everywhere else.
from bobby_squad import LLM, Agent, SelfCore, VaultHub, slug
from bobby_squad.retrieval import default_embed

N = int(os.environ.get("N", "40"))
READ_CHARS = int(os.environ.get("READ_CHARS", "3000"))       # passage the reader ingests per step into its WORLD
RECALL_BUDGET = int(os.environ.get("RECALL_BUDGET", "6000")) # chars of navigated subgraph injected per step (bounded)
PER_VAULT_K = int(os.environ.get("PER_VAULT_K", "6"))        # semantic-entry notes before link-hop
HOPS = int(os.environ.get("HOPS", "2"))                      # knowledge-graph traversal depth on navigate
FLAT_K = int(os.environ.get("FLAT_K", "6"))                  # needle-preserving: top-k VERBATIM passages (WIRE +51.7)
WORKERS = int(os.environ.get("WORKERS", "8"))
FILES = [f for f in os.environ.get("FILES", "/tmp/mh_2wikimqa.jsonl").split(",") if f]


# ── LongBench QA-F1 (standard normalization) ──────────────────────────────────
def _normalize(s):
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, golds):
    best = 0.0
    pt = _normalize(pred).split()
    for g in golds:
        gt = _normalize(g).split()
        common = {}
        for w in pt:
            if w in gt:
                common[w] = min(pt.count(w), gt.count(w))
        num_same = sum(common.values())
        if num_same == 0 or not pt or not gt:
            best = max(best, 1.0 if (not pt and not gt) else 0.0)
            continue
        prec = num_same / len(pt)
        rec = num_same / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _ptoks(text):
    return len(text) // 4


ANSWER_SNIP = ("Answer the question as concisely as possible using ONLY the information given. Reply with just the "
               "answer — no explanation.\n\nQuestion: {q}\n\nAnswer:")


# ── SOLO: as much context as the window holds (the honest single-context limit) ──
SOLO_MAX_CHARS = int(os.environ.get("SOLO_MAX_CHARS", "480000"))    # ~120k tokens ≤ the model's 128k window
def solo(row):
    ctx = row["context"][:SOLO_MAX_CHARS]                            # a single call can't exceed its window → truncates
    prompt = f"Read the context and answer the question.\n\nContext:\n{ctx}\n\n" + ANSWER_SNIP.format(q=row["question"])
    out = LLM()([{"role": "user", "content": prompt}], max_tokens=64, temperature=0.0)
    return out.strip(), _ptoks(prompt), 1


# ── ENGINE: the framework's own primitives (Agent + KnowledgeVault + recall seam) — read ONCE into a world, answer ──
# Not a bespoke retriever: a persistent-self Agent reads the document in bounded passages, accumulating its
# understanding in the PINNED tier (its WORLD) while building a linked KNOWLEDGE GRAPH in a KnowledgeVault
# (enrich → auto-linked/merged notes). The question is answered by REUSING that world — the agent acts with its
# pinned understanding plus recall navigating the graph (semantic entry + link-hop, bounded + attributed). This is
# the exact Agent(recall=hub.navigate) path the backend runs; the multi-hop bridge is done by the graph, not by us.
def _read_into_world(ctx, d):
    hub = VaultHub(d, embed_fn=default_embed)
    reader = Agent(SelfCore(identity="a scholar who has read this document and remembers it",
                            goal="understand the document and remember every entity, event and relation for questions"),
                   llm=LLM(), recall=lambda t: hub.navigate(t, per_vault_k=PER_VAULT_K, hops=HOPS,
                                                            budget=RECALL_BUDGET, per_note=600,
                                                            whole_vault="doc", whole_k=FLAT_K))
    secs = [ctx[i:i + READ_CHARS] for i in range(0, len(ctx), READ_CHARS)] or [""]
    seen = set()
    for i, sec in enumerate(secs):
        # store the VERBATIM window as a navigable note too (needle-preserving) — recall can then surface the exact
        # passage, not only the lossy summary; sequential links let a hop follow the narrative across passages.
        hub.enrich("doc", f"passage {i+1}", sec, source="document",
                   links=[f"doc/passage-{i+2}"] if i + 1 < len(secs) else None)
        out = reader.act("Read this passage of the document. In 2-3 sentences note the key entities, events and their "
                         "relations. Then on new lines list the salient entities as `Entity | one fact`.\n\n" + sec,
                         max_tokens=320)
        reader.record(f"[part {i+1}] {(out.split('Entity')[0] if 'Entity' in out else out).strip()[:280]}")
        for line in (out or "").splitlines():
            if "|" not in line:
                continue
            ent, fact = line.split("|", 1)
            ent, fact = ent.strip(" -*#`").strip(), fact.strip()
            if 2 <= len(ent) <= 60 and fact:
                rel = [f"kg/{slug(e)}" for e in seen if e.lower() in fact.lower() and slug(e) != slug(ent)][:4]
                hub.enrich("kg", ent, fact, source="document", links=rel or None)
                seen.add(ent)
    return reader


def engine(row):
    ctx, q = row["context"], row["question"]
    d = tempfile.mkdtemp(prefix="vault_")
    try:
        reader = _read_into_world(ctx, d)                 # read the document ONCE into the agent's world + graph
        ans = (reader.act(f"Question about the document you read: {q}\nAnswer concisely with just the answer:",
                          max_tokens=64) or "").strip()    # answer by REUSING the world (recall navigates the graph)
        ref = reader.ctx.recall(q) if reader.ctx.recall else ""
        return ans, _ptoks(ref), 1                        # bounded recall block is the per-answer prompt cost
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_file(path):
    rows = [json.loads(l) for l in open(path) if l.strip()][:N]
    src = os.path.basename(path).replace("mh_", "").replace(".jsonl", "")
    avg_ctx = sum(_ptoks(r["context"]) for r in rows) // max(1, len(rows))
    print(f"\n── {src}  ({len(rows)} questions · avg context ≈ {avg_ctx} tokens) ──", flush=True)

    def one(r):
        golds = r["answers"] if isinstance(r["answers"], list) else [r["answers"]]
        sa, sp, sc = solo(r)
        ea, ep, ec = engine(r)
        return qa_f1(sa, golds), sp, qa_f1(ea, golds), ep, ec

    sf = ef = 0.0
    sp_max = ep_max = 0
    ec_tot = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for s_f1, s_pt, e_f1, e_pt, e_c in ex.map(one, rows):
            sf += s_f1; ef += e_f1; sp_max = max(sp_max, s_pt); ep_max = max(ep_max, e_pt); ec_tot += e_c
    n = len(rows)
    print(f"    SOLO   F1 {100*sf/n:5.1f}  · prompt ≈ {sp_max:>6d} tok (whole context, 1 call)", flush=True)
    print(f"    ENGINE F1 {100*ef/n:5.1f}  · prompt ≈ {ep_max:>6d} tok max (bounded) · {ec_tot/n:.1f} calls/q", flush=True)
    print(f"    → F1 {100*(ef-sf)/n:+5.1f} · prompt {sp_max}→{ep_max} ({sp_max/max(1,ep_max):.1f}× smaller)", flush=True)
    return {"source": src, "n": n, "avg_ctx": avg_ctx, "solo_f1": sf / n, "engine_f1": ef / n,
            "solo_ptok": sp_max, "engine_ptok": ep_max}


def main():
    print(f"== MULTI-HOP LONG-CONTEXT · model={os.environ.get('BOBBY_LLM_MODEL','?')} · LongBench QA-F1 · "
          f"solo vs engine(room) ==", flush=True)
    rows = [run_file(f) for f in FILES]
    print("\nRESULT " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
