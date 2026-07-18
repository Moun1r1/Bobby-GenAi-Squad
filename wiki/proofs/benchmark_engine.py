#!/usr/bin/env python3
import collections
import hashlib
import json
import os
import re
import shutil
import string
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import Agent, SelfCore, LLM, VaultHub, slug
from bobby_squad.retrieval import default_embed

DATA = os.environ.get("NQA_PATH", "/tmp/lb/data/narrativeqa.jsonl")
N_DOCS = int(os.environ.get("N_DOCS", "5"))
MAX_Q_PER_DOC = int(os.environ.get("MAX_Q_PER_DOC", "12"))
READ_CHARS = int(os.environ.get("READ_CHARS", "3000"))       # passage size the agent reads per step
HOPS = int(os.environ.get("HOPS", "2"))                      # graph traversal depth on navigate


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


# ── read the book ONCE into a persistent WORLD (pinned understanding) + a knowledge graph (vault) ──
def read_into_world(ctx, book_id, d):
    hub = VaultHub(d, embed_fn=default_embed)
    reader = Agent(SelfCore(identity="a scholar who has read this entire book and remembers it",
                            goal="understand the book deeply and remember every character, event and relation for questions"),
                   llm=LLM(), recall=lambda t: hub.navigate(t, per_vault_k=6, hops=HOPS, budget=6000, per_note=600))
    secs = [ctx[i:i + READ_CHARS] for i in range(0, len(ctx), READ_CHARS)] or [""]
    seen = set()
    for i, sec in enumerate(secs):
        # verbatim window as a navigable note (needle-preserving) + sequential links, alongside the entity graph
        hub.enrich("doc", f"passage {i+1}", sec, source=book_id,
                   links=[f"doc/passage-{i+2}"] if i + 1 < len(secs) else None)
        out = reader.act("Read this passage of the book. In 2-3 sentences note the key events, characters and their "
                         "relations (who is whose family/ally/enemy, where things happen). Then on new lines list the "
                         "salient entities as `Entity | one fact`.\n\n" + sec, max_tokens=320)
        # accumulate the reading into the agent's WORLD (pinned tier — survives, reused for every question)
        summary = out.split("Entity")[0].strip() if "Entity" in out else out.strip()
        reader.record(f"[part {i+1}] {summary[:280]}")
        # build the knowledge GRAPH in the vault (auto-linked + merged by entity)
        for line in (out or "").splitlines():
            if "|" not in line:
                continue
            ent, fact = line.split("|", 1)
            ent, fact = ent.strip(" -*#`").strip(), fact.strip()
            if 2 <= len(ent) <= 60 and fact:
                rel = [f"kg/{slug(e)}" for e in seen if e.lower() in fact.lower() and slug(e) != slug(ent)][:4]
                hub.enrich("kg", ent, fact, source=book_id, links=rel or None)
                seen.add(ent)
    kg = hub.vaults.get("kg")
    return reader, (len(kg.notes) if kg else 0)


def main():
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    by_doc = collections.OrderedDict()
    for r in rows:
        by_doc.setdefault(hashlib.md5(r["context"].encode()).hexdigest(), []).append(r)
    docs = list(by_doc.values())[:N_DOCS]
    print(f"== BOOK-ROOM ENGINE (read once → world + graph, reuse world per question) vs SOLO · NarrativeQA · "
          f"{len(docs)} books · model={os.environ.get('BOBBY_LLM_MODEL','?')} ==\n", flush=True)

    sf = ef = 0.0
    stok = etok = 0
    nq = 0
    for di, qs in enumerate(docs):
        qs = qs[:MAX_Q_PER_DOC]
        ctx = qs[0]["context"]
        d = tempfile.mkdtemp(prefix="kg_")
        reader, n_notes = read_into_world(ctx, f"book{di}", d)         # ONE-TIME: build the world (reused below)
        etok += _ptoks(ctx)
        print(f"  book {di+1}: read into world → {len(reader.ctx.progress)} pinned parts · {n_notes} graph-notes · "
              f"{len(qs)} questions", flush=True)
        for r in qs:
            q = r.get("question") or r.get("input")
            golds = r["answers"] if isinstance(r["answers"], list) else [r["answers"]]
            # SOLO: re-read the whole book for this question
            sp = f"Material:\n{ctx[:480000]}\n\nAnswer concisely using only the material. Question: {q}\nAnswer:"
            sa = LLM()([{"role": "user", "content": sp}], max_tokens=64, temperature=0.0).strip()
            stok += _ptoks(sp); sf += qa_f1(sa, golds)
            # ENGINE: REUSE the reader's WORLD (pinned understanding + recall-navigated graph) to answer
            ea = (reader.act(f"Question about the book you read: {q}\nAnswer concisely with just the answer:",
                             max_tokens=64) or "").strip()
            ref = reader.ctx.recall(q) if reader.ctx.recall else ""
            etok += _ptoks(ref); ef += qa_f1(ea, golds)
            nq += 1
        shutil.rmtree(d, ignore_errors=True)
        print(f"    → running F1 solo {100*sf/nq:.1f} / engine {100*ef/nq:.1f} · tokens solo {stok:,} / engine {etok:,}",
              flush=True)

    print(f"\n== FINAL over {nq} questions, {len(docs)} books ==", flush=True)
    print(f"  SOLO    F1 {100*sf/nq:5.1f}  · {stok:,} tokens ({stok//nq:,}/Q)", flush=True)
    print(f"  ENGINE  F1 {100*ef/nq:5.1f}  · {etok:,} tokens ({etok//nq:,}/Q)", flush=True)
    print(f"  → engine used {stok/max(1,etok):.1f}× fewer tokens · F1 {100*(ef-sf)/nq:+.1f}", flush=True)
    print("\nRESULT " + json.dumps({"questions": nq, "books": len(docs), "solo_f1": sf/nq, "engine_f1": ef/nq,
                                    "solo_tok": stok, "engine_tok": etok}), flush=True)


if __name__ == "__main__":
    main()
