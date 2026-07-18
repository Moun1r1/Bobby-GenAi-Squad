#!/usr/bin/env python3
import concurrent.futures as cf
import json
import math
import os
import re
import shutil
import string
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import Agent, SelfCore, LLM, VaultHub, slug
from bobby_squad.retrieval import default_embed

N = int(os.environ.get("N", "200"))
RECALL_BUDGET = int(os.environ.get("RECALL_BUDGET", "6000"))  # chars of navigated subgraph injected per step (bounded)
PER_VAULT_K = int(os.environ.get("PER_VAULT_K", "6"))         # semantic-entry notes before link-hop
SOLO_MAX_CHARS = int(os.environ.get("SOLO_MAX_CHARS", "480000"))   # ~120k tokens ≤ the 128k window
WORKERS = int(os.environ.get("WORKERS", "6"))
FILES = [f for f in os.environ.get("FILES", "/tmp/mh_2wikimqa.jsonl").split(",") if f]


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
        if not pt or not gt:
            continue
        if common:
            prec, rec = common / len(pt), common / len(gt)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def gold(a):
    return a if isinstance(a, list) else [a]


def extract(t):
    return (t or "").strip().splitlines()[-1].strip() if (t or "").strip() else ""


def _ptoks(t):
    return len(t) // 4


ANSWER = ("Answer the question using ONLY the provided document knowledge. Reply with just the answer — no "
          "explanation.\nQuestion: {q}\nAnswer:")


def solo(row):
    ctx = row["context"][:SOLO_MAX_CHARS]
    prompt = f"Document:\n{ctx}\n\n" + ANSWER.format(q=row["question"])
    out = LLM()([{"role": "user", "content": prompt}], max_tokens=64, temperature=0.0)
    return out.strip(), _ptoks(prompt)


READ_CHARS = int(os.environ.get("READ_CHARS", "3000"))       # passage the reader ingests per step into its WORLD
FLAT_K = int(os.environ.get("FLAT_K", "6"))                  # needle-preserving: top-k VERBATIM passages (WIRE +51.7)


def framework(row):
    ctx, q = row["context"], row["question"]
    # The ENGINE is the framework's own primitives — a persistent-self Agent reads the document ONCE into its WORLD
    # (pinned tier) while building a linked KNOWLEDGE GRAPH in the vault (enrich → auto-linked/merged notes), then
    # answers by REUSING that world: the agent acts with its pinned understanding plus recall navigating the graph
    # (semantic entry + bounded link-hop). Same Agent(recall=hub.navigate) path the backend runs — no bespoke loop.
    d = tempfile.mkdtemp(prefix="vault_")
    try:
        hub = VaultHub(d, embed_fn=default_embed)
        reader = Agent(SelfCore(identity="a scholar who has read this document and remembers it",
                                goal="understand the document and remember every entity, event and relation for questions"),
                       llm=LLM(), recall=lambda t: hub.navigate(t, per_vault_k=PER_VAULT_K, hops=2,
                                                                budget=RECALL_BUDGET, per_note=600,
                                                                whole_vault="doc", whole_k=FLAT_K))
        secs = [ctx[i:i + READ_CHARS] for i in range(0, len(ctx), READ_CHARS)] or [""]
        seen = set()
        for i, sec in enumerate(secs):
            # verbatim window as a navigable note (needle-preserving) + sequential links, alongside the entity graph
            hub.enrich("doc", f"passage {i+1}", sec, source="document",
                       links=[f"doc/passage-{i+2}"] if i + 1 < len(secs) else None)
            out = reader.act("Read this passage of the document. In 2-3 sentences note the key entities, events and "
                             "their relations. Then on new lines list the salient entities as `Entity | one fact`.\n\n"
                             + sec, max_tokens=320)
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
        ans = (reader.act(f"Question about the document you read: {q}\nAnswer concisely with just the answer:",
                          max_tokens=64) or "").strip()
        ref = reader.ctx.recall(q) if reader.ctx.recall else ""
        return ans, _ptoks(ref)                          # bounded recall block is the per-answer prompt cost
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_file(path):
    rows = [json.loads(l) for l in open(path) if l.strip()][:N]
    src = os.path.basename(path).replace("mh_", "").replace(".jsonl", "")
    avg = sum(_ptoks(r["context"]) for r in rows) // max(1, len(rows))
    print(f"\n── {src}  ({len(rows)} Q · avg context ≈ {avg} tok) ──", flush=True)

    def one(r):
        g = gold(r["answers"])
        sa, sp = solo(r)
        fa, fp = framework(r)
        return qa_f1(sa, g), sp, qa_f1(fa, g), fp

    sf = ff = 0.0
    sp_m = fp_m = 0
    n = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for s_f1, s_p, f_f1, f_p in ex.map(one, rows):
            sf += s_f1; ff += f_f1; sp_m = max(sp_m, s_p); fp_m = max(fp_m, f_p); n += 1
    sci = 1.96 * math.sqrt((sf/n) * (1 - sf/n) / n)
    fci = 1.96 * math.sqrt((ff/n) * (1 - ff/n) / n)
    print(f"    SOLO       F1 {100*sf/n:5.1f} ±{100*sci:.1f}  · prompt ≈ {sp_m} tok", flush=True)
    print(f"    FRAMEWORK  F1 {100*ff/n:5.1f} ±{100*fci:.1f}  · recall ≈ {fp_m} tok (vault-navigated)", flush=True)
    print(f"    → F1 {100*(ff-sf)/n:+5.1f} · prompt {sp_m}→{fp_m}", flush=True)
    return {"source": src, "n": n, "avg_ctx": avg, "solo_f1": sf/n, "framework_f1": ff/n, "solo_tok": sp_m, "fw_tok": fp_m}


def main():
    print(f"== FRAMEWORK-NATIVE benchmark (vault + recall + persistent-self) · model={os.environ.get('BOBBY_LLM_MODEL','?')} "
          f"· LongBench QA-F1 ==", flush=True)
    rows = [run_file(f) for f in FILES]
    print("\nRESULT " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
