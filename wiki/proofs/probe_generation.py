#!/usr/bin/env python3
import json
import os
import re
import shutil
import string
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import LLM, VaultHub
from bobby_squad.retrieval import EmbeddingRetriever, default_embed

N = int(os.environ.get("N", "25"))
READ = int(os.environ.get("READ", "1500"))          # passage size (chars)
TOPK = int(os.environ.get("TOPK", "4"))             # passages the engine is allowed to see (bounded)
HOPS = int(os.environ.get("HOPS", "2"))
BUDGET = int(os.environ.get("BUDGET", "6000"))
PER_NOTE = int(os.environ.get("PER_NOTE", "1500"))  # keep passages whole in the navigated block (needle-preserving)
FILE = os.environ.get("FILE", "/tmp/mh_2wikimqa.jsonl")


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


def _entities(s):
    return set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", s))


ANSWER = ("Answer the question as concisely as possible using ONLY the information given. Reply with just the "
          "answer — no explanation.\nQuestion: {q}\nAnswer:")


def _ans(prompt):
    return (LLM()([{"role": "user", "content": prompt}], max_tokens=64, temperature=0.0) or "").strip()


def main():
    rows = [json.loads(l) for l in open(FILE) if l.strip()][:N]
    print(f"== GENERATION-GAP PROBE · {os.path.basename(FILE)} · N={len(rows)} · top-k={TOPK} · "
          f"model={os.environ.get('BOBBY_LLM_MODEL','?')} · temp=0 ==\n", flush=True)
    sf = ff = gf = 0.0
    stok = ftok = gtok = 0
    n = 0
    for r in rows:
        q = r.get("input") or r.get("question")
        golds = r["answers"] if isinstance(r["answers"], list) else [r["answers"]]
        ctx = r["context"]
        secs = [ctx[i:i + READ] for i in range(0, len(ctx), READ)] or [""]
        # shared retrieval — cosine top-k (the strong axis measured earlier)
        cos = EmbeddingRetriever(embed_fn=default_embed)
        cos.add_many(secs)
        topk = cos.search(q, TOPK) or secs[:TOPK]

        # SOLO — full document
        sp = f"Material:\n{ctx[:480000]}\n\n" + ANSWER.format(q=q)
        sa = _ans(sp); stok += _ptoks(sp); sf += qa_f1(sa, golds)

        # FLAT-k — bounded verbatim passages, straight into the answer step
        fp = "Excerpts from the document:\n\n" + "\n\n---\n\n".join(topk) + "\n\n" + ANSWER.format(q=q)
        fa = _ans(fp); ftok += _ptoks(fp); ff += qa_f1(fa, golds)

        # GRAPH-hop — same passages as a vault; answer over the NAVIGATED subgraph (entry + entity link-hop)
        d = tempfile.mkdtemp(prefix="probe_")
        try:
            hub = VaultHub(d, embed_fn=default_embed)
            ent = [_entities(s) for s in secs]
            for i, s in enumerate(secs):
                links = [f"doc/passage-{j+1}" for j in range(len(secs)) if j != i and ent[i] & ent[j]][:5]
                hub.enrich("doc", f"passage {i+1}", s, source="document", links=links or None)
            block = hub.navigate(q, per_vault_k=TOPK, hops=HOPS, budget=BUDGET, per_note=PER_NOTE)
            gp = f"Retrieved from the document (linked passages):\n{block}\n\n" + ANSWER.format(q=q)
            ga = _ans(gp); gtok += _ptoks(gp); gf += qa_f1(ga, golds)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        n += 1
        if n % 5 == 0:
            print(f"  {n}/{len(rows)} · F1 solo {100*sf/n:.1f} / flat-k {100*ff/n:.1f} / graph-hop {100*gf/n:.1f}",
                  flush=True)

    def verdict(name, f1, tok):
        d = 100 * (f1 - sf) / n
        v = "WIRE" if d >= 1.0 else ("MARGINAL" if d >= -1.0 else "DELETE")
        print(f"  {name:10s} F1 {100*f1/n:5.1f}  · {tok:>9,} tok ({tok//n:,}/Q)  · ΔF1 {d:+5.1f} vs solo  · {v}",
              flush=True)

    print(f"\n== FINAL over {n} questions ==", flush=True)
    print(f"  SOLO       F1 {100*sf/n:5.1f}  · {stok:>9,} tok ({stok//n:,}/Q)  · control", flush=True)
    verdict("FLAT-k", ff, ftok)
    verdict("GRAPH-hop", gf, gtok)
    print("\nRESULT " + json.dumps({"n": n, "solo_f1": sf/n, "flatk_f1": ff/n, "graphhop_f1": gf/n,
                                    "solo_tok": stok, "flatk_tok": ftok, "graphhop_tok": gtok}), flush=True)


if __name__ == "__main__":
    main()
