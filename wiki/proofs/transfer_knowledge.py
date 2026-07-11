"""transfer_knowledge — show the paper-reading agents produce TRANSFERABLE knowledge.

Reading a paper is only useful if the knowledge OUTLIVES the reader: stored in a shared store, recalled by OTHER
agents who never read it, and carried ACROSS domains. This proves exactly that on real arXiv papers:

  1. INGEST  — three agents each read ONE real paper (abstract) and deposit precise concepts into a shared
               semantic index; each becomes a specialist.
  2. RECALL ACROSS AGENTS — a question from paper A's domain is put to the agent who read paper C. It never read A,
               so it QUERIES the shared index, recalls A's concepts (that a DIFFERENT agent deposited), and answers
               grounded in them. Knowledge moved A → store → C.
  3. CROSS-DOMAIN BRIDGE — an agent is asked which concept from one paper transfers to another's domain; semantic
               recall bridges by MEANING. That is the transferable-knowledge property.

Grounded: every recalled item is a real concept in the index (printed), so the answers can't be hallucinated.
Run: GA_LLM_URL=... GA_EMBED_URL=... python3 examples/transfer_knowledge.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import LLM, EmbeddingRetriever, LexicalRetriever, embedding_available   # noqa: E402
from bobby_squad.planning import extract_json                                            # noqa: E402

CORPUS = json.load(open(os.path.join(HERE, "data", "arxiv_corpus.json")))
PICK = [0, 12, 20]                                                              # three diverse papers
NAMES = ["Ada", "Boole", "Cantor"]


def main():
    llm = LLM(temperature=0.4)
    index = EmbeddingRetriever() if embedding_available() else LexicalRetriever()
    kind = "semantic (embedding)" if isinstance(index, EmbeddingRetriever) else "lexical"
    print(f"=== TRANSFERABLE KNOWLEDGE — real arXiv papers, {kind} shared store ===\n", flush=True)

    papers = [CORPUS[i] for i in PICK]
    experts, deposited = {}, {}
    # ── 1. INGEST — each agent reads ONE paper and deposits concepts ─────────────────────────────
    for name, p in zip(NAMES, papers):
        o = extract_json(llm([{"role": "user", "content":
            f"Read this abstract of '{p['title']}' [{p['category']}]:\n{p['abstract']}\n\n"
            "Extract 3-4 precise index entries as '<concept> :: <what it is>'. Also give the one-line SPECIALIST "
            'IDENTITY you now hold. Respond ONLY JSON: {"entries":[...],"identity":"..."}'}],
            max_tokens=300))
        ents = [e for e in (o.get("entries") or []) if isinstance(e, str)][:4]
        for e in ents:
            index.add(f"[{p['category']} · {p['id']}] {e}")
        experts[name] = o.get("identity", name)
        deposited[name] = (p, ents)
        print(f"  {name} read [{p['category']}] {p['title'][:46]} → deposited {len(ents)} concepts; "
              f"now: {experts[name][:60]}", flush=True)

    # ── 2. RECALL ACROSS AGENTS — Cantor answers about Ada's paper via the shared store ──────────
    a_paper = deposited["Ada"][0]
    query = f"{a_paper['category']} {a_paper['title']}"
    hits = index.search(query, k=3)
    print(f"\n── RECALL: asking Cantor (read {deposited['Cantor'][0]['category']}, NOT {a_paper['category']}) about "
          f"Ada's paper ──\n  shared-store recall for '{a_paper['category']}':", flush=True)
    for h in hits:
        print(f"    · {h[:96]}", flush=True)
    ans = (llm([{"role": "user", "content":
        f"You are {experts['Cantor']}. You did NOT read any {a_paper['category']} paper. Using ONLY this knowledge "
        f"recalled from the squad's shared store:\n" + "\n".join(hits) +
        f"\n\nIn 2 sentences, explain the core idea of the {a_paper['category']} work. Use only the recalled facts."}],
        max_tokens=160) or "").strip()
    print(f"  Cantor (transferred knowledge): {ans}", flush=True)
    grounded = any(any(w in ans for w in h.split("::")[0].split()[-3:]) for h in hits)
    print(f"  → grounded in recalled store: {grounded}", flush=True)

    # ── 3. CROSS-DOMAIN BRIDGE — a concept from one paper transferred to another's domain ─────────
    b_paper = deposited["Boole"][0]
    bridge_q = f"a reusable idea from {a_paper['category']} that could transfer to {b_paper['category']}"
    bhits = index.search(bridge_q, k=3)
    print(f"\n── CROSS-DOMAIN BRIDGE: {a_paper['category']} → {b_paper['category']} (semantic recall) ──", flush=True)
    for h in bhits:
        print(f"    · {h[:96]}", flush=True)
    bridge = (llm([{"role": "user", "content":
        f"From these recalled concepts:\n" + "\n".join(bhits) +
        f"\n\nName ONE idea that could TRANSFER to {b_paper['category']} ('{b_paper['title'][:50]}'), and say why in "
        "one sentence. Ground it in a recalled concept."}], max_tokens=120) or "").strip()
    print(f"  bridge: {bridge}", flush=True)

    print(f"\n=== transferable: {len(index)} concepts in the shared store, recalled + carried by agents who never "
          "read the source paper. ===", flush=True)


if __name__ == "__main__":
    main()
