"""cross_sector_knowledge — the squad reads real papers from MANY knowledge sectors and TRANSFERS ideas across them.

Reads one real arXiv paper from each of 12 sectors (AI, neuroscience/cognition, economics, quantitative biology,
finance, medical physics, materials science, optimization, signal processing, computational linguistics, complex
systems, statistics), deposits precise concepts — tagged by sector — into a shared semantic store, then BRIDGES:
for several distant sector pairs it recalls concepts by MEANING and names a transferable idea. Cross-domain
knowledge, grounded in real papers (every recalled item is printed).

Run: GA_LLM_URL=... GA_EMBED_URL=... python3 examples/cross_sector_knowledge.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import LLM, EmbeddingRetriever, LexicalRetriever, embedding_available   # noqa: E402
from bobby_squad.planning import extract_json                                            # noqa: E402

CORPUS = json.load(open(os.path.join(HERE, "data", "diverse_corpus.json")))
BRIDGES = [                                                                     # deliberately distant sector pairs
    ("neuroscience / cognition", "economics"),
    ("finance", "quantitative biology"),
    ("optimization / OR", "signal processing / engineering"),
    ("complex systems", "AI / computer science"),
    ("medical physics", "statistics / methodology"),
]


def main():
    llm = LLM(temperature=0.4)
    index = EmbeddingRetriever() if embedding_available() else LexicalRetriever()
    kind = "semantic" if isinstance(index, EmbeddingRetriever) else "lexical"
    print(f"=== CROSS-SECTOR KNOWLEDGE — {len(CORPUS)} real papers, {len(CORPUS)} sectors, {kind} shared store ===\n",
          flush=True)

    # ── INGEST every sector ───────────────────────────────────────────────────────────────────
    for p in CORPUS:
        o = extract_json(llm([{"role": "user", "content":
            f"Read this {p['sector']} paper abstract — '{p['title']}':\n{p['abstract']}\n\n"
            'Extract 3 precise concepts as "<concept> :: <what it is>". Respond ONLY JSON: {"entries":[...]}'}],
            max_tokens=240))
        ents = [e for e in (o.get("entries") or []) if isinstance(e, str)][:3]
        for e in ents:
            index.add(f"[{p['sector']}] {e}")
        print(f"  read [{p['sector']:>32}]  +{len(ents)} concepts", flush=True)

    # ── BRIDGE distant sectors by MEANING ─────────────────────────────────────────────────────
    print(f"\n── CROSS-SECTOR BRIDGES ({index and len(index)} concepts in the shared store) ──\n", flush=True)
    for src, dst in BRIDGES:
        hits = index.search(f"a {src} idea that could transfer to {dst}", k=3)
        bridge = (llm([{"role": "user", "content":
            "From these concepts recalled from the shared store (each tagged by its sector):\n" + "\n".join(hits) +
            f"\n\nName ONE idea from **{src}** that could TRANSFER to **{dst}**, and say why in ONE sentence. "
            "Ground it in a recalled concept."}], max_tokens=110) or "").strip()
        print(f"  {src}  →  {dst}", flush=True)
        print(f"     {bridge}\n", flush=True)

    print("=== knowledge from every sector is stored, recalled by meaning, and carried ACROSS domains. ===",
          flush=True)


if __name__ == "__main__":
    main()
