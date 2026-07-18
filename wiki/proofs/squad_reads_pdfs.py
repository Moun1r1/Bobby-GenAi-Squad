#!/usr/bin/env python3
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import SelfCore, LLM, Society, LexicalRetriever
from bobby_squad.planning import extract_json

ROUNDS = 6
SECTION = 1500                 # chars per section
MAX_CHARS = 36000             # read up to ~first 15 pages of each paper (the dense core)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE = os.path.join(DATA, "pdfcache"); os.makedirs(CACHE, exist_ok=True)
CORPUS = json.load(open(os.path.join(DATA, "arxiv_corpus.json")))
NAMES = ["Ada", "Boole", "Cantor", "Dirac", "Euler"]

SOC = Society()
INDEX = LexicalRetriever()
_paper_cache = {}


def read_pdf(arxiv_id):
    """The agent OPENS THE ACTUAL PDF — download (cached) + extract text + chunk into sections."""
    if arxiv_id in _paper_cache:
        return _paper_cache[arxiv_id]
    base = arxiv_id.split("v")[0]
    path = os.path.join(CACHE, base.replace("/", "_") + ".pdf")
    try:
        if not os.path.exists(path):
            req = urllib.request.Request(f"https://arxiv.org/pdf/{base}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r, open(path, "wb") as f:
                f.write(r.read())
        from pypdf import PdfReader
        txt = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        txt = re.sub(r"\s+", " ", txt)[:MAX_CHARS]
        chunks = [txt[i:i + SECTION] for i in range(0, len(txt), SECTION)] or [""]
    except Exception as e:
        chunks = []
        SOC.broadcast("system", f"could not read PDF {base}: {type(e).__name__}")
    _paper_cache[arxiv_id] = chunks
    return chunks


# squad state
persona = {n: SelfCore(identity=f"{n}, an indexer with no specialty yet", goal="help the squad index the real papers") for n in NAMES}
claimed = {}                          # paper-index -> agent
current = {n: None for n in NAMES}    # paper-index the agent is currently reading
section = {n: {} for n in NAMES}      # agent -> {paper-index: last section read}
concepts = {n: [] for n in NAMES}


def claim_paper(name, llm):
    unclaimed = [i for i in range(len(CORPUS)) if i not in claimed]
    if not unclaimed:
        return None
    opts = "; ".join(f"{i}:[{CORPUS[i]['category']}] {CORPUS[i]['title'][:50]}" for i in unclaimed[:8])
    chatter = " | ".join(SOC.overheard(name)) or "(quiet)"
    prompt = (f"You are {persona[name].identity}. Your squad self-organizes with no assignments.\nRoom: {chatter}\n"
              f"UNCLAIMED papers (id:title): {opts}\nClaim ONE to read and index (cover breadth, avoid overlap). "
              'Respond ONLY JSON: {"claim": <paper id int>}')
    o = extract_json(llm([{"role": "user", "content": prompt}], max_tokens=40))
    try:
        c = int(o.get("claim"))
    except Exception:
        c = unclaimed[0]
    return c if c in unclaimed else unclaimed[0]


def index_section(name, pidx, llm):
    paper = CORPUS[pidx]
    chunks = read_pdf(paper["id"])
    if not chunks:
        return None
    pos = section[name].get(pidx, 0)
    if pos >= len(chunks):
        return "done"
    avail = chunks[pos:pos + 2]
    numbered = "\n\n".join(f"[section {pos + j}] {c}" for j, c in enumerate(avail))
    prompt = (f"You are {persona[name].identity}, reading the REAL PDF of '{paper['title'][:60]}' [{paper['category']}].\n"
              f"{numbered}\n\nDecide how many of these sections you can carefully index THIS turn (1..{len(avail)}) — "
              "denser = fewer — then extract precise index entries (objects, methods, results).\n"
              'Respond ONLY JSON: {"parsed":<int>,"entries":["<term> :: <gloss>", ...]}')
    o = extract_json(llm([{"role": "user", "content": prompt}], max_tokens=340))
    try:
        n = max(1, min(len(avail), int(o.get("parsed", 1))))
    except Exception:
        n = 1
    ents = [e for e in (o.get("entries") or []) if isinstance(e, str)][:6]
    for e in ents:
        INDEX.add(f"[{paper['category']} · {paper['id']} · sec {pos}-{pos + n - 1}] {e}")
    section[name][pidx] = pos + n
    concepts[name] += [e.split("::")[0].strip() for e in ents]
    return (n, len(chunks), pos + n)


def evolve(name, cat, llm):
    prompt = (f"You are {name}. From reading real papers you've learned: {', '.join(concepts[name][-8:])}.\n"
              "In ONE sentence give your evolved specialist identity (specialty + signature expertise). Just the sentence.")
    nid = (llm([{"role": "user", "content": prompt}], max_tokens=60) or "").strip().strip('"')
    if nid:
        persona[name].identity = nid


def main():
    print(f"SQUAD READS REAL PDFs — {len(CORPUS)} papers, {len(NAMES)} agents, {ROUNDS} rounds, self-organizing.\n")
    llm = LLM(temperature=0.5)
    for r in range(ROUNDS):
        for name in NAMES:
            p = current[name]
            if p is None or section[name].get(p, 0) >= len(read_pdf(CORPUS[p]["id"]) or [""]):
                nc = claim_paper(name, llm)
                if nc is None:
                    SOC.broadcast(name, "all papers claimed — standing by."); continue
                claimed[nc] = name; current[name] = nc; p = nc
                SOC.broadcast(name, f"opened PDF {CORPUS[p]['id']} [{CORPUS[p]['category']}] '{CORPUS[p]['title'][:40]}'")
            res = index_section(name, p, llm)
            if res in (None, "done"):
                current[name] = None; continue
            n, total, pos = res
            evolve(name, CORPUS[p]["category"], llm)
            SOC.broadcast(name, f"{CORPUS[p]['id']} sec {pos-n}-{pos-1}/{total} (parsed {n}) — now: {persona[name].identity[:48]}")

    print("── ROOM TRANSCRIPT ──")
    for nm, t in SOC.transcript():
        print(f"   {nm:6}: {t}")
    print("\n── EVOLVED SPECIALISTS ──")
    for n in NAMES:
        print(f"   {n:6} → {persona[n].identity}")
    print("\n── HOW DEEP INTO EACH CLAIMED PAPER (section reached) ──")
    for n in NAMES:
        got = " · ".join(f"{CORPUS[p]['id']}: sec {s}" for p, s in section[n].items())
        print(f"   {n:6}: {got or '—'}")
    print(f"\n   papers opened: {len(claimed)}/{len(CORPUS)} · index entries: {len(INDEX)}")
    for p in CORPUS[:2] + CORPUS[12:14]:
        terms = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", p["title"])][:3]
        hit = INDEX.search(" ".join(terms), k=1)
        print(f"   '{' '.join(terms)}' → {hit[0][:96] if hit else '(no hit)'}")


if __name__ == "__main__":
    main()
