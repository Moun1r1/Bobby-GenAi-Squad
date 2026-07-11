#!/usr/bin/env python3
"""SELF-ORGANIZING SQUAD that READS REAL CODEBASES END TO END — the squad_reads_pdfs pattern, for code.

Same logic as squad_reads_pdfs.py (no static assignment; agents CLAIM an unclaimed codebase, read it SECTION BY
SECTION, SELF-PACE by density — the agent itself decides how many sections it can carefully index this turn, the
auto-split — track EXACTLY which section it reached, and when a codebase is fully read they claim another). The only
change is read_pdf → read_code: instead of a PDF the agent opens the actual source of a repo, concatenated and
chunked. BOTH agents read; coverage is self-monitored (the agent says when it's done), never a hardcoded threshold.

The prompt each turn holds only the CURRENT sections (bounded) while the shared INDEX accumulates the whole
codebase — so a huge repo is read end to end with no context blowup. Run:
  HORIZON_APPS=/tmp/deep_corpus GA_LLM_URL=... python3 examples/squad_reads_code.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import SelfCore, LLM, Society, LexicalRetriever         # noqa: E402
from bobby_squad.planning import extract_json                            # noqa: E402

ROUNDS = int(os.environ.get("CODE_ROUNDS", "8"))
SECTION = 1600                                                                  # chars per section
MAX_CHARS = int(os.environ.get("CODE_MAXCHARS", "44000"))                       # dense core read per codebase
SRC = (".py", ".ts", ".tsx", ".js", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".rb", ".md")
SKIP = {"node_modules", ".git", "__pycache__", "dist", "build", ".venv", "venv", "vendor", "target", "test", "tests"}
APPS = os.environ.get("HORIZON_APPS", "/tmp/deep_corpus")
NAMES = ["Ada", "Boole"]                                                        # BOTH agents read the code

CORPUS = sorted([d for d in os.listdir(APPS)
                 if os.path.isdir(os.path.join(APPS, d)) and not d.startswith(".")])
SOC = Society()
INDEX = LexicalRetriever()
_code_cache = {}


def read_code(name):
    """The agent OPENS THE ACTUAL SOURCE of the repo: gather significant source files (README + biggest source),
    concatenate with file headers, chunk into sections — the exact analogue of read_pdf's page-chunking."""
    if name in _code_cache:
        return _code_cache[name]
    root = os.path.join(APPS, name)
    files = []
    for r, ds, fs in os.walk(root):
        ds[:] = [d for d in ds if d not in SKIP and not d.startswith(".")]
        if r[len(root):].count(os.sep) > 3:
            ds[:] = []; continue
        for fn in fs:
            if fn.endswith(SRC):
                p = os.path.join(r, fn)
                try:
                    files.append((os.path.getsize(p), p))
                except OSError:
                    pass
    files.sort(key=lambda x: (not x[1].lower().endswith(("readme.md",)), -x[0]))   # README first, then largest
    blob, used = [], 0
    for _, p in files:
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        header = f"\n\n===== FILE {os.path.relpath(p, root)} =====\n"
        blob.append(header + txt)
        used += len(header) + len(txt)
        if used >= MAX_CHARS:
            break
    full = re.sub(r"[ \t]+", " ", "".join(blob))[:MAX_CHARS]
    chunks = [full[i:i + SECTION] for i in range(0, len(full), SECTION)] or [""]
    _code_cache[name] = chunks
    return chunks


persona = {n: SelfCore(identity=f"{n}, a code cartographer with no specialty yet", goal="help the squad read the "
                       "real codebases end to end") for n in NAMES}
claimed, current, section, concepts = {}, {n: None for n in NAMES}, {n: {} for n in NAMES}, {n: [] for n in NAMES}


def claim_repo(name, llm):
    unclaimed = [i for i in range(len(CORPUS)) if i not in claimed]
    if not unclaimed:
        return None
    opts = "; ".join(f"{i}:{CORPUS[i]}" for i in unclaimed[:10])
    chatter = " | ".join(SOC.overheard(name)) or "(quiet)"
    prompt = (f"You are {persona[name].identity}. The squad self-organizes, no assignments.\nRoom: {chatter}\n"
              f"UNCLAIMED codebases (id:name): {opts}\nClaim ONE to read end to end (cover breadth, avoid overlap). "
              'Respond ONLY JSON: {"claim": <id int>}')
    o = extract_json(llm([{"role": "user", "content": prompt}], max_tokens=40))
    try:
        c = int(o.get("claim"))
    except Exception:
        c = unclaimed[0]
    return c if c in unclaimed else unclaimed[0]


def index_section(name, ridx, llm):
    repo = CORPUS[ridx]
    chunks = read_code(repo)
    pos = section[name].get(ridx, 0)
    if pos >= len(chunks):
        return "done"
    avail = chunks[pos:pos + 2]
    numbered = "\n\n".join(f"[section {pos + j}] {c}" for j, c in enumerate(avail))
    prompt = (f"You are {persona[name].identity}, reading the REAL source of the '{repo}' codebase.\n{numbered}\n\n"
              f"Decide how many of these sections you can carefully index THIS turn (1..{len(avail)}) — denser code = "
              "fewer — then extract precise index entries (modules, key functions/types, what they do).\n"
              'Respond ONLY JSON: {"parsed":<int>,"entries":["<symbol/file> :: <what it does>", ...]}')
    o = extract_json(llm([{"role": "user", "content": prompt}], max_tokens=340))
    try:
        n = max(1, min(len(avail), int(o.get("parsed", 1))))
    except Exception:
        n = 1
    ents = [e for e in (o.get("entries") or []) if isinstance(e, str)][:6]
    for e in ents:
        INDEX.add(f"[{repo} · sec {pos}-{pos + n - 1}] {e}")
    section[name][ridx] = pos + n
    concepts[name] += [e.split("::")[0].strip() for e in ents]
    return (n, len(chunks), pos + n)


def evolve(name, llm):
    prompt = (f"You are {name}. From the code you've read you've learned about: {', '.join(concepts[name][-8:])}.\n"
              "In ONE sentence give your evolved specialist identity (specialty + signature expertise). Just the sentence.")
    nid = (llm([{"role": "user", "content": prompt}], max_tokens=60) or "").strip().strip('"')
    if nid:
        persona[name].identity = nid


def main():
    print(f"SQUAD READS REAL CODE — {len(CORPUS)} codebases, {len(NAMES)} agents, {ROUNDS} rounds, self-organizing.\n",
          flush=True)
    llm = LLM(temperature=0.5)
    for r in range(ROUNDS):
        for name in NAMES:
            p = current[name]
            if p is None or section[name].get(p, 0) >= len(read_code(CORPUS[p])):
                nc = claim_repo(name, llm)
                if nc is None:
                    SOC.broadcast(name, "all codebases claimed — standing by."); continue
                claimed[nc] = name; current[name] = nc; p = nc
                SOC.broadcast(name, f"opened '{CORPUS[p]}' ({len(read_code(CORPUS[p]))} sections)")
            res = index_section(name, p, llm)
            if res in (None, "done"):
                current[name] = None; continue
            n, total, pos = res
            evolve(name, llm)
            SOC.broadcast(name, f"{CORPUS[p]} sec {pos-n}-{pos-1}/{total} (parsed {n}) — now: {persona[name].identity[:46]}")
        print(f"[round {r+1}/{ROUNDS}] index={len(INDEX)} · claimed={len(claimed)}/{len(CORPUS)}", flush=True)

    print("\n── HOW DEEP INTO EACH CLAIMED CODEBASE (section reached / total) ──")
    for nm in NAMES:
        for ridx, s in section[nm].items():
            print(f"   {nm:6} {CORPUS[ridx]:14} sec {s}/{len(read_code(CORPUS[ridx]))}"
                  + ("  ✅ END-TO-END" if s >= len(read_code(CORPUS[ridx])) else "  (in progress)"), flush=True)
    print("\n── EVOLVED SPECIALISTS ──")
    for n in NAMES:
        print(f"   {n:6} → {persona[n].identity}", flush=True)
    print(f"\n   codebases opened: {len(claimed)}/{len(CORPUS)} · index entries: {len(INDEX)}", flush=True)
    for repo in CORPUS[:4]:
        hit = INDEX.search(repo, k=1)
        print(f"   query '{repo}' → {hit[0][:96] if hit else '(no hit)'}", flush=True)


if __name__ == "__main__":
    main()
