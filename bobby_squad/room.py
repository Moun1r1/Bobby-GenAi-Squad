import json
import math
import os
import re
import urllib.request

from .llm import LLM
from .core import SelfCore
from .retrieval import LexicalRetriever, EmbeddingRetriever, embedding_available, _cos
from .society import Society
from .planning import extract_json

SECTION = 1500
MAX_CHARS = 30000
MAX_SECTIONS = 6            # index at most the first N sections (dense core) of each book


def read_pdf(arxiv_id, cache_dir):
    """Open the real PDF: download (cached) + extract text + chunk. Needs pypdf; returns [] if unavailable."""
    base = arxiv_id.split("v")[0]
    path = os.path.join(cache_dir, base.replace("/", "_") + ".pdf")
    try:
        if not os.path.exists(path):
            req = urllib.request.Request(f"https://arxiv.org/pdf/{base}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r, open(path, "wb") as f:
                f.write(r.read())
        from pypdf import PdfReader
        txt = re.sub(r"\s+", " ", "\n".join((p.extract_text() or "") for p in PdfReader(path).pages))[:MAX_CHARS]
        return [txt[i:i + SECTION] for i in range(0, len(txt), SECTION)][:MAX_SECTIONS]
    except Exception:
        return []


class KnowledgeRoom:
    def __init__(self, corpus, state_dir, llm=None, retention_tau=None):
        self.corpus = corpus                       # [{id, title, category}]  (books in the room)
        self.dir = state_dir
        self.pdfs = os.path.join(state_dir, "pdfs")
        os.makedirs(self.pdfs, exist_ok=True)
        self.llm = llm or LLM(temperature=0.4)
        self.index = LexicalRetriever()            # durable lexical store (query + fallback recall)
        self.semantic = EmbeddingRetriever() if embedding_available() else None   # paraphrase-robust recall (WIRE)
        # Privileged-subset retention gate (opt-in): drop a new memory that adds no distinct direction
        # (max cosine to existing ≥ τ). Proven WIRE at τ=0.9 — preserves recall, 16% smaller, beats random
        # pruning. See gains/retention_gains.py. None = off (keep everything).
        self.retention_tau = retention_tau
        self.experts = {}                          # domain -> {identity, concepts[], papers[], sections}
        self.progress = {}                         # paper_id -> sections indexed
        self._chunks = {}
        self.load()

    # ── persistence: knowledge compounds across sessions ──
    def load(self):
        ip = os.path.join(self.dir, "index.jsonl")
        texts = []
        if os.path.exists(ip):
            for line in open(ip):
                t = json.loads(line)["t"]
                self.index.add(t); texts.append(t)
        if self.semantic is not None:                           # reuse persisted vectors; embed only what's missing
            vp = os.path.join(self.dir, "vectors.json")
            if os.path.exists(vp):
                self.semantic.cache = json.load(open(vp))
            self.semantic.add_many(texts)
        ep = os.path.join(self.dir, "experts.json")
        if os.path.exists(ep):
            self.experts = json.load(open(ep))
        pp = os.path.join(self.dir, "progress.json")
        if os.path.exists(pp):
            self.progress = json.load(open(pp))

    def save(self):
        with open(os.path.join(self.dir, "index.jsonl"), "w") as f:
            for t in self.index.docs:
                f.write(json.dumps({"t": t}) + "\n")
        if self.semantic is not None:
            json.dump(self.semantic.cache, open(os.path.join(self.dir, "vectors.json"), "w"))
        json.dump(self.experts, open(os.path.join(self.dir, "experts.json"), "w"), indent=1)
        json.dump(self.progress, open(os.path.join(self.dir, "progress.json"), "w"), indent=1)

    def _add_entry(self, text):
        """Append a memory to BOTH stores so semantic recall stays in sync with the durable lexical index. With the
        privileged-subset gate on (retention_tau), a redundant memory — one within τ cosine of an existing one — is
        dropped (it adds no distinct direction). Returns True if kept. Proven WIRE: gains/retention_gains.py."""
        if self.retention_tau and self.semantic is not None and len(self.semantic):
            v = self.semantic.embed_fn([self.semantic.dp + text])
            if v and v[0]:
                if any(_cos(v[0], ev) >= self.retention_tau for ev in self.semantic.vecs):
                    return False                                       # redundant → privileged gate drops it
                self.index.add(text)                                   # distinct → keep (reuse the computed vector)
                self.semantic.cache[text] = v[0]
                self.semantic.docs.append(text); self.semantic.vecs.append(v[0])
                return True
        self.index.add(text)
        if self.semantic is not None:
            self.semantic.add(text)
        return True

    def _recall(self, note, k=2):
        """Recall from long-term memory. Embedding recall (paraphrase-robust, the proven WIRE) when available;
        lexical-by-term otherwise."""
        if self.semantic is not None and len(self.semantic):
            hits = self.semantic.search(note, k=k)             # full note (term+gloss) → match by meaning
        else:
            hits = self.index.search(note.split("::")[0], k=k)  # fallback: lexical by term
        return [h.split("] ", 1)[-1] for h in hits]

    # ── reusable outputs ──
    def query(self, q, k=5):
        return self.index.search(q, k)

    def load_expert(self, domain):
        """The transferable payoff: hand back the evolved specialist as a SelfCore to reuse in any Agent."""
        e = self.experts.get(domain)
        return SelfCore(identity=e["identity"], goal=f"apply my {domain} expertise") if e else None

    def ingest_text(self, doc_id, category, text, soc=None, max_sections=12):
        """Ingest ANY raw text (not just arXiv PDFs) through the same assess→recall→index→grow-expert pipeline:
        overlap/relevance filtered by semantic recall, durable + queryable, and it evolves the category expert.
        Returns (society, stats)."""
        soc = soc or Society()
        self.stats = {"added": 0, "redundant": 0, "irrelevant": 0}
        text = re.sub(r"\s+", " ", text)[:SECTION * max_sections]
        secs = [text[i:i + SECTION] for i in range(0, len(text), SECTION)][:max_sections]
        for s in range(self.progress.get(doc_id, 0), len(secs)):
            rel, ents, overlap = self._assess(category, {"id": doc_id}, s, secs[s])
            self.progress[doc_id] = max(self.progress.get(doc_id, 0), s + 1)
            if not rel:
                self.stats["irrelevant"] += 1
            elif not ents:
                self.stats["redundant"] += 1
                soc.broadcast("reader", f"{doc_id} sec {s}: overlaps existing knowledge ({overlap})")
            else:
                for e in ents:
                    self._add_entry(f"[{category} · {doc_id} · sec {s}] {e}")
                self.stats["added"] += len(ents)
                self._grow_expert(category, ents, doc_id)
                soc.broadcast("reader", f"{doc_id} sec {s} [{category}]: +{len(ents)} new")
        if category in self.experts:
            self._evolve_expert(category)
        self.save()
        return soc, self.stats

    def _search(self, q, k=8):
        if self.semantic is not None and len(self.semantic):
            return self.semantic.search(q, k)
        return self.index.search(q, k)

    def extend(self, lens_domain, max_ideas=8):
        """Flexible generalization (a workspace property) turned on the framework itself: take the lens domain's
        own principles — RECALLED from what was ingested, not hard-coded — and extend each into EVERY other
        domain, grounded in that domain's real recalled memories, proposing a reusable primitive for the
        generative multi-agent framework. Cross-domain bridging runs on the wired semantic recall. Proposals are
        GROUNDED hypotheses (they cite real memories); none is wired without a separate gain-proof."""
        principles = [h.split("] ", 1)[-1] for h in self._search(
            "workspace property verbal report directed modulation selectivity privileged subset internal reasoning", 6)]
        if not principles:
            return []
        targets = [d for d in self.experts if d != lens_domain]
        out = []
        for d in targets:
            hits = self._search(f"{d} central concept method result", 12)               # ground in REAL domain-d memory
            ground = [h.split("] ", 1)[-1] for h in hits if h.startswith(f"[{d} ")][:4]
            if not ground:
                ground = self.experts[d]["concepts"][-4:]
            pr = principles[len(out) % len(principles)]
            o = extract_json(self.llm([{"role": "user", "content":
                f"A principle from cognitive architecture (Global Workspace / J-Lens):\n  {pr}\n\n"
                f"Real material from the '{d}' domain of a knowledge world:\n- " + "\n- ".join(ground) +
                f"\n\nExtend that principle INTO one reusable primitive for a generative multi-agent framework, "
                f"specific to what '{d}' reasoning needs. Give a name, a one-line mechanism, and a one-line "
                'FALSIFIABLE check. ONLY JSON: {"primitive":"<name>","mechanism":"<how>","test":"<check>"}'}],
                max_tokens=220))
            if isinstance(o.get("primitive"), str) and o["primitive"].strip():
                out.append({"domain": d, "lens": pr[:64], "primitive": o["primitive"].strip(),
                            "mechanism": str(o.get("mechanism", "")).strip(), "test": str(o.get("test", "")).strip(),
                            "grounded_in": ground})
            if len(out) >= max_ideas:
                break
        return out

    def abstract(self, proposals):
        """The step that actually EXTENDS the framework: from per-domain extensions, distill the DOMAIN-INDEPENDENT
        primitive(s) that recur across them — the invariant, stripped of any single domain's specifics. Discovery
        was the union (extend); this is consensus-to-abstract (keep what generalizes). Still a GROUNDED hypothesis."""
        if len(proposals) < 2:
            return []
        body = "\n".join(f"- ({p['domain']}) {p['primitive']}: {p['mechanism']}" for p in proposals)
        o = extract_json(self.llm([{"role": "user", "content":
            "These are domain-specific extensions of Global-Workspace principles into a generative multi-agent "
            f"framework:\n{body}\n\nAcross them, identify 1-2 DOMAIN-INDEPENDENT reusable primitives the framework "
            "should gain — the invariant that recurs, stripped of any single domain's specifics. For each: a name, "
            "the domains it spans, a one-line mechanism, and a one-line FALSIFIABLE check. "
            'ONLY JSON: {"primitives":[{"name":"..","spans":[".."],"mechanism":"..","test":".."}]}'}], max_tokens=420))
        return [p for p in (o.get("primitives") or []) if isinstance(p, dict) and p.get("name")]

    def experiment(self, concept, max_trials=6, margin=0.2):
        """A CONTROLLED in-world PROOF — the J-Lens ablation method with its own falsifiability axis. Is the
        privileged memory actually LOAD-BEARING for the world's reasoning about `concept`? Answer a probe from the
        recalled context, then measure term-retention when we ablate the PRIVILEGED memory (treatment) vs a
        BACKGROUND memory from the SAME context (control). PROVEN-in-world iff the privileged ablation degrades
        reasoning meaningfully MORE than the background ablation, stably across patient trials. The control axis
        cancels the proxy's biases, so the differential — not the raw number — is the real, deterministic signal.
        Non-destructive (ablation applied to the recalled set; the store is untouched)."""
        ctx = self._search(concept, 8)
        if len(ctx) < 4:
            return {"proven": False, "trials": 0, "detail": "insufficient context"}
        privileged, background = ctx[0], ctx[-1]               # most- vs least-relevant node in the same context
        key = list(dict.fromkeys(re.findall(r"[a-zA-Z]{5,}", privileged.split("::")[-1])))[:5]
        if not key:
            return {"proven": False, "trials": 0, "detail": "no key terms"}
        q = f"Using ONLY these notes, explain '{concept}' precisely and technically:\n- "

        def retain(notes):
            a = (self.llm([{"role": "user", "content": q + "\n- ".join(notes)}], max_tokens=110) or "").lower()
            return sum(t.lower() in a for t in key) / len(key)

        t_degr, c_degr = [], []
        for _ in range(max_trials):
            base = retain(ctx[:5])
            if base == 0:
                continue
            treat = retain([x for x in ctx[:6] if x != privileged][:5])   # ablate PRIVILEGED
            ctrl = retain([x for x in ctx[:6] if x != background][:5])     # ablate BACKGROUND (control axis)
            t_degr.append(max(0.0, (base - treat) / base))
            c_degr.append(max(0.0, (base - ctrl) / base))
        n = len(t_degr)
        tm = sum(t_degr) / n if n else 0.0
        cm = sum(c_degr) / n if n else 0.0
        wins = sum(td > cd for td, cd in zip(t_degr, c_degr))
        eff = round(tm - cm, 2)
        proven = bool(n >= 3 and eff >= margin and wins / n >= 0.6)        # data-driven verdict, sign-stable
        return {"proven": proven, "effect": eff, "treatment": round(tm, 2), "control": round(cm, 2),
                "sign_stable": round(wins / n, 2) if n else 0.0, "trials": n,
                "privileged": privileged.split("] ", 1)[-1][:60], "background": background.split("] ", 1)[-1][:60],
                "key_terms": key}

    # ── internals ──
    def _sections(self, i):
        pid = self.corpus[i]["id"]
        if pid not in self._chunks:
            self._chunks[pid] = read_pdf(pid, self.pdfs)
        return self._chunks[pid]

    def _assess(self, domain, p, sec, text):
        """Overlap/relevance emerge from the agent's own LONG-TERM MEMORY, not an imposed check. Three natural
        beats: the expert (1) works through the material and drafts what stands out, (2) RECALLS from its
        long-term memory (retrieval over everything it has indexed) what it already holds on those topics, and
        (3) recognizes, from that recollection, which drafts genuinely add something new. Recall is the memory
        act; the skipping is the agent's own recognition."""
        e = self.experts.get(domain)
        persona = e["identity"] if e else f"a newcomer studying {domain}"
        o = extract_json(self.llm([{"role": "user", "content":
            f"You are {persona}. New material has crossed your desk:\n{text}\n\nWork through it and draft the "
            'index entries that stand out to you. ONLY JSON: {"notes": ["<term> :: <gloss>", ...]}'}], max_tokens=300))
        draft = [x for x in (o.get("notes") or []) if isinstance(x, str)][:6]
        if not draft:
            return True, [], ""
        recalled = []                                                  # RECALL from long-term memory (by meaning)
        for note in draft:
            recalled += self._recall(note, k=2)
        recalled = list(dict.fromkeys(recalled))[:12]
        if not recalled:
            return True, draft, "nothing to recall"
        o2 = extract_json(self.llm([{"role": "user", "content":
            f"You are {persona}. You just drafted these notes:\n- " + "\n- ".join(draft) +
            "\n\nBut recalling what you ALREADY know:\n- " + "\n- ".join(recalled) +
            "\n\nWhich of your drafts genuinely add something BEYOND what you already recall? Keep only those "
            '(it is fine to keep none). ONLY JSON: {"keep": ["<the kept draft lines>"]}'}], max_tokens=240))
        keep = [x for x in (o2.get("keep") or []) if isinstance(x, str)]
        return True, keep, f"{max(0, len(draft) - len(keep))} recalled as known"

    def _grow_expert(self, domain, entries, pid):
        e = self.experts.setdefault(domain, {"identity": f"an emerging {domain} specialist", "concepts": [],
                                             "papers": [], "sections": 0})
        e["concepts"] = (e["concepts"] + [x.split("::")[0].strip() for x in entries])[-40:]
        if pid not in e["papers"]:
            e["papers"].append(pid)
        e["sections"] += 1

    def _evolve_expert(self, domain):
        e = self.experts[domain]
        nid = (self.llm([{"role": "user", "content":
            f"You have studied {e['sections']} sections across {len(e['papers'])} {domain} papers and mastered: "
            f"{', '.join(e['concepts'][-14:])}.\nIn ONE sentence, state your evolved specialist identity "
            "(specialty + signature expertise). Just the sentence."}], max_tokens=70) or "").strip().strip('"')
        if nid:
            e["identity"] = nid

    # ── the self-organizing, autoscaling work session ──
    def work(self, budget=24, min_a=2, max_a=10, load=3, soc=None):
        soc = soc or Society()
        self.stats = {"added": 0, "redundant": 0, "irrelevant": 0}   # this session's overlap/relevance filtering
        queue = []
        for i, p in enumerate(self.corpus):
            done = self.progress.get(p["id"], 0)
            for s in range(done, len(self._sections(i))):
                queue.append((i, s))
        queue = queue[:budget]
        if not queue:
            soc.broadcast("room", "no new sections — room is fully current.")
            return soc, [len(self.index)]
        workers = [f"W{k}" for k in range(min_a)]
        nid, qi, hist = min_a, 0, []
        while qi < len(queue):
            remaining = len(queue) - qi
            target = max(min_a, min(max_a, math.ceil(remaining / load)))
            while len(workers) < target:
                workers.append(f"W{nid}"); nid += 1
                soc.broadcast("mgr", f"SPAWN W{nid-1} (backlog {remaining})")
            for w in list(workers):
                if qi >= len(queue):
                    break
                i, s = queue[qi]; qi += 1
                p = self.corpus[i]
                rel, ents, overlap = self._assess(p["category"], p, s, self._sections(i)[s])
                self.progress[p["id"]] = max(self.progress.get(p["id"], 0), s + 1)   # section processed either way
                if not rel:
                    self.stats["irrelevant"] += 1
                    soc.broadcast(w, f"{p['id']} sec {s}: OFF-TOPIC, skipped")
                elif not ents:
                    self.stats["redundant"] += 1
                    soc.broadcast(w, f"{p['id']} sec {s}: overlaps existing knowledge ({overlap}), nothing new")
                else:
                    for e in ents:
                        self._add_entry(f"[{p['category']} · {p['id']} · sec {s}] {e}")
                    self.stats["added"] += len(ents)
                    self._grow_expert(p["category"], ents, p["id"])
                    soc.broadcast(w, f"{p['id']} sec {s} [{p['category']}]: +{len(ents)} new")
            if len(queue) - qi < len(workers) and len(workers) > min_a:
                for _ in range(len(workers) - max(min_a, len(queue) - qi)):
                    workers.pop(); soc.broadcast("mgr", "RETIRE (idle)")
            hist.append(len(workers))
        for d in {self.corpus[i]["category"] for i, _ in queue}:
            self._evolve_expert(d)
        self.save()
        return soc, hist
