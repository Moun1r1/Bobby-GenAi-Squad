import os
import re
from typing import Callable, Dict, List, Optional, Set

from .retrieval import EmbeddingRetriever, LexicalRetriever, embedding_available
from .dedup import near_dup

_LINK = re.compile(r"\[\[([^\]|#]+)")           # [[slug]] / [[slug|alias]] / [[slug#head]] → slug
_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "").strip().lower()).strip("-")
    return s or "note"


def link_id(l: str) -> str:
    """A link token → a note id. `vault/note` (cross-vault) keeps its vault prefix; `note` is same-vault."""
    l = (l or "").strip()
    if "/" in l:
        v, _, note = l.partition("/")
        return f"{slug(v)}/{slug(note)}"
    return slug(l)


class Note:
    __slots__ = ("id", "title", "tags", "source", "body", "links", "path")

    def __init__(self, id: str, title: str, body: str, tags=None, source: str = "",
                 links=None, path: str = ""):
        self.id = id
        self.title = title or id
        self.body = body or ""
        self.tags: List[str] = list(tags or [])
        self.source = source
        self.links: Set[str] = set(links or [])
        self.path = path

    def render(self) -> str:
        fm = [f"title: {self.title}"]
        if self.tags:   fm.append("tags: " + ", ".join(self.tags))
        if self.source: fm.append(f"source: {self.source}")
        if self.links:  fm.append("links: " + ", ".join(f"[[{l}]]" for l in sorted(self.links)))
        return "---\n" + "\n".join(fm) + "\n---\n\n" + self.body.rstrip() + "\n"


class KnowledgeVault:
    """A directory of linked markdown notes, indexed as a graph + a semantic entry index."""

    def __init__(self, root: str, embed_fn: Optional[Callable] = None, entry_head: int = 600,
                 cache: Optional[dict] = None):
        self.root = root
        self.entry_head = entry_head
        os.makedirs(root, exist_ok=True)
        self.notes: Dict[str, Note] = {}
        self.backlinks: Dict[str, Set[str]] = {}
        self._use_embed = embedding_available(embed_fn) if embed_fn is not None else embedding_available()
        # a SHARED embed cache (text→vec) → reloading a vault re-parses but re-embeds only NEW/changed note text
        self._ret = EmbeddingRetriever(embed_fn, cache=cache) if self._use_embed else LexicalRetriever()
        self._entry_of: Dict[str, str] = {}        # entry-text -> note id (map retriever hits back to notes)
        self._load()

    # -- parsing / indexing --------------------------------------------------
    def _parse(self, path: str) -> Optional[Note]:
        try:
            raw = open(path, encoding="utf-8").read()
        except Exception:
            return None
        title, tags, source, fm_links = "", [], "", []
        m = _FM.match(raw)
        body = raw[m.end():] if m else raw
        if m:
            for line in m.group(1).splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k, v = k.strip().lower(), v.strip()
                if k == "title":    title = v
                elif k == "tags":   tags = [t.strip() for t in v.split(",") if t.strip()]
                elif k == "source": source = v
                elif k == "links":  fm_links = _LINK.findall(v)
        nid = slug(title) if title else slug(os.path.splitext(os.path.basename(path))[0])
        links = {link_id(l) for l in fm_links} | {link_id(l) for l in _LINK.findall(body)}
        links.discard(nid)
        return Note(nid, title or nid, body, tags, source, links, path)

    def _entry_text(self, n: Note) -> str:
        return f"{n.title}. {' '.join((n.tags))}. {n.body[:self.entry_head]}"

    def _index_note(self, n: Note) -> None:
        self.notes[n.id] = n
        for l in n.links:
            self.backlinks.setdefault(l, set()).add(n.id)
        et = self._entry_text(n)
        self._entry_of[et] = n.id
        self._ret.add(et)

    def _load(self) -> None:
        for dp, _dn, fns in os.walk(self.root):
            for fn in fns:
                if fn.endswith(".md"):
                    n = self._parse(os.path.join(dp, fn))
                    if n:
                        self._index_note(n)

    # -- read / navigate -----------------------------------------------------
    def get(self, nid: str) -> Optional[Note]:
        return self.notes.get(slug(nid))

    def neighbors(self, nid: str) -> List[str]:
        n = self.notes.get(nid)
        out = list(n.links) if n else []
        out += [b for b in self.backlinks.get(nid, ()) if b not in out]
        return [x for x in out if x in self.notes]

    def search(self, query: str, k: int = 3) -> List[str]:
        ids, seen = [], set()
        for et in self._ret.search(query, k=k * 2):
            nid = self._entry_of.get(et)
            if nid and nid not in seen:
                seen.add(nid); ids.append(nid)
            if len(ids) >= k:
                break
        return ids

    def entry_vec(self, nid: str):
        """The stored embedding of a note's entry text (for a learned reranker to re-score). None in lexical mode."""
        n = self.notes.get(nid)
        if n is None or not self._use_embed:
            return None
        return getattr(self._ret, "cache", {}).get(self._entry_text(n))

    def navigate(self, query: str, k: int = 3, hops: int = 1, budget: int = 1800,
                 per_note: int = 520) -> str:
        """Semantic ENTRY + bounded link-hop expansion → an attributed markdown block (the local subgraph). The agent
        sees each note's excerpt AND its [[links]], so it can ask to traverse further. Empty string if the vault is
        empty / nothing matches."""
        entry = self.search(query, k=k)
        if not entry:
            return ""
        order: List[str] = list(entry)
        frontier = list(entry)
        for _ in range(max(0, hops)):
            nxt = []
            for nid in frontier:
                for nb in self.neighbors(nid):
                    if nb not in order:
                        order.append(nb); nxt.append(nb)
            frontier = nxt
        parts = ["# Knowledge vault — navigated for this step (apply it; enrich it as you learn)"]
        used = len(parts[0])
        for nid in order:
            n = self.notes.get(nid)
            if not n:
                continue
            head = f"\n\n## [[{n.id}]]" + (f"  ·  {n.source}" if n.source else "")
            links = ("\n→ links: " + " ".join(f"[[{l}]]" for l in sorted(n.links))) if n.links else ""
            block = head + "\n" + n.body.strip()[:per_note] + links
            if used + len(block) > budget:
                break
            parts.append(block); used += len(block)
        return "".join(parts)

    # -- write / enrich ------------------------------------------------------
    def _persist(self, n: Note) -> None:
        n.path = n.path or os.path.join(self.root, f"{n.id}.md")
        try:
            with open(n.path, "w", encoding="utf-8") as f:
                f.write(n.render())
        except Exception:
            pass

    def enrich(self, title: str, body: str, source: str = "", links=None, tags=None,
               dedup: bool = True) -> Optional[str]:
        """Add or extend a note from something learned. If the note exists, APPEND the new content unless it's a near
        duplicate of what's already there (so the graph grows without repeating). New notes are AUTO-LINKED to their
        nearest semantic neighbour so nothing is orphaned. Returns the note id (None if it was a pure duplicate)."""
        body = (body or "").strip()
        if not body:
            return None
        nid = slug(title)
        want_links = {link_id(l) for l in (links or [])}
        # auto-link a brand-new note to its nearest existing neighbour (semantic), so it joins the graph
        if nid not in self.notes and not want_links:
            near = [x for x in self.search(f"{title}. {body[:200]}", k=1) if x != nid]
            want_links |= set(near)
        existing = self.notes.get(nid)
        if existing:
            if dedup and near_dup(body, [existing.body]):
                existing.links |= want_links            # still strengthen the graph even if content is known
                self._persist(existing)
                return None
            existing.body = existing.body.rstrip() + f"\n\n## {source or 'enrichment'}\n{body}"
            existing.links |= want_links
            if tags: existing.tags = list(dict.fromkeys(existing.tags + list(tags)))
            n = existing
        else:
            n = Note(nid, title, body, tags, source, want_links)
        # de-index old entry text, re-index the note (graph + semantic entry)
        self._reindex(n)
        self._persist(n)
        return nid

    def link(self, a: str, b: str) -> None:
        a, b = slug(a), slug(b)
        n = self.notes.get(a)
        if n and b in self.notes and b != a:
            n.links.add(b); self.backlinks.setdefault(b, set()).add(a); self._persist(n)

    def ingest_file(self, path: str, source: str, title: Optional[str] = None, tags=None) -> Optional[str]:
        """Pull an EXTERNAL source (framework code, another pipeline's doc, the gemma-challenge repo…) into the vault
        as a note. Keeps provenance in `source`. Long files are summarised by head — the note points at the path."""
        try:
            text = open(path, encoding="utf-8").read()
        except Exception:
            return None
        title = title or os.path.splitext(os.path.basename(path))[0].replace("_", " ")
        body = f"_ingested from `{path}`_\n\n" + text[:4000]
        return self.enrich(title, body, source=source, tags=(tags or ["ingested"]))

    def ingest_dir(self, root: str, source: str, patterns=(".py", ".md"), max_files: int = 40,
                   skip=("__pycache__", ".git", "node_modules", "test", "example")) -> List[str]:
        """Ingest a REPO TREE (e.g. a local clone of google-deepmind/gemma) into the vault so its source is navigable —
        the way agents actually 'go further' into a foundation repo. Bounded (max_files, matched extensions, skip
        noise dirs); each file becomes a linked note keyed `{source}:{relpath}`. Returns the ingested note ids."""
        if not os.path.isdir(root):
            return []
        out: List[str] = []
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if not any(s in d.lower() for s in skip)]
            for fn in sorted(fns):
                if len(out) >= max_files:
                    return out
                if not fn.endswith(tuple(patterns)):
                    continue
                rel = os.path.relpath(os.path.join(dp, fn), root)
                nid = self.ingest_file(os.path.join(dp, fn), source=f"{source}:{rel}",
                                       title=f"{source} {rel}")
                if nid:
                    out.append(nid)
        return out

    def _reindex(self, n: Note) -> None:
        self.notes[n.id] = n
        for l in n.links:
            self.backlinks.setdefault(l, set()).add(n.id)
        et = self._entry_text(n)
        if et not in self._entry_of:                    # add fresh entry text (old one is left inert — cheap)
            self._entry_of[et] = n.id
            self._ret.add(et)

    def harvest_dpo(self) -> List[dict]:
        """Turn every note's `## dpo` block into preference pairs {prompt, chosen, rejected} — the curated bad→good
        corrections the self-DPO flywheel trains on (each capability's KNOWN anti-pattern is a ready `rejected`).
        Format inside a `## dpo` section: repeating lines `- prompt:` / `- chosen:` / `- rejected:`."""
        pairs: List[dict] = []
        for n in self.notes.values():
            low = n.body.lower()
            i = low.find("\n## dpo")
            if i < 0:
                continue
            section = n.body[i + 1:]
            end = section.find("\n## ", 3)               # stop at the next heading
            if end > 0:
                section = section[:end]
            cur: Dict[str, str] = {}
            for line in section.splitlines():
                s = line.strip()
                for key in ("prompt", "chosen", "rejected"):
                    pre = f"- {key}:"
                    if s.lower().startswith(pre):
                        cur[key] = s[len(pre):].strip()
                if {"prompt", "chosen", "rejected"} <= cur.keys():
                    pairs.append({"prompt": cur["prompt"], "chosen": cur["chosen"], "rejected": cur["rejected"],
                                  "source": n.id})
                    cur = {}
        return pairs

    def stats(self) -> dict:
        edges = sum(len(n.links) for n in self.notes.values())
        return {"notes": len(self.notes), "edges": edges, "embed": self._use_embed,
                "sources": sorted({n.source.split(":")[0] for n in self.notes.values() if n.source})}


class VaultHub:
    """MANY vaults, cross-linked and reused dynamically. Each subdirectory of `root` is a KnowledgeVault; a note may
    link to another vault's note with `[[vault/note]]`. The self-learning loop navigates ACROSS all vaults, enriches
    into any of them, and can CREATE new ones on the fly — so knowledge is organized by domain yet reused everywhere.
    Single-vault callers keep using KnowledgeVault; the hub is the multi-vault layer over it."""

    def __init__(self, root: str, embed_fn: Optional[Callable] = None, default: str = "foundation",
                 retriever_path: Optional[str] = None):
        self.root = root
        self.embed_fn = embed_fn
        self.default = slug(default)
        os.makedirs(root, exist_ok=True)
        self.vaults: Dict[str, KnowledgeVault] = {}
        self._embed_cache: dict = {}                        # shared text→vec cache; reloads reuse it (no re-embed)
        self._sigs: Dict[str, tuple] = {}                  # per-vault dir signature (files+mtimes) for hot-reload
        self._last_check = 0.0
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                self.vaults[name] = KnowledgeVault(p, embed_fn, cache=self._embed_cache)
                self._sigs[name] = self._dir_sig(p)
        if not self.vaults:
            self.create(self.default)
        # LEARNED RECALL — reuse the trained RetrievalEncoder (torch-free) to re-rank entry, replacing plain cosine
        # in EVERY pipeline. None → cosine. Default path lives next to the vaults.
        self.retriever_path = retriever_path or os.path.join(os.path.dirname(root), "encoders", "retrieval.npz")
        self.retriever = None
        self._query_cache: Dict[str, list] = {}
        self.reload_retriever()

    def reload_retriever(self) -> bool:
        """(Re)load the exported retrieval encoder — call after training so the learned recall is reused immediately."""
        try:
            from .learned_retriever import load_retriever
            self.retriever = load_retriever(self.retriever_path)
        except Exception:
            self.retriever = None
        return self.retriever is not None

    def _query_vec(self, query: str):
        if query in self._query_cache:
            return self._query_cache[query]
        v = None
        try:
            from .retrieval import default_embed
            out = (self.embed_fn or default_embed)(["search_query: " + query])
            v = out[0] if out else None
        except Exception:
            v = None
        self._query_cache[query] = v
        return v

    def _entries(self, query: str, per_vault_k: int = 2):
        """Entry notes across all vaults. With a trained retriever loaded, RE-RANK the cosine candidates with the
        learned scorer (learned recall); otherwise plain cosine. Returns [(vault, nid)]."""
        self.maybe_reload()                                # pick up hand-edited / new notes without a restart
        cands = []
        for vn, v in self.vaults.items():
            for nid in v.search(query, k=per_vault_k * 2):
                cands.append((vn, nid))
        if self.retriever is not None:
            qv = self._query_vec(query)
            if qv is not None:
                scored = []
                for vn, nid in cands:
                    ev = self.vaults[vn].entry_vec(nid)
                    if ev is not None:
                        scored.append((float(self.retriever.score(qv, [ev])[0]), vn, nid))
                if scored:
                    scored.sort(reverse=True)
                    return [(vn, nid) for _s, vn, nid in scored]
        return cands

    def names(self) -> List[str]:
        return list(self.vaults)

    def create(self, name: str) -> KnowledgeVault:
        name = slug(name)
        if name not in self.vaults:
            p = os.path.join(self.root, name)
            self.vaults[name] = KnowledgeVault(p, self.embed_fn, cache=self._embed_cache)
            self._sigs[name] = self._dir_sig(p)
        return self.vaults[name]

    def _dir_sig(self, path: str) -> tuple:
        """A cheap signature of a vault dir: sorted (filename, mtime) for its .md files. Changes iff notes are
        added / removed / edited — the hot-reload trigger."""
        try:
            return tuple(sorted((e.name, int(e.stat().st_mtime))
                                for e in os.scandir(path) if e.is_file() and e.name.endswith(".md")))
        except Exception:
            return ()

    def _touch(self, name: str) -> None:
        """Refresh a vault's signature after a WRITE-THROUGH (enrich/link/ingest), so our own writes — already applied
        in memory — don't trigger a needless reload."""
        v = self.vaults.get(slug(name))
        if v:
            self._sigs[slug(name)] = self._dir_sig(v.root)

    def reload(self) -> dict:
        """HOT-RELOAD: re-scan the vaults root — add new vaults, reload CHANGED ones (by dir signature), drop removed.
        Unchanged vaults are left intact; the shared embed cache means a reload re-parses but re-embeds only new text.
        Makes hand-edited / externally-added notes go live WITHOUT a backend restart."""
        changed = []
        seen = set()
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name)
            if not os.path.isdir(p):
                continue
            seen.add(name)
            sig = self._dir_sig(p)
            if name not in self.vaults or self._sigs.get(name) != sig:
                self.vaults[name] = KnowledgeVault(p, self.embed_fn, cache=self._embed_cache)
                self._sigs[name] = sig
                changed.append(name)
        for name in list(self.vaults):                     # a vault dir was deleted
            if name not in seen:
                del self.vaults[name]; self._sigs.pop(name, None); changed.append(f"-{name}")
        if changed:
            self._query_cache.clear()
            self.reload_retriever()
        return {"changed": changed, "vaults": list(self.vaults)}

    def maybe_reload(self, throttle: float = 2.0) -> None:
        """Throttled auto hot-reload: at most one dir-scan every `throttle` seconds, so per-step recall stays cheap
        but disk edits appear within ~throttle seconds."""
        import time as _time
        now = _time.time()
        if now - self._last_check < throttle:
            return
        self._last_check = now
        cur = {}
        for name in os.listdir(self.root):
            p = os.path.join(self.root, name)
            if os.path.isdir(p):
                cur[name] = self._dir_sig(p)
        if cur != self._sigs:                              # something added / removed / edited on disk
            self.reload()

    def vault(self, name: str) -> KnowledgeVault:
        return self.vaults.get(slug(name)) or self.create(name)

    def _resolve(self, cur_vault: str, link: str):
        """A link token (in `cur_vault`) → (vault, note_id). Bare tokens stay in the current vault."""
        if "/" in link:
            v, _, note = link.partition("/")
            return slug(v), slug(note)
        return slug(cur_vault), slug(link)

    def _qualify(self, cur_vault: str, link: str) -> str:
        v, n = self._resolve(cur_vault, link)
        return f"{v}/{n}"

    def navigate(self, query: str, per_vault_k: int = 2, hops: int = 1, budget: int = 2200,
                 per_note: int = 460, whole_vault: Optional[str] = None, whole_k: int = 0) -> str:
        """Semantic entry across EVERY vault + cross-vault link-hop → one attributed block. The step sees the local
        subgraph spanning vaults, with `[[vault/note]]` links it can traverse or enrich.

        NEEDLE-PRESERVING recall (whole_vault/whole_k, off by default → byte-identical to before): the top-`whole_k`
        entry notes of `whole_vault` are returned WHOLE (untruncated by per_note) FIRST, before the summarised
        subgraph. Proven WIRE +51.7 F1 on exceeds-window extractive retrieval (wiki/proofs/probe_generation.py): a
        summarised/truncated subgraph drops a single needle; the verbatim passage keeps it. The graph walk still runs
        for multi-hop bridging — this unions verbatim-needle + graph-hop."""
        parts = ["# Knowledge vaults — navigated across all vaults for this step (apply it; enrich or create as you learn)"]
        used = len(parts[0])
        included: set = set()
        # ── needle-preserving: top-k WHOLE notes from a priority vault (verbatim), before anything is truncated ──
        wv = self.vaults.get(whole_vault) if whole_vault else None
        if wv and whole_k > 0:
            for nid in wv.search(query, k=whole_k):
                n = wv.notes.get(nid)
                if not n or (whole_vault, nid) in included:
                    continue
                block = (f"\n\n## [[{whole_vault}/{n.id}]] (verbatim)" + (f"  ·  {n.source}" if n.source else "")
                         + "\n" + n.body.strip())
                if used + len(block) > budget:
                    break
                parts.append(block); used += len(block); included.add((whole_vault, nid))
        # ── semantic entry + cross-vault link-hop (the summarised subgraph, for multi-hop bridging) ──
        order: List[tuple] = []
        for vn, nid in self._entries(query, per_vault_k=per_vault_k)[:per_vault_k * len(self.vaults)]:
            if (vn, nid) not in order:
                order.append((vn, nid))
        frontier = list(order)
        for _ in range(max(0, hops)):
            nxt = []
            for (vn, nid) in frontier:
                v = self.vaults.get(vn)
                n = v.notes.get(nid) if v else None
                if not n:
                    continue
                for l in n.links:
                    tv, tn = self._resolve(vn, l)
                    if tv in self.vaults and tn in self.vaults[tv].notes and (tv, tn) not in order:
                        order.append((tv, tn)); nxt.append((tv, tn))
            frontier = nxt
        for (vn, nid) in order:
            if (vn, nid) in included:
                continue
            n = self.vaults[vn].notes.get(nid)
            if not n:
                continue
            head = f"\n\n## [[{vn}/{n.id}]]" + (f"  ·  {n.source}" if n.source else "")
            links = ("\n→ links: " + " ".join(f"[[{self._qualify(vn, l)}]]" for l in sorted(n.links))) if n.links else ""
            block = head + "\n" + n.body.strip()[:per_note] + links
            if used + len(block) > budget:
                break
            parts.append(block); used += len(block)
        return "".join(parts) if len(parts) > 1 else ""

    def enrich(self, vault: str, title: str, body: str, source: str = "", links=None, tags=None) -> Optional[str]:
        """Write a note into `vault` (created on demand). `links` may include `[[vault/note]]` cross-vault refs."""
        nid = self.vault(vault).enrich(title, body, source=source, links=links, tags=tags)
        self._touch(vault)                                 # our own write is already in memory — don't reload for it
        return nid

    def link(self, src: str, dst: str) -> None:
        """Add a (possibly cross-vault) edge src → dst, both `vault/note`."""
        sv, sn = self._resolve(self.default, src)
        v = self.vaults.get(sv)
        if v and sn in v.notes:
            v.notes[sn].links.add(self._qualify(sv, dst))
            v._persist(v.notes[sn])
            self._touch(sv)

    def ingest_dir(self, vault: str, path: str, source: str, **kw) -> List[str]:
        r = self.vault(vault).ingest_dir(path, source, **kw)
        self._touch(vault)
        return r

    def ingest_file(self, vault: str, path: str, source: str, **kw) -> Optional[str]:
        r = self.vault(vault).ingest_file(path, source, **kw)
        self._touch(vault)
        return r

    def harvest_dpo(self) -> List[dict]:
        out = []
        for vn, v in self.vaults.items():
            for p in v.harvest_dpo():
                out.append({**p, "vault": vn})
        return out

    def note(self, qualified: str) -> Optional[dict]:
        """One note across vaults, with cross-vault neighbours (links ∪ backlinks from any vault)."""
        if "/" not in qualified:
            for vn, v in self.vaults.items():
                if slug(qualified) in v.notes:
                    qualified = f"{vn}/{slug(qualified)}"; break
        vn, _, nid = qualified.partition("/")
        vn = slug(vn)
        v = self.vaults.get(vn)
        n = v.get(nid) if v else None
        if not n:
            return None
        nbrs = [self._qualify(vn, l) for l in sorted(n.links)]
        for ovn, ov in self.vaults.items():                       # cross-vault backlinks
            for on in ov.notes.values():
                for l in on.links:
                    if self._resolve(ovn, l) == (vn, n.id):
                        b = f"{ovn}/{on.id}"
                        if b not in nbrs:
                            nbrs.append(b)
        return {"id": f"{vn}/{n.id}", "vault": vn, "title": n.title, "tags": n.tags, "source": n.source,
                "body": n.body, "links": [x for x in nbrs if x != f"{vn}/{n.id}"]}

    def graph(self) -> dict:
        nodes, ids = [], set()
        for vn, v in self.vaults.items():
            for n in v.notes.values():
                nid = f"{vn}/{n.id}"
                ids.add(nid)
                nodes.append({"id": nid, "vault": vn, "title": n.title, "tags": n.tags, "source": n.source,
                              "chars": len(n.body), "links": [self._qualify(vn, l) for l in sorted(n.links)]})
        edges = [{"source": nd["id"], "target": l} for nd in nodes for l in nd["links"] if l in ids]
        return {"nodes": nodes, "edges": edges, "stats": self.stats()}

    def search(self, query: str, k: int = 3) -> List[str]:
        return [f"{vn}/{nid}" for vn, nid in self._entries(query, per_vault_k=k)[:k * 2]]

    def stats(self) -> dict:
        per = {vn: v.stats() for vn, v in self.vaults.items()}
        return {"names": list(self.vaults), "vaults": per,
                "notes": sum(s["notes"] for s in per.values()),
                "edges": sum(s["edges"] for s in per.values()),
                "embed": any(s["embed"] for s in per.values()),
                "recall": ("learned" if self.retriever is not None else "cosine")}
