from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .router import OODGate


# ── cross-run plugin persistence ──────────────────────────────────────────────────────────────────
def _handler_recipe(plugin) -> Optional[dict]:
    """Extract the reconstruction recipe from a frozen burn-in handler via the
    markers `make_*` set (`_pattern`/`_op`/`_transform`/`_src`), else None."""
    h = plugin.handler
    for attr, rtype in (("_pattern", "regex"), ("_op", "math"), ("_transform", "transform"), ("_src", "code")):
        if hasattr(h, attr):
            return {"type": rtype, "value": getattr(h, attr)}
    # image grid-counter and any marker-less deterministic handler: not reconstructable by recipe
    return None


def _rebuild_handler(recipe: dict) -> Optional[Callable]:
    from . import burn_in as B
    t, v = recipe.get("type"), recipe.get("value")
    if not isinstance(v, str):
        return None
    if t == "regex":
        return B.make_extractor(v)
    if t == "math":
        return B.make_aggregator(v)
    if t == "transform":
        return B.make_transform(v)
    if t == "code":
        return B.make_codeplugin(v)
    return None


@dataclass
class PluginStore:
    """Persist proof-gated frozen plugins across runs. JSON on disk at `path`."""
    path: str
    records: List[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "PluginStore":
        recs = []
        if os.path.exists(path):
            recs = json.load(open(path)).get("plugins", [])
        return cls(path=path, records=recs)

    def snapshot(self, engine, provenance: str = "") -> int:
        """Serialize every reconstructable frozen plugin in `engine.registry`.
        Returns the number newly captured (skips functional twins already stored)."""
        seen = {(r["recipe"]["type"], str(r["recipe"]["value"])) for r in self.records}
        added = 0
        for p in engine.registry.active():
            recipe = _handler_recipe(p)
            if recipe is None:
                continue
            key = (recipe["type"], str(recipe["value"]))
            if key in seen:
                continue
            proof = p.proof if isinstance(p.proof, dict) else {}
            gate = proof.get("competence")
            self.records.append({
                "name": p.name,
                "tags": list(p.tags),
                "kind": p.kind,
                "recipe": recipe,
                "proof": {"verdict": proof.get("verdict", "WIRE"), "kind": proof.get("kind"),
                          "hypothesis": proof.get("hypothesis"), "score": proof.get("score"),
                          "competence": _gate_to_json(gate)},
                "provenance": provenance or getattr(p, "provenance", ""),
            })
            seen.add(key)
            added += 1
        return added

    def save(self) -> str:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        json.dump({"plugins": self.records}, open(self.path, "w"), indent=2)
        return self.path

    def restore(self, engine) -> int:
        """Rehydrate stored plugins into `engine` (via `Engine.promote`). Returns
        the number successfully promoted. Twins already present are rejected by the
        registry's AST fingerprint, exactly as during a live run."""
        n = 0
        for r in self.records:
            handler = _rebuild_handler(r["recipe"])
            if handler is None:
                continue
            gate = _gate_from_json(r["proof"].get("competence"))
            proof = dict(r["proof"])
            proof["competence"] = gate
            code = r["recipe"]["value"] if r["recipe"]["type"] == "code" else None
            if engine.promote(r["name"], handler, tags=r["tags"], proof=proof, kind=r.get("kind", "static"),
                              code=code, provenance="soma:" + (r.get("provenance") or "")):
                n += 1
        return n

    def covered_caps(self) -> set:
        caps = set()
        for r in self.records:
            caps.update(r.get("tags", []))
        return caps


def _gate_to_json(gate) -> Optional[dict]:
    if isinstance(gate, OODGate):
        return {"mu": list(gate.mu), "inv_std": list(gate.inv_std), "tau": gate.tau}
    return None


def _gate_from_json(d) -> Optional[OODGate]:
    if isinstance(d, dict) and "mu" in d:
        return OODGate(mu=d["mu"], inv_std=d["inv_std"], tau=d["tau"])
    return None


# ── verified training-corpus emitter ──────────────────────────────────────────────────────────────
@dataclass
class DistillationCorpus:
    """Collect verified (input → output) traces into an SFT/DPO training corpus.

    A record is *verified* iff (a) it was served by a proof-gated frozen plugin
    (deterministic, gate-checked), or (b) it was solved by the model and graded
    correct. Only verified records are emitted, so every label is trustworthy.
    """
    records: List[dict] = field(default_factory=list)
    _seen: set = field(default_factory=set)

    def record(self, *, input: str, output: str, capability: str, source: str,
               correct: Optional[bool] = None, prompt: Optional[str] = None) -> bool:
        """Add one trace. `source` ∈ {"plugin","model"}. For source="model" pass
        `correct`; the record is kept only if correct is True. Plugin-served traces
        are always verified. Dedups on (input, output). Returns True if kept."""
        if source == "model" and not correct:
            return False
        out = (output or "").strip()
        if not out:
            return False
        key = (source, (input or "").strip(), out)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.records.append({"input": (input or "").strip(), "output": out, "capability": capability,
                             "source": source, "prompt": (prompt or "").strip() or None})
        return True

    def collect_run(self, tickets: List[dict], signals_rows: List[dict]) -> int:
        """Join a burn-in's tickets (which carry input/blob/gold) with its Signals
        rows (route, correct) by index, emitting verified records. Uses the ticket
        gold as the label for correct model traces (== the model's output when
        correct under set-equality), and for plugin routes too (deterministic)."""
        kept = 0
        for i, row in enumerate(signals_rows):
            if i >= len(tickets):
                break
            t = tickets[i]
            gold = t.get("gold")
            label = "\n".join(gold) if isinstance(gold, (list, tuple)) else str(gold)
            route = row.get("route")
            src = "plugin" if route == "frozen" else "model"
            if self.record(input=t.get("blob", ""), output=label, capability=t.get("cap", t.get("kind", "")),
                           source=src, correct=bool(row.get("correct")), prompt=t.get("prompt")):
                kept += 1
        return kept

    def emit_sft(self, path: str, style: str = "messages") -> int:
        """Write an SFT `.jsonl`. style="messages" → OpenAI chat format
        {"messages":[user,assistant]}; style="text" → {"prompt","completion"}."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n = 0
        with open(path, "w") as f:
            for r in self.records:
                user = r.get("prompt") or r["input"]
                if style == "messages":
                    obj = {"messages": [{"role": "user", "content": user},
                                        {"role": "assistant", "content": r["output"]}],
                           "capability": r["capability"], "source": r["source"]}
                else:
                    obj = {"prompt": user, "completion": r["output"],
                           "capability": r["capability"], "source": r["source"]}
                f.write(json.dumps(obj) + "\n")
                n += 1
        return n

    def emit_dpo(self, path: str, pairs: List[dict]) -> int:
        """Serialize DPO preference pairs {prompt, chosen, rejected} (e.g. from
        `encoders.trajectory_dpo` / `vault.harvest_dpo`) to `.jsonl`."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for p in pairs:
                f.write(json.dumps({"prompt": p.get("prompt", ""), "chosen": p.get("chosen", ""),
                                    "rejected": p.get("rejected", "")}) + "\n")
        return len(pairs)

    def coverage(self) -> Dict[str, int]:
        cov: Dict[str, int] = {}
        for r in self.records:
            cov[r["capability"]] = cov.get(r["capability"], 0) + 1
        return cov

    def stats(self) -> dict:
        by_src = {"plugin": 0, "model": 0}
        for r in self.records:
            by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        return {"n": len(self.records), "by_source": by_src, "coverage": self.coverage()}
