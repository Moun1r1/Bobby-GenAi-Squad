from typing import Callable, List, Optional, Tuple


# a guard is: None (unconditional) · a predicate ctx->bool (deterministic) · ("llm", prompt_template) (LLM boolean)
Guard = object


class FSM:
    def __init__(self, initial: str):
        self.initial = initial
        self.transitions: dict = {}                  # src -> [(dst, guard)]

    def add(self, src: str, dst: str, guard: Optional[Guard] = None) -> "FSM":
        self.transitions.setdefault(src, []).append((dst, guard))
        return self

    def is_valid(self, src: str, dst: str) -> bool:
        return any(d == dst for d, _ in self.transitions.get(src, []))

    def edges(self, src: str) -> List[str]:
        return [d for d, _ in self.transitions.get(src, [])]

    def next(self, state: str, ctx: Optional[dict] = None, llm: Optional[Callable] = None,
             max_tokens: int = 8) -> Tuple[Optional[str], str]:
        """Return (next_state, how). `how` ∈ {deterministic, llm, stuck}. Deterministic guards are tried first (free);
        an LLM boolean guard is used only if nothing deterministic applies and `llm` is provided."""
        ctx = ctx or {}
        edges = self.transitions.get(state, [])
        for dst, g in edges:                         # 1) deterministic predicate guards — no LLM
            if callable(g) and not isinstance(g, tuple):
                try:
                    if g(ctx):
                        return dst, "deterministic"
                except Exception:
                    pass
        if len(edges) == 1 and edges[0][1] is None:
            return edges[0][0], "deterministic"      # 2) a sole unconditional edge (never shadows a guarded one)
        for dst, g in edges:                         # 3) LLM boolean guard — only on ambiguity
            if isinstance(g, tuple) and g and g[0] == "llm" and llm is not None:
                prompt = g[1].format(**ctx) + "\nReply with only YES or NO."
                ans = (llm([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0) or "")
                if ans.strip().upper().startswith("YES"):
                    return dst, "llm"
        return None, "stuck"


def cluster_match(frozen_fn: Callable[[dict], str], llm_fn: Callable[[dict], str], held_out: List[dict],
                  scorer: Callable[[str, str], float], threshold: float = 0.8) -> dict:
    """Seam-2 distillation gate: a frozen plugin may replace the LLM for a query cluster ONLY if it PROVABLY matches the
    LLM's answers on a HELD-OUT slice of that cluster. `held_out` items are payloads; `scorer(a, b) -> [0,1]` measures
    answer agreement. Returns {match, agreement, n}; promote only when `match` is True. Prevents over-eager distillation
    (freezing a plugin that merely fired the same transition but behaves differently)."""
    if not held_out:
        return {"match": False, "agreement": 0.0, "n": 0, "reason": "no held-out"}
    agree = 0.0
    for payload in held_out:
        a = frozen_fn(payload)
        b = llm_fn(payload)
        agree += 1.0 if scorer(a, b) >= 0.5 else 0.0
    agreement = agree / len(held_out)
    return {"match": agreement >= threshold, "agreement": round(agreement, 3), "n": len(held_out)}
