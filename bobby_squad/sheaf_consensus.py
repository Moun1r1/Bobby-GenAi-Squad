from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np


def _norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ── build the shared edge space (the sheaf overlap): cluster items across agents ───────────────────
def _cluster(proposals: Sequence[Iterable[str]], embed: Optional[Callable[[str], Sequence[float]]],
             merge_tau: float):
    """Cluster proposed items across agents into candidate facts.

    Returns (reps, support): `reps[c]` is a representative string for candidate c;
    `support` is an [n_agents, n_candidates] {0,1} matrix — the per-agent local
    view over the shared candidate space (support[i,c]=1 iff agent i proposed an
    item that fell in cluster c).
    """
    items: List[str] = []
    owners: List[int] = []
    for i, ps in enumerate(proposals):
        for s in ps:
            s = (s or "").strip()
            if s:
                items.append(s)
                owners.append(i)
    if not items:
        return [], np.zeros((len(proposals), 0))

    reps: List[str] = []
    rep_vecs: List[np.ndarray] = []
    assign: List[int] = []
    vecs = [_norm(embed(s)) for s in items] if embed is not None else []

    for k, s in enumerate(items):
        cid = -1
        if embed is not None:
            best, bj = merge_tau, -1
            for j, rv in enumerate(rep_vecs):
                cos = float(np.dot(vecs[k], rv))
                if cos >= best:
                    best, bj = cos, j
            cid = bj
        else:
            low = s.lower()
            cid = next((j for j, r in enumerate(reps) if r.lower() == low), -1)
        if cid == -1:
            cid = len(reps)
            reps.append(s)
            if embed is not None:
                rep_vecs.append(vecs[k])
        assign.append(cid)

    support = np.zeros((len(proposals), len(reps)))
    for k, cid in enumerate(assign):
        support[owners[k], cid] = 1.0
    return reps, support


# ── the ADMM consensus core (scaled-dual, a port of admm.run_admm) ────────────────────────────────
@dataclass
class ConsensusResult:
    accepted: List[str]
    mode: str = "consensus"                                          # "consensus" or "union" (regime fallback)
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))       # consensus inclusion per candidate
    support_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    candidates: List[str] = field(default_factory=list)
    redundancy: float = 0.0                                          # fraction of candidates >=2 agents cover
    primal_res: float = 0.0                                          # ||x - z|| at convergence
    consistency_rms: float = 0.0                                     # residual sheaf disagreement
    iters: int = 0


def _consensus_admm(support: np.ndarray, rho: float, iters: int, alpha: float):
    """Scaled-dual ADMM over candidate inclusion. `support` is [N,C] in {0,1}.
    Returns (z, primal_res, consistency_rms). Only agents that had a view vote on
    a candidate (the sheaf restriction: an agent that never saw the content does
    not vote it down)."""
    n, c = support.shape
    had_view = (support.sum(axis=1, keepdims=True) > 0).astype(float)
    part = np.repeat(had_view, c, axis=1)
    part_count = np.clip(part.sum(axis=0), 1.0, None)

    x = support.copy()
    z = support.mean(axis=0, keepdims=True).repeat(n, axis=0)
    y = np.zeros_like(support)
    w = 1.0 / (1.0 + 1.0 / rho)                                      # prox_f mix weight (pull toward own vote)
    primal = 0.0
    for _ in range(iters):
        z_prev = z
        x = np.clip(w * (z - y) + (1.0 - w) * support, 0.0, 1.0)     # local x-update
        x_relax = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
        zc = np.clip(((x_relax + y) * part).sum(axis=0) / part_count, 0.0, 1.0)   # sheaf z-update
        z = np.repeat(zc[None, :], n, axis=0)
        y = y + part * (x_relax - z)                                 # dual ascent
        primal = float(np.linalg.norm((x_relax - z) * part))
    zc = z[0]
    consistency = float(np.sqrt(np.mean((support - zc[None, :]) ** 2 * part) + 1e-12))
    return zc, primal, consistency


def sheaf_consensus(proposals: Sequence[Iterable[str]],
                    embed: Optional[Callable[[str], Sequence[float]]] = None,
                    merge_tau: float = 0.88, accept_tau: Optional[float] = None,
                    conditional: bool = True, min_overlap: float = 0.34,
                    rho: float = 1.0, iters: int = 16, alpha: float = 1.0) -> ConsensusResult:
    """Reconcile N agents' proposed item-sets into a consensus set.

    proposals  : list (per agent) of iterables of item strings.
    embed      : str→vec for semantic matching (None ⇒ exact string match).
    merge_tau  : cosine threshold to treat two items as the same fact.
    accept_tau : consensus threshold to keep a fact. Default = strict majority of
                 the agents that had a view (filters singletons).
    conditional: if True (default) fall back to plain union when agents partition
                 the work (redundancy < `min_overlap`), so it never prunes
                 disjoint coverage. Set False to always run consensus.
    min_overlap: minimum fraction of candidates that >=2 agents must cover for
                 consensus to engage.
    """
    reps, support = _cluster(proposals, embed, merge_tau)
    union = list(reps)
    n_agents = len(proposals)
    if len(reps) == 0 or n_agents < 2:
        return ConsensusResult(accepted=union, mode="union", candidates=reps)

    covered = support.sum(axis=0)                                    # agents per candidate
    redundancy = float(np.mean(covered >= 2)) if len(reps) else 0.0
    if conditional and redundancy < min_overlap:                    # agents partition the work → union is correct
        return ConsensusResult(accepted=union, mode="union", candidates=reps, redundancy=redundancy,
                               support_frac=covered / n_agents)

    n_view = int((support.sum(axis=1) > 0).sum())
    if accept_tau is None:
        accept_tau = (math.floor(n_view / 2) + 1) / max(n_view, 1) - 1e-9
    zc, primal, consistency = _consensus_admm(support, rho, iters, alpha)
    accepted = [reps[i] for i in range(len(reps)) if zc[i] >= accept_tau]
    return ConsensusResult(accepted=accepted, mode="consensus", z=zc,
                           support_frac=(support.sum(axis=0) / max(n_view, 1)), candidates=reps,
                           redundancy=redundancy, primal_res=primal, consistency_rms=consistency, iters=iters)


# ── drop-in squad harvest ─────────────────────────────────────────────────────────────────────────
def make_consensus_harvest(embed=None, min_agents_before_consensus: int = 2, **kw):
    """Return a `harvest(result, acc)` for `squad_solve` that buffers per-agent
    results and, once >= `min_agents_before_consensus` have accrued, keeps the
    sheaf-consensus set. While the buffer is small (or agents partition the work),
    it returns the plain union, so it is a safe superset of the default harvest.

    Example:
        from bobby_squad import squad_solve, make_consensus_harvest
        from bobby_squad.retrieval import default_embed
        squad_solve(agents, units, work,
                    harvest=make_consensus_harvest(embed=default_embed))
    """
    buffer: List[List[str]] = []

    def harvest(result, acc):
        buffer.append(list(result or []))
        if len(buffer) < max(2, min_agents_before_consensus):
            union = set()
            for b in buffer:
                union |= set(b)
            return union
        return set(sheaf_consensus(buffer, embed=embed, **kw).accepted)

    return harvest
