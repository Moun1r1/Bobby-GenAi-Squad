import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. VALUE / PREFERENCE HEAD — the learned critic (serves long-horizon-improvement / self-DPO) ─────────────────
class ValueHead(nn.Module):
    """Pooled LM hidden (or response embedding) of (prompt+response) → a scalar quality, CONDITIONED on the world
    state (`d_cond`): "how good is this direction, given where we are". Trained by ranking chosen above rejected —
    a cheap deterministic critic the flywheel runs unattended. d_cond=0 → unconditional (standalone) critic."""

    def __init__(self, d_model: int, d_cond: int = 0, hidden: int = 512):
        super().__init__()
        self.d_cond = d_cond
        self.net = nn.Sequential(nn.LayerNorm(d_model + d_cond), nn.Linear(d_model + d_cond, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, feats: torch.Tensor, world: torch.Tensor = None) -> torch.Tensor:   # feats [B,d_model], world [B,d_cond]
        if self.d_cond and world is not None:
            feats = torch.cat([feats, world], dim=-1)                  # condition on world state
        return self.net(feats).squeeze(-1)                            # [B] quality

    @staticmethod
    def ranking_loss(v_chosen: torch.Tensor, v_rejected: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        return F.relu(margin - (v_chosen - v_rejected)).mean()          # chosen should outscore rejected by `margin`


# ── 2. RETRIEVAL / MEMORY ENCODER — learned recall (serves memory-selection; upgrades cosine entry) ──────────────
class RetrievalEncoder(nn.Module):
    """A bi-encoder over item vectors (nomic embeddings of vault notes / memory). Learns to score the RELEVANT item
    above distractors for a query — a learned recall policy that beats raw cosine on held-out queries."""

    def __init__(self, d_in: int, d: int = 256):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.k = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, q_vec: torch.Tensor, cand_vecs: torch.Tensor) -> torch.Tensor:
        # q_vec [B, d_in], cand_vecs [B, N, d_in] → scores [B, N]
        qe = F.normalize(self.q(q_vec), dim=-1)
        ce = F.normalize(self.k(cand_vecs), dim=-1)
        return torch.einsum("bd,bnd->bn", qe, ce)

    @staticmethod
    def infonce(scores: torch.Tensor, pos_idx: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
        return F.cross_entropy(scores / temp, pos_idx)                  # positive is item pos_idx among candidates


# ── 3. TRAJECTORY SELF-MONITOR — learned metacognition (serves loops-system) ─────────────────────────────────────
class TrajectoryMonitor(nn.Module):
    """A tiny transformer over a sequence of event embeddings (move/tool/result) → the agent's regime
    {productive, looping, drifting, converged}. Trained on the deterministic behavior signals as labels, so a learned
    monitor generalizes past the threshold rules and can drive the loop's stop/replan decision."""

    N_CLASSES = 4  # 0 productive · 1 looping · 2 drifting · 3 converged

    def __init__(self, d_in: int, d: int = 128, heads: int = 4, layers: int = 2, d_cond: int = 0):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.wproj = nn.Linear(d_cond, d) if d_cond else None           # world conditioning → added to the CLS slot
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, self.N_CLASSES)

    def forward(self, ev: torch.Tensor, world: torch.Tensor = None, key_padding_mask=None) -> torch.Tensor:
        b = ev.shape[0]
        cls = self.cls.expand(b, -1, -1)
        if self.wproj is not None and world is not None:               # "am I looping — GIVEN this world state?"
            cls = cls + self.wproj(world).unsqueeze(1)
        x = torch.cat([cls, self.proj(ev)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=ev.device), key_padding_mask], dim=1)
        h = self.tr(x, src_key_padding_mask=key_padding_mask)
        return self.head(h[:, 0])                                       # [B, N_CLASSES] from the CLS slot


# ── THE COUPLED CORE — world encoder as the HUB; monitor + value CONDITION on world state ────────────────────────
class SelfMonitor(nn.Module):
    """The unified self-monitoring core. The WorldEncoder is the hub: it turns world-state vectors into world tokens,
    and BOTH the trajectory monitor ("am I looping?") and the value head ("how good is this direction?") condition on
    that world state. One `assess()` call answers all three — natural metacognition, no hand-written prompts."""

    def __init__(self, d_world: int, d_feat: int, d_event: int, d_model: int = 256, k: int = 8):
        super().__init__()
        from .world_layer import WorldEncoder                           # the hub (same module the LM prefix uses)
        self.world = WorldEncoder(d_world, d_model, k=k, layers=2)
        self.value = ValueHead(d_feat, d_cond=d_model)
        self.monitor = TrajectoryMonitor(d_event, d_cond=d_model)

    def world_state(self, world_vecs, mask=None):
        return self.world(world_vecs, mask).mean(dim=1)                 # [B, d_model] pooled world state

    def assess(self, world_vecs, resp_feats, event_seq, mask=None):
        w = self.world_state(world_vecs, mask)
        return {"world": w,
                "value": self.value(resp_feats, world=w),              # conditioned on world
                "regime": self.monitor(event_seq, world=w)}            # conditioned on world


# regime indices (shared with TrajectoryMonitor)
PRODUCTIVE, LOOPING, DRIFTING, CONVERGED = 0, 1, 2, 3


def trajectory_dpo(steps):
    """AUTO-HARVEST preference pairs from a SCORED trajectory — no hand labels. `steps` = a list of dicts each with
    `response` (str) and a scalar `value` (from the value head / an outcome proxy), plus optional `outcome` in
    {'pass','fail'}. Emits {prompt, chosen, rejected} when the agent IMPROVED (higher-value step ≻ the prior lower one),
    REGRESSED (prior ≻ the worse next), or on challenge SUCCESS/FAILURE (passing step ≻ failing step). This is the
    self-monitor + value head closing the DPO loop from the agent's own trajectory."""
    pairs = []
    prompt = "Given the task and where the agent is, produce the next step."
    for a, b in zip(steps, steps[1:]):
        va, vb = a.get("value", 0.0), b.get("value", 0.0)
        if vb - va > 0.05:                                             # improvement → the better step is chosen
            pairs.append({"prompt": prompt, "chosen": b["response"], "rejected": a["response"], "why": "improvement"})
        elif va - vb > 0.05:                                          # regression → the earlier better step is chosen
            pairs.append({"prompt": prompt, "chosen": a["response"], "rejected": b["response"], "why": "regression"})
    passes = [s for s in steps if s.get("outcome") == "pass"]
    fails = [s for s in steps if s.get("outcome") == "fail"]
    for p in passes:                                                  # challenge success ≻ failure
        for f in fails:
            pairs.append({"prompt": prompt, "chosen": p["response"], "rejected": f["response"], "why": "challenge"})
    return pairs


def self_test():
    d = 384
    vh = ValueHead(d)
    vc, vr = vh(torch.randn(8, d)), vh(torch.randn(8, d))
    assert vc.shape == (8,), vc.shape
    print("ValueHead OK — quality", tuple(vc.shape), "| params", sum(p.numel() for p in vh.parameters()))

    re = RetrievalEncoder(768)
    sc = re(torch.randn(8, 768), torch.randn(8, 5, 768))
    assert sc.shape == (8, 5), sc.shape
    print("RetrievalEncoder OK — scores", tuple(sc.shape), "| params", sum(p.numel() for p in re.parameters()))

    tm = TrajectoryMonitor(64, d_cond=256)
    lo = tm(torch.randn(8, 10, 64), world=torch.randn(8, 256))
    assert lo.shape == (8, 4), lo.shape
    print("TrajectoryMonitor OK — regimes", tuple(lo.shape), "| params", sum(p.numel() for p in tm.parameters()))

    sm = SelfMonitor(d_world=768, d_feat=768, d_event=64, d_model=256, k=8)
    a = sm.assess(torch.randn(8, 5, 768), torch.randn(8, 768), torch.randn(8, 10, 64))
    assert a["value"].shape == (8,) and a["regime"].shape == (8, 4) and a["world"].shape == (8, 256)
    print("SelfMonitor OK — world hub conditions value+regime |", {k: tuple(v.shape) for k, v in a.items()})

    traj = [{"response": "loop again", "value": 0.1}, {"response": "changed approach", "value": 0.6, "outcome": "pass"},
            {"response": "broke it", "value": 0.2, "outcome": "fail"}]
    dp = trajectory_dpo(traj)
    assert dp and all({"prompt", "chosen", "rejected"} <= p.keys() for p in dp)
    print("trajectory_dpo OK —", len(dp), "auto-harvested pairs, reasons:", sorted({p["why"] for p in dp}))


if __name__ == "__main__":
    self_test()
