"""bobby_squad.world_layer — a trainable WORLD TRANSFORMER LAYER that feeds the framework's world-state to a
frozen LM as EMBEDDINGS instead of chat text (see the `world-transformer-layer` vault note).

The chat bottleneck: every step re-serializes the whole world (goal, vault subgraph, memory) into tokens. Instead:
  WorldEncoder — perceiver-style learned latent slots cross-attend to a bank of world-state vectors (vault-note /
                 memory embeddings) → K fixed "world tokens" in the LM's embedding space.
  WorldPrefixLM — prepend those world tokens to the frozen LM's input embeddings. Only the encoder trains; the base
                 LM is frozen. State enters as vectors (fixed size, differentiable) — chat is the OUTPUT channel only.

Pure torch (no framework deps) so it is pushed to the GPU worker and run there. Trained with a real objective:
the frozen LM must predict a world-grounded target BETTER with the world tokens than without — that delta is the
proof the layer works (and that world-as-embedding beats no-world), memory-safe by construction (tiny encoder,
frozen base).
"""
from typing import Optional

import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    """One perceiver block: latent slots attend to the world-vector bank, then a FFN. Pre-norm + residual."""

    def __init__(self, d_model: int, heads: int = 8, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n1q = nn.LayerNorm(d_model)
        self.n1k = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * mlp_mult), nn.GELU(),
                                 nn.Linear(d_model * mlp_mult, d_model))

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        a, _ = self.attn(self.n1q(q), self.n1k(ctx), self.n1k(ctx), key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        q = q + self.mlp(self.n2(q))
        return q


class WorldEncoder(nn.Module):
    """World-state vectors [B, N, d_world] → K world tokens [B, K, d_model]. K learned slots, `layers` cross-attn
    blocks. Small by design (memory-safe): this is the only thing that trains."""

    def __init__(self, d_world: int, d_model: int, k: int = 16, heads: int = 8, layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(d_world, d_model)
        self.slots = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.blocks = nn.ModuleList([CrossAttnBlock(d_model, heads) for _ in range(layers)])
        self.out = nn.LayerNorm(d_model)

    def forward(self, world_vecs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = world_vecs.shape[0]
        ctx = self.in_proj(world_vecs)                                    # [B, N, d_model]
        q = self.slots.unsqueeze(0).expand(b, -1, -1)                     # [B, K, d_model]
        kpm = (~mask.bool()) if mask is not None else None               # True = PAD (ignored)
        for blk in self.blocks:
            q = blk(q, ctx, key_padding_mask=kpm)
        return self.out(q)                                               # [B, K, d_model] world tokens


class WorldPrefixLM(nn.Module):
    """Prepend the world tokens to a FROZEN causal LM's input embeddings and score the target. Base LM never trains."""

    def __init__(self, base_lm, d_world: int, k: int = 16, heads: int = 8, layers: int = 2):
        super().__init__()
        self.lm = base_lm
        for p in self.lm.parameters():
            p.requires_grad_(False)
        d_model = base_lm.get_input_embeddings().embedding_dim
        self.encoder = WorldEncoder(d_world, d_model, k=k, heads=heads, layers=layers)
        self.k = k

    def forward(self, input_ids, attention_mask, labels, world_vecs=None, world_mask=None):
        tok_emb = self.lm.get_input_embeddings()(input_ids)              # [B, T, d]
        b, t, _ = tok_emb.shape
        if world_vecs is not None:                                      # WITH world tokens
            wtok = self.encoder(world_vecs, world_mask)                 # [B, K, d] (fp32 encoder)
            inp = torch.cat([wtok.to(tok_emb.dtype), tok_emb], dim=1)   # cast to the frozen base's dtype (bf16)
            am = torch.cat([torch.ones(b, self.k, device=input_ids.device, dtype=attention_mask.dtype), attention_mask], dim=1)
            pre = torch.full((b, self.k), -100, device=input_ids.device, dtype=labels.dtype)  # prefix isn't a target
            lab = torch.cat([pre, labels], dim=1)
            return self.lm(inputs_embeds=inp, attention_mask=am, labels=lab).loss
        return self.lm(inputs_embeds=tok_emb, attention_mask=attention_mask, labels=labels).loss  # WITHOUT (baseline)

    def trainable_parameters(self):
        return (p for p in self.encoder.parameters() if p.requires_grad)


def self_test():
    """No-LM shape check — verifies the encoder wiring without downloading weights."""
    enc = WorldEncoder(d_world=768, d_model=256, k=16, layers=2)
    wv = torch.randn(4, 12, 768)
    out = enc(wv)
    assert out.shape == (4, 16, 256), out.shape
    print("world_layer self-test OK — world tokens", tuple(out.shape),
          "| encoder params", sum(p.numel() for p in enc.parameters()))


if __name__ == "__main__":
    self_test()
