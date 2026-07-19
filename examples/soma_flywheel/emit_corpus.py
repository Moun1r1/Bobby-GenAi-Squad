#!/usr/bin/env python3
"""Emit the SOMA distillation corpus for fine-tuning: verified (prompt -> gold)
pairs drawn from Bobby's burn-in task families. Gold labels are deterministic
(exact set-equality graded), so every training label is trustworthy — the same
"verified only" guarantee DistillationCorpus enforces.

Writes train.jsonl (chat messages) and test.jsonl (prompt+gold for eval). Train
and test use disjoint seeds so we measure generalization, not memorization.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B
from bobby_squad import DistillationCorpus

OUT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]        # 8 x mixed tickets = verified train set
TEST_SEEDS = [90, 91, 92, 93, 94, 95]          # disjoint held-out, multiple seeds for a tighter CI

# capability families to include (skip 'prose' — irreducible/open generation, never a clean label)
INCLUDE = {"extract", "math", "code", "image", "algo"}


def to_label(gold):
    return "\n".join(gold) if isinstance(gold, (list, tuple)) else str(gold)


corpus = DistillationCorpus()
for s in TRAIN_SEEDS:
    for t in B.generate_mixed(seed=s, per=12):
        if t.get("kind") not in INCLUDE:
            continue
        corpus.record(input=t.get("blob", ""), output=to_label(t["gold"]),
                      capability=t.get("kind", ""), source="plugin", prompt=t["prompt"])

n = corpus.emit_sft(os.path.join(OUT, "soma_train.jsonl"), style="messages")
print("train records:", n, "coverage:", corpus.coverage(), flush=True)

# held-out test set (prompt + gold), same families, multiple disjoint seeds
test = []
for s in TEST_SEEDS:
    for t in B.generate_mixed(seed=s, per=12):
        if t.get("kind") not in INCLUDE:
            continue
        test.append({"prompt": t["prompt"], "gold": t["gold"], "kind": t.get("kind", ""), "seed": s})
with open(os.path.join(OUT, "soma_test.jsonl"), "w") as f:
    for r in test:
        f.write(json.dumps(r) + "\n")
print("test records:", len(test), flush=True)
