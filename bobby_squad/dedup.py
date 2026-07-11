"""bobby_squad.dedup — deterministic near-duplicate detection for progress-dedup.

Fixes the "looping" failure: agents redo a step they already did. We compare content-word overlap (Jaccard +
containment), ignoring latex/code spans so structurally-distinct math/code steps aren't falsely merged.
"""
import re
from typing import Iterable, Set


def words(s: str) -> Set[str]:
    s = re.sub(r"\$[^$]*\$", " ", s)      # drop inline latex so formulas don't dominate the comparison
    s = re.sub(r"`[^`]*`", " ", s)        # drop inline code
    return set(w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if len(w) > 3)


def near_dup(step: str, prior: Iterable[str], jaccard: float = 0.6, containment: float = 0.85) -> bool:
    """True if `step` is a near-duplicate of anything in `prior` (either high overlap, or one contains the other)."""
    a = words(step)
    if not a:
        return True
    for p in prior:
        b = words(p)
        if not b:
            continue
        inter = len(a & b)
        if inter / len(a | b) > jaccard or inter / min(len(a), len(b)) > containment:
            return True
    return False
