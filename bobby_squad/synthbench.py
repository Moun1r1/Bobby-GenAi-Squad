import random
import string
from typing import List

_WORDS = ("the meeting notes budget review timeline vendor invoice shipment logistics quarterly report metric "
          "dashboard onboarding retro standup roadmap backlog sprint incident postmortem rollout").split()


def _rand_key(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(4)) + "-" + str(rng.randint(1000, 9999))


def _distractor(rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(8, 16))) + "."


def make(seed: int, length_chars: int = 8000, needles: int = 1) -> List[dict]:
    """Return `needles` QA items over ONE shared haystack. Each item: {context, question, answers, needle}.
    The needle is 'The <TOPIC> access code is <KEY>.' — <KEY> is random (unguessable); the question asks for it."""
    rng = random.Random(seed)
    topics = rng.sample(["vault", "archive", "server", "lab", "depot", "relay", "gateway", "console"],
                        k=min(needles, 8))
    planted = [(t, _rand_key(rng)) for t in topics]
    # build a haystack of distractors, insert each needle at a pseudo-random depth
    body: List[str] = []
    while sum(len(s) for s in body) < length_chars:
        body.append(_distractor(rng))
    for t, key in planted:
        pos = rng.randint(1, max(1, len(body) - 1))
        body.insert(pos, f"The {t} access code is {key}.")
    context = " ".join(body)
    return [{"context": context, "question": f"What is the {t} access code?", "answers": [key], "needle": key}
            for t, key in planted]


def dataset(seed: int, n_docs: int = 8, **kw) -> List[dict]:
    """A flat list of QA rows across `n_docs` independent haystacks (fields match the memory_gains harness)."""
    rows: List[dict] = []
    for i in range(n_docs):
        for item in make(seed * 1000 + i, **kw):
            rows.append({"context": item["context"], "input": item["question"], "answers": item["answers"]})
    return rows
