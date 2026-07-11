"""bobby_squad.society — a shared feed so persistent-self agents can speak to and overhear each other.

Minimal and thread-safe: agents broadcast an utterance (usually their reflect() self-summary) and read the
recent utterances of others (which they fold into working memory). This is what turns N solo agents into a
society — and where the finding "action-persistence > narrative-persistence under social influence" appears.
"""
import threading
from typing import List, Tuple


class Society:
    def __init__(self):
        self.feed: List[Tuple[str, str]] = []
        self._lock = threading.Lock()

    def broadcast(self, name: str, text: str) -> None:
        with self._lock:
            self.feed.append((name, text))

    def overheard(self, name: str, k: int = 3, lookback: int = 10) -> List[str]:
        """The last `k` utterances from agents OTHER than `name`."""
        with self._lock:
            return [f"{n}: {t}" for n, t in self.feed[-lookback:] if n != name][-k:]

    def transcript(self) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self.feed)
