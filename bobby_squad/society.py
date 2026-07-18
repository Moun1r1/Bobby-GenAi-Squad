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
