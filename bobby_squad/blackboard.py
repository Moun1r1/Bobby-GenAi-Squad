from typing import List, Optional

from .engine import Event


class Blackboard:
    def __init__(self, log):
        self.log = log

    def post(self, topic: str, data, by: str = "agent") -> str:
        """Post a row (a contract, a finding, a fragment). Returns the row id (the event id)."""
        return self.log.append(Event("board.post", {"topic": topic, "data": data, "by": by})).id

    def rows(self, topic: Optional[str] = None) -> List[dict]:
        return [{"id": e.id, "topic": e.payload.get("topic"), "data": e.payload.get("data"), "by": e.payload.get("by"),
                 "claimed_by": self.claimant(e.id)}
                for e in self.log.read("board.post") if topic is None or e.payload.get("topic") == topic]

    def _claims(self, row_id: str):
        return [e for e in self.log.read("board.claim") if e.payload.get("row") == row_id]

    def is_claimed(self, row_id: str) -> bool:
        return bool(self._claims(row_id))

    def claimant(self, row_id: str) -> Optional[str]:
        c = self._claims(row_id)
        return c[0].payload.get("by") if c else None            # the FIRST claim wins

    def claim(self, row_id: str, by: str) -> bool:
        """Try to claim a row. Returns True only if `by` is the first claimant (optimistic concurrency). A row already
        claimed by someone else returns False without a second execution."""
        if self.is_claimed(row_id):
            return False
        self.log.append(Event("board.claim", {"row": row_id, "by": by}))
        return self.claimant(row_id) == by
