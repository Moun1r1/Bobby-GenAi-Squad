"""board_tools — give the swarm a TOOL to ORGANIZE its own shared board, so states + structure are the agents',
not a hardcoded lifecycle. Composes the sandbox investigation tools with three board moves the agent self-selects:

  board()              — read the shared board (every idea grouped by the state the swarm gave it, with content).
  set_state(idea, s)   — assign an idea ANY state that fits the work (exploring / promising / blocked /
                         ready-to-build / needs-evidence / …). States are emergent — the swarm defines them.
  merge(keep, fold)    — fold one idea into another when they're the same direction.

The agent picks WHEN to organize (it's a move in the space, not a scripted pass). The deterministic identity FLOOR in
IdeaLedger.admit still repels regeneration independent of whatever state an agent assigns, so organizing is safe.
"""
from .agent_tools import SandboxTools, SANDBOX_SCHEMAS

BOARD_SCHEMAS = [
    {"type": "function", "function": {"name": "board", "description": "Read the shared board: every idea grouped by "
     "the STATE the swarm gave it, with its content. Look before you propose so you build on what's there instead "
     "of restating it.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_state", "description": "Organize the board — give an idea (by its "
     "label) ANY state that fits the work right now (e.g. exploring, promising, blocked-on-X, ready-to-build, "
     "needs-evidence, merged). States are the swarm's to define; there is no fixed list.",
     "parameters": {"type": "object", "properties": {"idea": {"type": "string", "description": "the idea's label"},
        "state": {"type": "string", "description": "any state that fits"}}, "required": ["idea", "state"]}}},
    {"type": "function", "function": {"name": "merge", "description": "Fold one idea into another (both by label) "
     "when they are the same direction — consolidate the board.", "parameters": {"type": "object", "properties": {
        "keep": {"type": "string", "description": "label of the idea to keep"},
        "fold": {"type": "string", "description": "label of the idea to fold in"}}, "required": ["keep", "fold"]}}},
]


class BoardTools(SandboxTools):
    """Investigation (inherited) + board-organization, so an agent can read the repo AND organize the shared board
    in the same self-directed loop. `ledger` is the shared IdeaLedger the swarm writes to."""

    def __init__(self, ledger, repo_root, sandbox_root, **kw):
        super().__init__(repo_root, sandbox_root, **kw)
        self.ledger = ledger
        self.schemas = SANDBOX_SCHEMAS + BOARD_SCHEMAS

    def _find(self, label):
        lab = (label or "").lower().strip()
        if not lab:
            return None
        for it in self.ledger.ideas:                       # substring match first
            if lab in it["label"].lower() or it["label"].lower() in lab:
                return it
        toks = {w for w in lab.split() if len(w) > 3}       # else best token overlap
        best, bj = None, 0
        for it in self.ledger.ideas:
            j = len(toks & {w for w in it["label"].lower().split() if len(w) > 3})
            if j > bj:
                bj, best = j, it
        return best

    def board(self):
        return "\n".join(self.ledger.signal()) or "(board empty)"

    def set_state(self, idea, state):
        it = self._find(idea)
        if it is None:
            return f"(no idea on the board matches '{idea}' — call board() to see the labels)"
        self.ledger.set_state(it, state, by="agent")
        return f"organized: [{it['label']}] → state '{it['status']}'"

    def merge(self, keep, fold):
        a, b = self._find(keep), self._find(fold)
        if a is None or b is None or a is b:
            return f"(need two DIFFERENT ideas that exist; got {keep!r} and {fold!r})"
        self.ledger.merge(a, b, by="agent")
        return f"merged [{b['label']}] into [{a['label']}]"

    def run_json(self, name, args):
        if name == "board":
            return self.board()
        if name == "set_state":
            return self.set_state(args.get("idea", ""), args.get("state", ""))
        if name == "merge":
            return self.merge(args.get("keep", ""), args.get("fold", ""))
        return super().run_json(name, args)
