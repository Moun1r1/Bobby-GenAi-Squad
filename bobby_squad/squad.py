from collections import deque


def _default_harvest(result, acc):
    acc |= set(result or [])                           # default accumulator = set union of harvested items
    return acc


def squad_solve(agents, units, work, verify=None, split=None, harvest=_default_harvest,
                accumulated=None, max_passes=80, observer=None):
    """Run a self-organizing squad over a recursive shared board until it drains (self-scaled plateau).

    agents            : list of workers (Agents). Round-robin over the shared board = coverage with no assigned roles.
    units             : initial work units — the SCOPE / content size (a big topic just has more/again-splittable units).
    work(agent, unit) : do one unit; returns this pass's raw result (e.g. the agent's answer / found items).
    verify(unit, acc) : run-DON'T-ask QUALITY GATE — a REAL check returning True (done) / False (under-covered).
                        None → every unit is accepted after one pass (no recursion).
    split(unit)       : RECURSION — return sub-units for an under-covered unit (e.g. halve a file's line-range), or a
                        falsy value if it can't be split further (accept the partial). None → no recursion.
    harvest(res, acc) : merge a pass result into the accumulator (default: set union). Returns the new accumulator.
    accumulated       : initial accumulator (default: empty set). Shared across all passes = the anti-forgetting memory.
    observer          : optional callable({kind:'pass', n, unit, size, done, board}) — watch the world live.

    Returns {result, passes, units_left}. units_left==0 means the board fully drained (converged by exhaustion).
    """
    board = deque(units)
    acc = accumulated if accumulated is not None else set()
    passes = 0
    while board and passes < max_passes:
        unit = board.popleft()
        agent = agents[passes % len(agents)]           # self-organizing: whichever worker is next takes the next unit
        acc = harvest(work(agent, unit), acc)
        passes += 1
        done = True if verify is None else bool(verify(unit, acc))
        if not done and split:                         # under-covered → sub-partition and re-queue (fractal depth)
            subs = split(unit)
            if subs:
                board.extend(subs)
        if observer:
            observer({"kind": "pass", "n": passes, "unit": unit, "done": done, "board": len(board),
                      "size": len(acc) if hasattr(acc, "__len__") else None})
    return {"result": acc, "passes": passes, "units_left": len(board)}
