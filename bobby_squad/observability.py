"""bobby_squad.observability — RunStats: turn a swarm run into publish-ready, machine-readable stats.

Two hooks, zero behaviour change:
  • `metered = stats.meter(llm)` wraps ANY LLM callable to record per-call latency + token usage + errors
    (delegates every other attribute, incl. `.chat` / `.last_usage`, so it is a drop-in).
  • `Agent(observer=stats.observer)` counts moves / tool calls / cycles per agent from the live event stream.

Then `stats.round(...)` snapshots per-round health, and `stats.finalize(...)` emits ONE dict (and `save()` writes
JSON) with totals, latency percentiles, per-agent activity, per-round series, and a `stability` block (peak board,
dedup-reject rate, plateau round, error rate). It is the reusable substrate behind the published stats — run the
harness, cite `stats.json`; no hand-copied numbers.
"""
import json
import time
from collections import defaultdict


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return round(s[i], 3)


class _MeteredLLM:
    """Transparent proxy: times `__call__` / `.chat`, accumulates token usage from the inner llm's `last_usage`,
    counts failures. Every other attribute passes straight through, so callers can't tell it apart from the real LLM."""
    def __init__(self, inner, stats):
        self._inner, self._stats = inner, stats

    def __call__(self, messages, max_tokens=160, temperature=None):
        t = time.perf_counter()
        ok, out = True, ""
        try:
            out = self._inner(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:                      # a live run must survive one bad call — record it, don't crash
            ok = False
            self._stats._error(repr(e)[:200])
        self._stats._call(time.perf_counter() - t, getattr(self._inner, "last_usage", {}) or {}, ok)
        return out

    def chat(self, *a, **k):
        t = time.perf_counter()
        ok, msg = True, {}
        try:
            msg = self._inner.chat(*a, **k)
        except Exception as e:
            ok = False
            self._stats._error(repr(e)[:200])
        self._stats._call(time.perf_counter() - t, getattr(self._inner, "last_usage", {}) or {}, ok)
        return msg

    def __getattr__(self, n):                       # last_usage, model, url, temperature, … → the real llm
        return getattr(self._inner, n)


class RunStats:
    """Collects a full run into publish-ready stats. Thread-unsafe by design (the swarm protocol is sequential)."""
    def __init__(self, run: str = "run", model: str = "", agents: int = 0):
        self.meta = {"run": run, "model": model, "agents": agents}
        self.t0 = time.perf_counter()
        self.latencies = []                         # per-LLM-call seconds
        self.calls_ok = self.calls_err = 0
        self.tok_prompt = self.tok_completion = 0
        self.errors = []
        self.by_agent = defaultdict(lambda: {"moves": 0, "tools": 0, "cycles": 0, "results": 0})
        self.rounds = []
        self.peak_board = 0

    # ── hooks ────────────────────────────────────────────────────────────────────────────────────
    def meter(self, llm):
        """Wrap an LLM callable so every call is timed + token-counted. Returns a drop-in replacement."""
        return _MeteredLLM(llm, self)

    def observer(self, e: dict):
        """Pass to Agent(observer=…): counts moves / tools / cycles per agent from the event stream."""
        a = e.get("agent", "?")
        k = e.get("kind")
        rec = self.by_agent[a]
        if k == "move_start":
            rec["moves"] += 1
        elif k == "tool":
            rec["tools"] += 1
        elif k == "cycle":
            rec["cycles"] += 1
            rec["results"] += int(e.get("results", 0) or 0)

    # ── internal recorders (called by the meter) ──────────────────────────────────────────────────
    def _call(self, dt, usage, ok):
        self.latencies.append(dt)
        self.calls_ok += int(ok)
        self.calls_err += int(not ok)
        self.tok_prompt += int(usage.get("prompt_tokens", 0) or 0)
        self.tok_completion += int(usage.get("completion_tokens", 0) or 0)

    def _error(self, msg):
        self.errors.append(msg)

    # ── per-round snapshot ─────────────────────────────────────────────────────────────────────────
    def round(self, idx, board_size, new_findings, rejected, extra=None):
        """Snapshot one round's health. `board_size` = distinct ideas on the board (dedup already applied)."""
        self.peak_board = max(self.peak_board, int(board_size))
        row = {"round": idx, "board": int(board_size), "new": int(new_findings), "rejected": int(rejected),
               "elapsed_s": round(time.perf_counter() - self.t0, 1),
               "calls": len(self.latencies), "tokens": self.tok_prompt + self.tok_completion}
        if extra:
            row.update(extra)
        self.rounds.append(row)
        return row

    # ── final report ───────────────────────────────────────────────────────────────────────────────
    def finalize(self, ideas=None, extra=None):
        wall = time.perf_counter() - self.t0
        total_calls = self.calls_ok + self.calls_err
        seen = sum(r["new"] + r["rejected"] for r in self.rounds)
        rejected = sum(r["rejected"] for r in self.rounds)
        # plateau = first round after which no new distinct idea was admitted (novelty dried)
        plateau = None
        for r in self.rounds:
            if r["new"] == 0 and plateau is None:
                plateau = r["round"]
            elif r["new"] > 0:
                plateau = None
        out = {
            **self.meta,
            "wall_seconds": round(wall, 1),
            "llm": {
                "calls": total_calls, "ok": self.calls_ok, "errors": self.calls_err,
                "error_rate": round(self.calls_err / max(1, total_calls), 4),
                "tokens_prompt": self.tok_prompt, "tokens_completion": self.tok_completion,
                "tokens_total": self.tok_prompt + self.tok_completion,
                "latency_s": {"mean": round(sum(self.latencies) / max(1, len(self.latencies)), 3),
                              "p50": _pct(self.latencies, 50), "p90": _pct(self.latencies, 90),
                              "p95": _pct(self.latencies, 95), "max": round(max(self.latencies or [0]), 3)},
                "throughput_calls_per_min": round(total_calls / max(1e-9, wall) * 60, 1),
            },
            "per_agent": {a: dict(v) for a, v in sorted(self.by_agent.items())},
            "rounds": self.rounds,
            "stability": {
                "peak_board": self.peak_board,
                "dedup_reject_rate": round(rejected / max(1, seen), 4),   # rising with agents ⇒ dedup absorbs the swarm
                "plateau_round": plateau,                                 # None ⇒ never plateaued in the given rounds
                "call_error_rate": round(self.calls_err / max(1, total_calls), 4),
                "agents_active": sum(1 for v in self.by_agent.values() if v["cycles"] > 0 or v["moves"] > 0),
                "agents_configured": self.meta.get("agents", 0),
                "exceptions": len(self.errors),
            },
        }
        if ideas is not None:
            out["ideas"] = ideas
        if extra:
            out.update(extra)
        return out

    def save(self, path, ideas=None, extra=None):
        data = self.finalize(ideas=ideas, extra=extra)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
