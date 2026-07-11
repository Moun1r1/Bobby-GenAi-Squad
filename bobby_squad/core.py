"""bobby_squad.core — the proven persistent-self mechanism, domain-free.

The one idea (see FINDINGS.md): split everything that enters an agent's context into two tiers and never let
compaction touch tier A.

  TIER A — SELF-CORE + PROGRESS   (pinned, always injected, immune to compaction)
  TIER B — WORKING MEMORY         (scrolling window, wiped on compaction)

Plus a self-model loop (re-ground), a convergence gate (stop when solved), and progress-dedup (don't repeat).
Nothing here is speculative — every piece reproduces a measured result.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .dedup import near_dup
from .planning import GUIDANCE_WF, PLAN_WF, EXECUTE_WF, EXECUTE_SOLO_WF, RESEARCH_WF, extract_json


@dataclass
class SelfCore:
    """Tier A identity. Small, durable, always injected verbatim — the thing compaction must never erase."""
    identity: str = ""
    goal: str = ""
    values: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    def render(self) -> str:
        p = []
        if self.identity:    p.append(f"You are {self.identity}.")
        if self.goal:        p.append(f"Your standing goal: {self.goal}.")
        if self.values:      p.append("Your values: " + "; ".join(self.values) + ".")
        if self.constraints: p.append("Constraints: " + "; ".join(self.constraints) + ".")
        return " ".join(p)


class PersistentContext:
    """Assembles the two-tier context and models real compaction.

    pinned=True  → the proven architecture: self-core + accumulated progress are always injected and SURVIVE
                   compaction; only the working window is wiped.
    pinned=False → the naive baseline: a generic system prompt, and everything (goal + progress) lives in the
                   working window, so compaction erases it.
    """

    def __init__(self, self_core: SelfCore, window: int = 6, pinned: bool = True,
                 generic_system: str = "You are a helpful assistant.", progress_cap: int = 60):
        self.self_core = self_core
        self.window = window
        self.pinned = pinned
        self.generic_system = generic_system
        self.progress_cap = progress_cap
        self.working: List[Tuple[str, str]] = []   # [(role, text)] — compactable
        self.progress: List[str] = []              # accumulated results — pinned (if pinned)

    def observe(self, text: str, role: str = "user") -> None:
        self.working.append((role, text))

    def record(self, item: str) -> None:
        """Commit a result to memory. Pinned → survives compaction; naive → lives only in the working window."""
        if self.pinned:
            self.progress.append(item)
        else:
            self.working.append(("assistant", item))   # naive: a result is just another compactable turn

    def compact(self, consolidate: bool = False) -> None:
        """Real compaction: wipe the working tier. The pinned tier (self-core + progress) is untouched.

        MEMORY-GATE (proven WIRE +191% retention, gains/proposals_gain.py): with consolidate=True, DISTINCT
        working-tier results are first moved into the pinned progress so valuable content isn't lost at the wipe —
        near_dup gates out noise/duplicates so the bounded pinned tier isn't filled with junk. (The proof used a
        ground-truth importance flag; novelty is the engine's available importance proxy.) Default False → unchanged,
        so the measured persistent-self behaviour is preserved."""
        if consolidate and self.pinned:
            for role, text in self.working:
                if role == "assistant" and text and not near_dup(text, self.progress):
                    self.progress.append(text)
        self.working = []

    def system_prompt(self) -> str:
        if not self.pinned:
            return self.generic_system
        s = self.self_core.render()
        if self.progress:
            s += ("\n\nProgress so far (pinned — continue from here, do not repeat any):\n"
                  + "\n".join(f"- {p}" for p in self.progress[-self.progress_cap:]))
        return s

    def messages(self, task: str) -> List[dict]:
        msgs = [{"role": "system", "content": self.system_prompt()}]
        for role, text in self.working[-self.window:]:
            msgs.append({"role": role, "content": text})
        msgs.append({"role": "user", "content": task})
        return msgs


class Agent:
    """A persistent-self agent: observe → act (grounded in the pinned tier) → record → converge, with optional
    automatic compaction. `llm` is any callable (messages: list[dict], max_tokens: int) -> str."""

    def __init__(self, self_core: SelfCore, llm: Callable, window: int = 6, pinned: bool = True,
                 compact_every: Optional[int] = None, name: Optional[str] = None, tools=None, observer=None):
        self.ctx = PersistentContext(self_core, window=window, pinned=pinned)
        self.llm = llm
        self.name = name or self_core.identity or "agent"
        self.compact_every = compact_every
        self.tools = tools           # optional ReadOnlyTools — when present, carry_out() grounds moves in real evidence
        self.observer = observer     # optional callable(event:dict) — a live STREAM of what the agent does in the world
        self.steps = 0
        self.last_trace = []         # tool calls from the most recent carry_out (for provenance)
        self.thought = ""            # the agent's evolving self-narrative (updated by reflect(); feeds its planning)
        self.evaluation = {}         # native TPB cognition from the last select_target (attitude/norm/control/reasoning)

    def _emit(self, kind: str, **data) -> None:
        """Emit one observation event to the stream (target/plan/move/tool/verify/…) — so you can WATCH the world and
        find bugs. Instrumentation only; never affects behaviour. Swallows observer errors."""
        if self.observer:
            try:
                self.observer({"agent": self.name, "kind": kind, **data})
            except Exception:
                pass

    # -- perception / memory ------------------------------------------------
    def observe(self, text: str, role: str = "user") -> None:
        self.ctx.observe(text, role)

    def record(self, item: str) -> None:
        self.ctx.record(item)

    def compact(self, consolidate: bool = False) -> None:
        self.ctx.compact(consolidate=consolidate)      # Memory-Gate opt-in (proven WIRE +191%)

    # -- action -------------------------------------------------------------
    def act(self, task: str, max_tokens: int = 160) -> str:
        """Produce the next step toward the goal, grounded in the pinned self-core + progress. Auto-compacts at
        the START of the step on cadence (a fresh session begins by wiping working memory) — so after a wipe a
        pinned agent resumes from its pinned progress while a naive one is genuinely lost."""
        if self.compact_every and self.steps > 0 and self.steps % self.compact_every == 0:
            self.compact()
        out = (self.llm(self.ctx.messages(task), max_tokens=max_tokens) or "").strip()
        self.ctx.observe(out, role="assistant")
        self.steps += 1
        return out

    def next_step(self, instruction: str, dedup: bool = True, max_tokens: int = 160) -> Optional[str]:
        """One advancing step with progress-dedup: if the step repeats prior progress, retry once, else give up
        (returns None). On success the step is recorded to the pinned tier."""
        step = self.act(instruction, max_tokens=max_tokens)
        if step.upper().startswith("DONE") or not step:
            return None
        if dedup and near_dup(step, self.ctx.progress):
            step = self.act(instruction + "\nNOTE: that repeats a step already done — give a genuinely different, "
                                          "later step, or reply DONE.", max_tokens=max_tokens)
            if step.upper().startswith("DONE") or not step or near_dup(step, self.ctx.progress):
                return None
        self.record(step)
        return step

    # -- self-model loop + convergence -------------------------------------
    def reflect(self, prompt: str = "In one sentence, restate who you are and the goal you are working toward "
                                    "and its status.", max_tokens: int = 80) -> str:
        """Re-ground: a short self-summary from the pinned self-core (the self-model loop). Returns the utterance
        (useful to broadcast to a Society)."""
        msgs = [{"role": "system", "content": self.ctx.self_core.render()},
                {"role": "user", "content": prompt}]
        self.thought = (self.llm(msgs, max_tokens=max_tokens) or "").strip()
        return self.thought

    # -- self-taught loop (Needs→Plan→Cognition, world-free) — the agent behaves ALONE ----------------
    def _progress_text(self, cap: int = 20) -> str:
        # the agent plans from what it has "done" — which lives in the PINNED tier (survives compaction) for a
        # pinned agent, or in the wipeable WORKING tier for a naive one. This is what makes the compaction A/B honest.
        items = self.ctx.progress if self.ctx.pinned else [t for _r, t in self.ctx.working]
        return "\n".join(f"- {p}" for p in items[-cap:]) or "(nothing yet)"

    def select_target(self, max_tokens: int = 340) -> str:   # 200 truncated the verbose TPB JSON → false DONE (audit D)
        """Stage 1 — the agent SELF-SELECTS its next target (TPB guidance selection) from its own goal + thought
        + progress. Nothing is handed in; the target is the agent's choice."""
        prompt = GUIDANCE_WF.format(identity=self.ctx.self_core.identity, goal=self.ctx.self_core.goal,
                                    thought=self.thought or "(just starting)", progress=self._progress_text())
        o = extract_json(self.llm([{"role": "user", "content": prompt}], max_tokens=max_tokens))
        self.evaluation = o.get("evaluation") or {}   # native affect (attitude) + reasoning — the agent's own cognition
        target = str(o.get("selected_target", "")).strip()
        self._emit("target", target=target)
        return target

    def make_plan(self, target: str, max_steps: int = 4, max_tokens: int = 280) -> list:
        """Stage 2 — the agent SELF-GENERATES the steps to reach the target it chose."""
        prompt = PLAN_WF.format(target=target, goal=self.ctx.self_core.goal,
                                thought=self.thought or "(just starting)", progress=self._progress_text(),
                                max_steps=max_steps)
        o = extract_json(self.llm([{"role": "user", "content": prompt}], max_tokens=max_tokens))
        steps = ((o.get("plan") or {}).get("steps")) or []
        steps = [s for s in steps if isinstance(s, dict) and s.get("intention")]
        self._emit("plan", target=target, steps=[s.get("intention", "")[:70] for s in steps])
        return steps

    def execute(self, intention: str, max_tokens: int = 1024, mode: str = "discussion",
                material: Optional[str] = None) -> str:   # ceiling, not a target — model stops at its own end
        """Stage 3 — carry out the agent's OWN self-chosen intention.

        mode="discussion" (default) — grounds in what it has HEARD (working memory) and engages peers; for a
          Society / live exchange.
        mode="solo" — grounds in PROVIDED `material` (or the working tier if none) and reports a finding; for solo
          document/code tasks, where the discussion prompt answers NOTHING by design because it treats the material
          as already-said. The self-generated intention is unchanged; only the carry-out fits the task."""
        if mode == "solo":
            src = material if material is not None else ("\n".join(t for _r, t in self.ctx.working) or "(no material)")
            prompt = EXECUTE_SOLO_WF.format(identity=self.ctx.self_core.identity, goal=self.ctx.self_core.goal,
                                            intention=intention, material=src)
        else:
            heard = [t for r, t in self.ctx.working if r != "assistant"][-5:]
            context = "\n".join(f"- {h}" for h in heard) or "(nothing yet — you open the discussion)"
            prompt = EXECUTE_WF.format(identity=self.ctx.self_core.identity, goal=self.ctx.self_core.goal,
                                       intention=intention, context=context)
        return (self.llm([{"role": "user", "content": prompt}], max_tokens=max_tokens) or "").strip()

    # -- tool-grounded, open-move carry-out (the generative-engine way: one neutral frame, any self-chosen move) --
    def carry_out(self, intention: str, move: str = "", max_rounds: int = 10, max_tokens: int = 2000) -> str:
        """Carry out ANY self-chosen move (mine/invent/compose/critique/experiment/…) through ONE behavior-neutral,
        TOOL-GROUNDED frame. With tools, the agent investigates real evidence before it claims (native tool-loop);
        without tools it degrades to the solo executor. No per-move prompt exists — the agent decides how to do its
        move; the tools keep it honest. Records the tool trace on self.last_trace for provenance.

        max_tokens is generous (2000) because a tool-call turn that WRITES a full experiment script must not be
        truncated mid-code (a small budget was producing SyntaxErrors); the extra rounds let it write→run→fix→green."""
        if self.tools is None:
            return self.execute(intention, max_tokens=max_tokens, mode="solo")
        from .agent_tools import investigate as _inv
        self._emit("move_start", move=move, intention=intention[:70])
        prompt = RESEARCH_WF.format(identity=self.ctx.self_core.render(), goal=self.ctx.self_core.goal,
                                    progress=self._progress_text(), move=move or "(self-directed)", intention=intention)
        answer, trace = _inv(self.llm, prompt, self.tools, max_rounds=max_rounds, max_tokens=max_tokens,
                             on_event=self._emit)         # stream each tool call live
        self.last_trace = trace
        self._emit("move_end", move=move, tools=[t[0] for t in trace], chars=len(answer))
        return answer

    def research_cycle(self, max_steps: int = 3, dedup: bool = True, max_rounds: int = 10, replans: int = 2) -> dict:
        """One self-directed, tool-grounded research cycle: SELF-SELECT a target → SELF-GENERATE a plan whose steps
        each name their OWN move → carry each out tool-grounded → record to pinned progress.

        ADAPTIVE RE-PLANNING (squad fix #3): the plan is no longer generated once and blindly followed — after a
        round of steps, make_plan is called AGAIN against the UPDATED progress (so an early result reshapes the next
        steps), up to `replans` times, stopping when a round adds nothing new. max_rounds default raised 6→10 (fix #2:
        the old budget exhausted before build→run→fix). Returns {target, results}."""
        target = self.select_target()
        if not target or target.upper().startswith("DONE"):
            return {"target": None, "results": []}
        results = []
        for _round in range(max(1, replans)):
            new = 0
            for s in self.make_plan(target, max_steps=max_steps):   # re-planned each round from updated progress
                intention, move = s.get("intention", ""), s.get("type", "")
                if dedup and near_dup(intention, [r["intention"] for r in results] + self.ctx.progress):
                    continue
                out = (self.carry_out(intention, move, max_rounds=max_rounds) or "").strip()
                if out and not out.upper().startswith(("NOTHING", "NONE")):
                    self.record(f"[{move}] {intention} → {out[:200]}")
                    results.append({"intention": intention, "move": move, "result": out})
                    new += 1
            if new == 0:                                   # plan exhausted → stop re-planning
                break
        self.steps += 1
        self._emit("cycle", target=target, results=len(results))
        return {"target": target, "results": results}

    def autonomous_loop(self, verify_fn=None, max_cycles: int = 8, max_steps: int = 3, max_rounds: int = 10) -> dict:
        """END-TO-END loop, from the squad's own self-diagnosis of why a single research_cycle can't finish real work:
          • research_cycle is ONE-SHOT — it never loops back to select_target. Here we repeat it, each cycle
            re-selecting the target from UPDATED progress → adaptive re-planning (fixes the static-plan limitation).
          • the old convergence check ASKS 'is it done? YES/NO' with no external verification. Here the exit condition
            is `verify_fn()` — a REAL check (e.g. a run's exit code / expected output), RUN not asked. This is the
            core fix for 'they declare victory in prose without executing'. Falls back to converged() if no verifier.
          • carry_out's 6-round budget is too small for build→run→fix; we pass a larger max_rounds through.
        Returns {cycles, verified}."""
        cycles = []
        for i in range(max_cycles):
            v = bool(verify_fn() if verify_fn else self.converged())
            self._emit("check", cycle=i, verified=v)
            if v:
                break
            res = self.research_cycle(max_steps=max_steps, max_rounds=max_rounds)
            cycles.append(res)
            if res.get("target") is None:                  # nothing left to pursue
                break
        verified = bool(verify_fn() if verify_fn else self.converged())
        self._emit("done", cycles=len(cycles), verified=verified)
        return {"cycles": cycles, "verified": verified}

    def autonomous_cycle(self, max_steps: int = 4, do_execute: bool = True, dedup: bool = True) -> dict:
        """One full self-taught cycle: SELF-SELECT a target → SELF-GENERATE a plan → carry out each own
        intention → record results to pinned progress. The agent behaves alone; nothing is prescribed."""
        target = self.select_target()
        if not target or target.upper().startswith("DONE"):
            return {"target": None, "done": True, "results": []}
        results = []
        for s in self.make_plan(target, max_steps=max_steps):
            intention = s.get("intention", "")
            if dedup and near_dup(intention, [r["intention"] for r in results] + self.ctx.progress):
                continue
            out = self.execute(intention) if do_execute else ""
            self.record(f"[{s.get('type', '')}] {intention}" + (f" → {out[:160]}" if out else ""))
            results.append({"intention": intention, "type": s.get("type", ""), "result": out})
        self.steps += 1
        return {"target": target, "done": False, "results": results}

    def converged(self, max_tokens: int = 10) -> bool:   # 5 could truncate "YES"+whitespace before the stop (audit V)
        """Cheap YES/NO gate: is the goal fully achieved? WEAK — this ASKS the model, with no external check, so it can
        declare victory in prose without executing anything (the squad's fix #4). Prefer verify() when tools exist."""
        prog = "\n".join(f"{i+1}. {p}" for i, p in enumerate(self.ctx.progress)) or "(nothing yet)"
        q = (f"Goal: {self.ctx.self_core.goal}\nProgress so far:\n{prog}\n\nIs this goal now FULLY and correctly "
             "achieved with nothing essential missing? Answer with ONLY the single word YES or NO.")
        return (self.llm([{"role": "user", "content": q}], max_tokens=max_tokens) or "").strip().upper().startswith("YES")

    def verify(self, check_script: str = "", marker: str = "") -> bool:
        """RUN-don't-ask verification (squad fix #4): instead of asking the model 'is it done?', RUN a real check and
        read the OUTCOME. With `check_script`, write+run it and require exit 0 (+ optional `marker` in the output);
        otherwise run every sandbox script and pass if any exits 0 (+ marker). Falls back to converged() if the agent
        has no runnable tools. This is the external, ground-truth convergence the ASK-based gate lacks."""
        run = getattr(self.tools, "run", None)
        if not callable(run):
            return self.converged()
        import glob
        import os
        ok = lambda out: "[exit 0" in out and (marker in out if marker else True)   # noqa: E731
        if check_script and hasattr(self.tools, "write"):
            self.tools.write("_verify.py", check_script)
            return ok(run("_verify.py"))
        sb = getattr(self.tools, "sandbox", None)
        if sb:
            for p in glob.glob(os.path.join(sb, "**", "*.py"), recursive=True):
                if ok(run(os.path.relpath(p, sb))):
                    return True
        return False

    # -- read-only investigation (verify before you claim) -----------------------------------------
    def investigate(self, task: str, root: str, max_rounds: int = 6, max_tokens: int = 500):
        """Investigate a codebase with READ-ONLY tools (grep / read / ls / find) before answering — so the agent
        VERIFIES against real code instead of guessing. Returns (answer, tool_trace). Strictly read-only, sandboxed
        to `root`."""
        from .agent_tools import ReadOnlyTools, investigate as _inv
        return _inv(self.llm, task, ReadOnlyTools(root), max_rounds=max_rounds, max_tokens=max_tokens)


def stream_observer(e: dict) -> None:
    """A ready-made observer: prints the live event stream so you can WATCH what an agent does in the world
    (target → plan → move → each tool call → cycle → verify → done) and spot bugs. Pass to Agent(observer=...)."""
    a, k = e.get("agent", "?"), e.get("kind")
    line = {
        "target":     lambda: f"◆ TARGET  {e.get('target', '')[:90]}",
        "plan":       lambda: f"▤ PLAN    {e.get('steps')}",
        "move_start": lambda: f"▶ {e.get('move', 'move')}  {e.get('intention', '')}",
        "tool":       lambda: f"    · {e.get('name')}({e.get('args')})",
        "tool_done":  lambda: f"    ✓ {e.get('name')} → {e.get('out', '')}",
        "move_end":   lambda: f"◀ {e.get('move', 'move')} done · tools={e.get('tools')} · {e.get('chars')}c",
        "cycle":      lambda: f"⟳ cycle → {e.get('results')} results",
        "check":      lambda: f"? verify@cycle{e.get('cycle')} = {e.get('verified')}",
        "done":       lambda: f"■ DONE  cycles={e.get('cycles')}  verified={e.get('verified')}",
    }.get(k, lambda: f"· {k}: {e}")
    print(f"  [{a}] {line()}", flush=True)
