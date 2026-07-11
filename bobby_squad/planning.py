"""bobby_squad.planning — AgentSociety's self-taught Needs→Plan→Cognition loop, made world-free.

The agent generates its OWN behavior: it self-selects a target (Theory-of-Planned-Behaviour guidance
selection) and then self-generates the steps to reach it (detailed plan). These prompts mirror the structure
of AgentSociety's `GUIDANCE_SELECTION_PROMPT` and `DETAILED_PLAN_PROMPT`, with the simulation-only fields
(weather / location / mobility·economy action types) removed and replaced by domain-general step types. We
provide the MECHANISM; the target, the plan, and the content are entirely the agent's own — no task is handed in.
"""
import json
import re

# stage 1 — the agent SELF-SELECTS what to do next (TPB), from its own goal + thought + progress
GUIDANCE_WF = """As an intelligent agent's decision system, help me determine the most useful thing to do next to advance my standing goal.

My role: {identity}
My standing goal: {goal}
My current thought: {thought}
What I have already done:
{progress}

Please evaluate and choose along these three dimensions (Theory of Planned Behaviour):
1. Attitude: my own preference and evaluation of the option
2. Subjective Norm: how relevant standards or others would view it
3. Perceived Control: how feasible it is for me to do right now

Respond ONLY in JSON:
{{"selected_target": "<the next concrete sub-goal I choose to pursue now, in my own words>",
  "evaluation": {{"attitude": <0-1>, "subjective_norm": <0-1>, "perceived_control": <0-1>, "reasoning": "<why>"}}}}"""

# stage 2 — the agent SELF-GENERATES the steps to fulfil the target it just chose
PLAN_WF = """As an intelligent agent's plan system, generate the specific steps needed to fulfil the target I selected.

Target: {target}
My standing goal: {goal}
My current thought: {thought}
What I have already done:
{progress}

Notes:
1. "type" is a SHORT label YOU choose for the kind of move each step makes — pick whatever genuinely fits it
   (e.g. investigate, invent, compose, experiment, critique, synthesize, reason, verify). You are NOT limited to a
   fixed list; name the move the work needs.
2. include only steps necessary to fulfil the target (at most {max_steps} steps)
3. each intention must be concise, concrete and NOT repeat what I have already done

Respond ONLY in JSON:
{{"plan": {{"target": "{target}", "steps": [{{"intention": "<concise intention>", "type": "<the move you choose>"}}]}}}}"""

# stage 3 (research) — a SINGLE behavior-neutral, TOOL-GROUNDED frame that carries out ANY self-chosen move. It does
# NOT script the move (no per-move prompt): it hands the agent its own intention + tools and tells it to ground
# itself in real evidence before claiming. Mine / invent / compose / critique / experiment all run through this one
# frame — the agent decides HOW to do its move; the tools keep it honest. This is the generative-engine way: grow
# the move-space + tools, never add a prompt per behavior.
RESEARCH_WF = """{identity}. My standing goal: {goal}.

What I have already established (do NOT repeat any of it):
{progress}

The move I chose for this step: {move}
My intention: {intention}

Carry it out now, GROUNDING yourself in reality before you claim: read the real code/files, and when a claim needs
proof, WRITE and RUN a check in the sandbox. Output the concrete result of this move, tied to what I actually read
or ran. If there is genuinely nothing new and true to add, it is correct to output only: NOTHING."""

# stage 3 — the agent carries out ITS OWN intention and produces the actual result, IN the live exchange
EXECUTE_WF = """{identity}. My standing goal: {goal}.

What has already been said in the discussion (do NOT restate any of it):
{context}

My next intention: {intention}

Carry it out now and output the actual RESULT — real content, ONE substantive new contribution. Engage directly
with what was said above: build on a specific point, or challenge it and say which and why. No filler, no flowery
restatement, no generic praise, no summary of what others said. If you have nothing genuinely new, say only: NOTHING."""

# stage 3 (solo) — carry out the self-chosen intention against PROVIDED MATERIAL, not a live discussion. Use this
# for solo document/code tasks: the discussion-oriented EXECUTE_WF above (correctly) answers NOTHING there because
# it is told to add a contribution BEYOND what was "said", so it treats the material as already-said and declines.
EXECUTE_SOLO_WF = """{identity}. Standing goal: {goal}.

Work ONLY from this material:
{material}

Your intention: {intention}

Carry it out against the material and output the concrete RESULT — one substantive finding grounded in the
material, citing the specific evidence (a name, number, or line). If the material does not support it, say only: NONE."""


def extract_json(text: str):
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        # tolerant fallback: strip trailing commas
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception:
            return {}
