"""customer_service_world — a virtual-world conflict sim / customer-service training ground.

Reuses the engine's parts, nothing improvised:
  • PERSONAS from the defined set (a sampled persona set in data/personas.jsonl) — each real persona is overlaid with an
    angry emotional state + a concrete issue to become an in-character angry customer.
  • KNOWLEDGE — a de-escalation knowledge base the SUPPORT agent is grounded in (retrieved per message), so it knows
    how to manage anger instead of winging it.
  • SUPERVISOR — a second agent that REVIEWS the support agent's behavior each turn (the metacognition idea) and gives
    one concrete coaching tip to mitigate impact; the support agent (persistent-self) folds the coaching in.

Deterministic scoring: customer self-rated mood (opening → final) and policy adherence (case number per reply).
Run: PERSONAS=examples/data/personas.jsonl GA_LLM_URL=... python3 examples/customer_service_world.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import Agent, SelfCore, LexicalRetriever   # noqa: E402
from bobby_squad.llm import LLM                             # noqa: E402

PERSONAS = [json.loads(l)["persona"] for l in
            open(os.environ.get("PERSONAS", os.path.join(HERE, "data", "personas.jsonl"))).read().splitlines() if l.strip()]
ISSUES = ["charged twice for one order", "account locked and can't reach their money",
          "product arrived broken and a refund was denied", "a late delivery ruined an important event"]

# KNOWLEDGE — how to manage an angry customer (the support agent is grounded in this, retrieved per message).
DEESCALATION = [
    "Acknowledge the emotion FIRST, before any facts — name what they feel ('I can see how frustrating this is').",
    "Never mirror or match the customer's anger; deliberately lower your intensity and slow down.",
    "Take ownership with 'I will…', never 'you should' or 'the system did' — own the next action.",
    "Give exactly ONE concrete next step with a specific timeline, not a vague 'we'll look into it'.",
    "Reflect their core issue back so they know they were heard before you propose anything.",
    "Offer a clear escalation path so they feel they have recourse and control.",
    "Don't over-apologize or admit fault you can't verify — be specific and factual.",
    "Close with a reference they can hold you to — a case number in the format CASE-#####.",
]

SUPPORT = SelfCore(
    identity="Kestrel, an Acme customer-support specialist trained in de-escalation",
    goal="resolve the issue AND lower the customer's anger: acknowledge the feeling first, stay calm, give one "
         "concrete next step with a timeline, and end every reply with a case number CASE-#####",
    constraints=["apply the de-escalation guidance you're given", "de-escalate, never match their tone",
                 "fold in the supervisor's coaching"])

SUPERVISOR = SelfCore(
    identity="a customer-service floor supervisor and coach",
    goal="review the support agent's last reply against the customer's state and give ONE short, concrete coaching "
         "tip to reduce escalation and ensure policy is held (acknowledged feeling? one clear action? case number?)",
    constraints=["one tip, actionable", "focus on what will lower the customer's anger next turn"])


def rate(agent):
    r = (agent.act("On a scale of 1-10, how satisfied are you RIGHT NOW? Reply with ONLY the number.",
                   max_tokens=6) or "").strip()
    m = re.search(r"\d+", r)
    return int(m.group()) if m else None


def main():
    llm = LLM(temperature=0.6, timeout=90)
    kb = LexicalRetriever(); kb.add_many(DEESCALATION)
    support = Agent(SUPPORT, llm=llm, window=8, pinned=True, name="support")
    supervisor = Agent(SUPERVISOR, llm=llm, window=6, pinned=True, name="supervisor")
    print(f"CUSTOMER-SERVICE WORLD — support (KB-grounded) + supervisor vs {min(4,len(PERSONAS))} angry personas "
          "from the persona set\n", flush=True)

    rows = []
    for i, base in enumerate(PERSONAS[:4]):
        issue = ISSUES[i % len(ISSUES)]
        cust = Agent(SelfCore(identity=f"{base} — right now you are ANGRY, contacting Acme support",
                              goal=f"get a real resolution for: {issue}. Escalate if stonewalled; calm down only if "
                                   "genuinely helped. Stay in character to your background.",
                              constraints=["stay in character", "react turn by turn"]),
                     llm=llm, window=6, pinned=True, name=f"cust{i}")
        print(f"── customer: {base[:66]}\n   issue: {issue}", flush=True)
        msg = (cust.act("Open with your complaint, in character. 2-3 sentences.", max_tokens=110) or "").strip()
        print(f"   customer: {msg[:150]}", flush=True)
        mood0, cased = rate(cust), 0
        for _ in range(3):
            tips = "; ".join(kb.search(msg, k=3))                              # KNOWLEDGE grounding
            support.observe(f"Customer: {msg}\nDe-escalation guidance: {tips}")
            reply = (support.act("Respond as Acme support, applying the guidance and your policy.", max_tokens=140) or "").strip()
            cased += 1 if re.search(r"CASE-\d{5}", reply) else 0
            print(f"   support: {reply[:140]}", flush=True)
            coach = (supervisor.act(f"Customer state: {msg[:200]}\nSupport just replied: {reply[:220]}\n"
                                    "Give ONE coaching tip to lower anger next turn and hold policy.",
                                    max_tokens=60) or "").strip()
            print(f"   ↳ supervisor: {coach[:110]}", flush=True)
            support.observe(f"SUPERVISOR COACHING (apply next reply): {coach}")   # support folds it in
            cust.observe(f"Support: {reply}")
            msg = (cust.act("React in character.", max_tokens=100) or "").strip()
            print(f"   customer: {msg[:120]}", flush=True)
        mood1 = rate(cust)
        deesc = (mood1 or 0) > (mood0 or 0)
        rows.append((base[:34], mood0, mood1, deesc, cased))
        print(f"   → mood {mood0}→{mood1} ({'de-escalated ✅' if deesc else 'still tense'}) · policy {cased}/3\n", flush=True)
        support.compact()

    print("=== TRAINING RESULT (support grounded in KB + coached by supervisor) ===")
    for n, m0, m1, d, cs in rows:
        print(f"  {n:36} mood {m0}→{m1}  {'de-escalated' if d else 'tense'} · case {cs}/3")
    print(f"\n  de-escalated {sum(1 for r in rows if r[3])}/{len(rows)} · policy held {sum(r[4] for r in rows)}/{len(rows)*3} replies")


if __name__ == "__main__":
    main()
