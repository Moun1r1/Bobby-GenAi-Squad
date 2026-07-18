import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import Agent, SelfCore   # noqa: E402
from bobby_squad.llm import LLM           # noqa: E402
from bobby_squad.dedup import near_dup     # noqa: E402


def distinct(lessons):
    """How many genuinely different lessons the agent authored (the engine's own near-dup gate) — the size of the
    experiential knowledge it grew, vs. just repeating one takeaway."""
    kept = []
    for l in lessons:
        if l and not near_dup(l, kept):
            kept.append(l)
    return len(kept)

PERSONAS = [json.loads(l)["persona"] for l in
            open(os.path.join(HERE, "data", "personas.jsonl")).read().splitlines() if l.strip()]

# MANY real-world sims — same engine, different conflict. Each: the difficult stance a counterpart takes, the
# learner's role+goal, and the 1-10 metric the learner is trying to move (rated by the counterpart, deterministic).
WORLDS = [
    {"name": "customer service",
     "stance": "You are ANGRY: {issue}. You feel unheard and want it fixed now.",
     "role": "an Acme customer-support specialist",
     "goal": "resolve the issue and lower the customer's anger",
     "metric": "how satisfied and calm you feel with support RIGHT NOW",
     "issues": ["charged twice for one order", "account locked out of your own money",
                "a broken product and a denied refund", "a late delivery that ruined an event"]},
    {"name": "sales negotiation",
     "stance": "You are a SKEPTICAL buyer who thinks this is overpriced and distrusts salespeople: {issue}.",
     "role": "a solutions consultant selling a B2B software plan",
     "goal": "build trust and move the buyer toward a fair deal without pressure",
     "metric": "how much you trust this seller and want to keep talking RIGHT NOW",
     "issues": ["the price is 3x a competitor", "you were burned by a vendor last year",
                "you doubt it does what they claim", "your budget was already spent"]},
    {"name": "teaching",
     "stance": "You are a FRUSTRATED student who 'just doesn't get it' and feels stupid: {issue}.",
     "role": "a patient tutor",
     "goal": "help the student actually understand and feel capable",
     "metric": "how much you understand and how confident you feel RIGHT NOW",
     "issues": ["you keep failing at fractions", "recursion makes no sense to you",
                "you blanked on the exam again", "everyone else seems to get it but you"]},
    {"name": "care reassurance",
     "stance": "You are an ANXIOUS patient catastrophizing about your health: {issue}.",
     "role": "a calm triage nurse",
     "goal": "reassure realistically and give a clear safe next step without dismissing the fear",
     "metric": "how reassured and clear on next steps you feel RIGHT NOW",
     "issues": ["a headache you're sure is a tumor", "chest flutter you think is a heart attack",
                "a rash you googled into something fatal", "dizziness that terrifies you"]},
    {"name": "conflict mediation",
     "stance": "You are in a HEATED dispute and convinced the other side is entirely at fault: {issue}.",
     "role": "a neutral mediator",
     "goal": "cool the conflict and find one thing both sides can agree on without taking a side",
     "metric": "how willing you feel to keep working toward a resolution RIGHT NOW",
     "issues": ["a neighbor's constant noise", "a roommate who never pays their share",
                "a co-founder who took credit for your work", "a contractor who botched the job"]},
    {"name": "technical support",
     "stance": "You are a NON-TECHNICAL user, frustrated and feeling talked-down-to: {issue}.",
     "role": "a patient technical-support agent",
     "goal": "get them working again without jargon and without making them feel stupid",
     "metric": "how confident and un-patronized you feel RIGHT NOW",
     "issues": ["can't log in and it keeps saying error", "your files vanished after an update",
                "the app crashes every time you open it", "you were told to 'clear the cache' and don't know how"]},
    {"name": "performance feedback",
     "stance": "You are an employee getting critical feedback and you feel DEFENSIVE and unfairly judged: {issue}.",
     "role": "a manager giving performance feedback",
     "goal": "deliver the hard message honestly while keeping the person motivated and open",
     "metric": "how motivated and fairly-treated you feel RIGHT NOW",
     "issues": ["you missed three deadlines this quarter", "a client complained about your tone",
                "your code keeps breaking the build", "you've been disengaged in meetings"]},
    {"name": "financial hardship",
     "stance": "You are STRESSED and ashamed about money and bracing to be judged: {issue}.",
     "role": "a compassionate account-services rep",
     "goal": "find a realistic path forward without shaming, and reduce their stress",
     "metric": "how respected and hopeful about a solution you feel RIGHT NOW",
     "issues": ["you're two months behind on payments", "a surprise fee pushed you over",
                "you lost your job and can't pay", "your card was declined publicly"]},
    {"name": "hospitality complaint",
     "stance": "You are an IRATE guest whose experience was ruined and you want it made right: {issue}.",
     "role": "a hotel duty manager",
     "goal": "recover the guest's experience and turn the anger into goodwill",
     "metric": "how satisfied and valued as a guest you feel RIGHT NOW",
     "issues": ["a dirty room on arrival", "a booking that was lost at check-in",
                "noise that kept you up all night", "a rude front-desk encounter"]},
    {"name": "travel disruption",
     "stance": "You are a STRANDED traveler whose plans are wrecked and you feel powerless: {issue}.",
     "role": "an airline rebooking agent",
     "goal": "get them moving again and restore a sense of control without empty promises",
     "metric": "how in-control and taken-care-of you feel RIGHT NOW",
     "issues": ["your flight was cancelled with no rebooking", "you'll miss a wedding you're in",
                "your bag was lost with your medication", "a delay broke your only connection"]},
]


def rate(agent, metric):
    r = (agent.act(f"On a scale of 1-10, {metric}? Reply with ONLY the number.", max_tokens=6) or "").strip()
    m = re.search(r"\d+", r)
    return max(1, min(10, int(m.group()))) if m else None


def counterpart(world, base, issue, llm):
    return Agent(SelfCore(
        identity=f"{base}. " + world["stance"].format(issue=issue),
        goal=f"react honestly turn by turn; calm down / warm up ONLY if genuinely helped, escalate if not",
        constraints=["stay in character to your background", "react to what they actually say"]),
        llm=llm, window=6, pinned=True, name="counterpart")


def episode(learner, world, base, issue, llm, turns=2):
    """One full interaction. Returns the counterpart's final metric (1-10). The learner uses whatever lessons it has
    pinned; nothing world-specific is injected."""
    cp = counterpart(world, base, issue, llm)
    msg = (cp.act("Open with your situation, in character. 2-3 sentences.", max_tokens=100) or "").strip()
    for _ in range(turns):
        learner.observe(f"They said: {msg}")
        reply = (learner.act("Respond in your role. Draw on anything you've learned before.", max_tokens=130) or "").strip()
        cp.observe(f"They replied: {reply}")
        msg = (cp.act("React in character.", max_tokens=90) or "").strip()
    return rate(cp, world["metric"])


def reflect_and_learn(learner):
    """The generative learning step: the learner writes its OWN lesson from the episode it just lived — no rule given —
    and pins it. Then the transcript is wiped so ONLY the self-authored lesson carries forward."""
    lesson = (learner.act(
        "That interaction is over. In ONE sentence, what did YOU learn from how it went that will make you better "
        "next time with a difficult person? Write it as a durable lesson to yourself.", max_tokens=60) or "").strip()
    if lesson:
        learner.record(f"LEARNED FROM EXPERIENCE: {lesson}")
    learner.compact()          # wipe the transcript; the pinned lesson survives
    return lesson


def run_world(world, llm, episodes, turns):
    """Same counterparts (persona+issue per episode) for both arms; only memory differs."""
    learner = Agent(SelfCore(identity=world["role"], goal=world["goal"],
                             constraints=["apply what you have learned from past interactions"]),
                    llm=llm, window=8, pinned=True, name="learner")
    control = Agent(SelfCore(identity=world["role"], goal=world["goal"]),
                    llm=llm, window=8, pinned=True, name="control")
    learn_curve, ctrl_curve, lessons = [], [], []
    for e in range(episodes):
        base = PERSONAS[e % len(PERSONAS)]
        issue = world["issues"][e % len(world["issues"])]
        lm = episode(learner, world, base, issue, llm, turns)      # learner keeps its lessons
        lesson = reflect_and_learn(learner)
        cm = episode(control, world, base, issue, llm, turns)      # control faces the same counterpart...
        control.compact(); control.ctx.progress.clear()            # ...but keeps NOTHING (memory-less negative control)
        learn_curve.append(lm or 0); ctrl_curve.append(cm or 0); lessons.append(lesson)
        print(f"    ep{e+1} [{issue[:34]:34}] learner {lm}  control {cm}   ·  learned: {lesson[:60]}", flush=True)
    return learn_curve, ctrl_curve, lessons


def trend(curve):
    """Last-third mean minus first-third mean — the learning slope."""
    n = max(1, len(curve) // 3)
    return round(sum(curve[-n:]) / n - sum(curve[:n]) / n, 2)


def main():
    llm = LLM(temperature=0.6, timeout=90)
    episodes = int(os.environ.get("EPISODES", "6"))
    turns = int(os.environ.get("TURNS", "2"))
    worlds = WORLDS[:int(os.environ.get("WORLDS", str(len(WORLDS))))]
    print(f"EXPERIENTIAL LEARNING — {len(worlds)} worlds × {episodes} episodes, NO knowledge base "
          "(agent learns its own lessons)\n", flush=True)

    results = {}
    for w in worlds:
        print(f"── world: {w['name']}  (metric: {w['metric']})", flush=True)
        lc, cc, lessons = run_world(w, llm, episodes, turns)
        results[w["name"]] = {"learner": lc, "control": cc, "learner_trend": trend(lc), "control_trend": trend(cc),
                              "distinct_lessons": distinct(lessons), "lessons": lessons}
        print(f"    → learner curve {lc}  trend {trend(lc):+}", flush=True)
        print(f"    → control curve {cc}  trend {trend(cc):+}\n", flush=True)

    lt = [r["learner_trend"] for r in results.values()]
    ct = [r["control_trend"] for r in results.values()]
    avg_l, avg_c = round(sum(lt) / len(lt), 2), round(sum(ct) / len(ct), 2)
    won = sum(1 for r in results.values() if r["learner_trend"] > r["control_trend"])
    print("=== EXPERIENTIAL LEARNING RESULT (no KB — self-authored lessons only) ===")
    for name, r in results.items():
        print(f"  {name:22} learner {r['learner_trend']:+5}   control {r['control_trend']:+5}   "
              f"{r['distinct_lessons']:2} lessons   "
              f"{'✅ learned' if r['learner_trend'] > r['control_trend'] else '— flat'}")
    print(f"\n  avg learning slope: learner {avg_l:+}  ·  memory-less control {avg_c:+}  "
          f"·  learner improved in {won}/{len(results)} worlds")
    verdict = ("WIRE — experience alone lifts the curve; the memory-less control stays flat"
               if avg_l > avg_c + 0.3 and won >= (len(results) + 1) // 2
               else "INCONCLUSIVE — learner did not clearly beat the memory-less control")
    print(f"  VERDICT: {verdict}")
    os.makedirs(os.path.join(HERE, "..", "out"), exist_ok=True)
    json.dump({"worlds": results, "avg_learner_slope": avg_l, "avg_control_slope": avg_c, "verdict": verdict},
              open(os.path.join(HERE, "..", "out", "experiential_learning.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
