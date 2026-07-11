"""quickstart — the smallest useful Bobby GenAi Squad program.

Point it at any OpenAI-compatible endpoint first:
    export BOBBY_LLM_URL="http://localhost:8000/v1/chat/completions"
    export BOBBY_LLM_MODEL="your-served-model-id"

Then:  python examples/quickstart.py
"""
from bobby_squad import Agent, SelfCore, LLM, squad_solve

llm = LLM(temperature=0.4)

# 1) One persistent-self agent doing exactly what it's asked — behavior comes from the request, not a script.
solo = Agent(SelfCore(identity="a precise generalist", goal="do exactly what the request asks"), llm=llm)
print("HAIKU:\n", solo.carry_out("Write a 3-line haiku about entropy.", max_rounds=2), "\n")

# 2) A self-organizing squad covering a target — recursive shared board, no orchestrator.
#    Here: two agents enumerate the numbers a small rule-set should produce (a toy 'coverage' task).
TARGET = set(range(1, 21))
agents = [Agent(SelfCore("an enumerator", "list the numbers in the given range"), llm=llm, name=f"a{i}")
          for i in range(2)]

def work(agent, unit):
    lo, hi = unit
    ans = agent.carry_out(f"List every integer from {lo} to {hi}, comma-separated, digits only.", max_rounds=1)
    return {n for n in TARGET if str(n) in ans.replace(",", " ").split()}

def verify(unit, acc):
    lo, hi = unit
    return set(range(lo, hi + 1)) <= acc or (hi - lo) <= 4          # covered, or small enough to accept

def split(unit):
    lo, hi = unit
    mid = (lo + hi) // 2
    return [(lo, mid), (mid + 1, hi)] if (hi - lo) > 4 else None

r = squad_solve(agents, [(1, 20)], work, verify=verify, split=split)
print(f"COVERAGE: {len(r['result'])}/{len(TARGET)} in {r['passes']} passes (squad_solve, no orchestrator)")
