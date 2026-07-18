#!/usr/bin/env python3
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import self_improve_engine as SI
from bobby_squad import Engine, LLM, SelfCore, Agent
from bobby_squad.agent_tools import SandboxTools
from bobby_squad.retrieval import default_embed

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


# ── set up the flywheel's sandbox + a thinking-off researcher (fast) ──
sbx = tempfile.mkdtemp(prefix="fe_")
shutil.copy(os.path.join(HERE, "memory_gains.py"), os.path.join(sbx, "memory_gains.py"))
with open(os.path.join(sbx, "smoke.py"), "w") as f:
    f.write(SI.SMOKE)
tools = SandboxTools(repo_root=os.path.join(SI.PUBLISHED, "bobby_squad"), sandbox_root=sbx, timeout=120)
researcher = Agent(SelfCore(identity="a memory-engine researcher who writes and self-repairs retrieval methods",
                            goal="write a Candidate(Method) that builds and answers"),
                   llm=LLM(temperature=0.6, extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
                   tools=tools)

# ── the LLM invents + self-repairs a real method ──
mech, method = SI.generate_candidate(researcher, tools, sbx)
chk("LLM wrote a BUILDING memory method (invent → self-repair)", method is not None)

if method is not None:
    # ── promote the LLM-generated method into the kernel as a frozen plugin ──
    code = open(os.path.join(sbx, "cand.py")).read()
    promoted = SI.promote_method(eng := Engine(tempfile.mkdtemp(prefix="feng_")), "flywheel_cand1", method, code,
                                 proof={"verdict": "WIRE", "dF1": 3.0}, tags=("memqa",),
                                 embed_fn=default_embed,
                                 competence_qs=["Who wrote the first algorithm?", "Who was Ada Lovelace's father?"])
    chk("promoted the LLM-generated method into the engine registry", promoted)

    # ── a query routes to the LLM-generated frozen plugin and it answers from a document ──
    r = eng.emit("memqa", {"cap": "memqa", "q": "Who wrote the first algorithm?",
                           "ctx": "Ada Lovelace, daughter of Lord Byron, wrote the first algorithm in 1843 for the "
                                  "Analytical Engine."})
    chk("engine routed the query to the promoted plugin, which answered", isinstance(r, str) and len(r) > 0)
    chk("SKILL_PROMOTED recorded on the spine", any(e.kind == "SKILL_PROMOTED" for e in eng.log.read()))
    print("  mechanism:", mech[:80])
    print("  answer:", repr((r or "")[:80]))

shutil.rmtree(sbx, ignore_errors=True)
print("\n== %d PASS / %d FAIL ==" % (len(ok), len(bad)))
sys.exit(1 if bad else 0)
