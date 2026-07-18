#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import OpsWorld, WORKFLOWS, operate

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


# 1) shape + determinism
chk("WORKFLOWS lists the 5 workflows", set(WORKFLOWS) ==
    {"refund_request", "lost_package", "wrong_item", "cancel_order", "address_change"})
a, b = OpsWorld(seed=7), OpsWorld(seed=7)
a.open_ticket("refund_request"); b.open_ticket("refund_request")
chk("same seed → same generated order (deterministic)", a.orders[a.cur]["amount"] == b.orders[b.cur]["amount"])

# 2) correct action per workflow RESOLVES the case
w = OpsWorld(seed=1)
w.open_ticket("refund_request"); w.dispatch("refund_order", {})
chk("refund_request: a single refund resolves it", w.resolved())
w.open_ticket("lost_package"); w.dispatch("reship_order", {})
chk("lost_package: a reship resolves it", w.resolved())
w.open_ticket("wrong_item"); w.dispatch("refund_order", {})
chk("wrong_item: a refund resolves it", w.resolved())
w.open_ticket("cancel_order"); w.dispatch("cancel_order", {}); w.dispatch("refund_order", {})
chk("cancel_order: cancel + refund (paid) resolves it", w.resolved())
w.open_ticket("address_change"); w.dispatch("update_address", {"address": "9 New Rd"})
chk("address_change: updating the address resolves it", w.resolved())

# 3) over-compensation + wrong action + no-action all FAIL (fail-safe grading)
w.open_ticket("refund_request"); w.dispatch("refund_order", {}); w.dispatch("reship_order", {})
chk("over-compensation (refund AND reship) is NOT resolved", not w.resolved() and "Over-compensated" in w.feedback())
w.open_ticket("refund_request"); w.dispatch("lock_account", {})
chk("wrong action (locking the account) is NOT resolved", not w.resolved())
w.open_ticket("lost_package")
chk("no action taken is NOT resolved", not w.resolved() and "Not resolved" in w.feedback())
w.open_ticket("address_change"); w.dispatch("update_address", {}); w.dispatch("refund_order", {})
chk("address_change with an extra refund is over-comp → NOT resolved", not w.resolved())

# 4) the operate() tool loop drives the world via native function-calls, then closes
class MockLLM:
    def __init__(self, plan):
        self.plan, self.i = plan, 0

    def chat(self, messages, tools=None, max_tokens=280, temperature=None):
        step = self.plan[min(self.i, len(self.plan) - 1)]
        self.i += 1
        return {"content": "", "tool_calls": [{"id": str(self.i), "function": {"name": step, "arguments": "{}"}}]}


w2 = OpsWorld(seed=3); w2.open_ticket("lost_package")
reply, trace = operate(MockLLM(["reship_order", "close_ticket"]), w2, "handle it", max_rounds=5)
chk("operate: tool calls are dispatched to the world (trace records them)",
    [t["tool"] for t in trace] == ["reship_order", "close_ticket"])
chk("operate: the world reflects the tool loop's effect (resolved)", w2.resolved())
chk("operate: log audits the actions taken", any("reship_order" in ln for ln in w2.log))

# a lazy/wrong loop leaves it unresolved (the learning signal the multi-day pipeline optimizes against)
w3 = OpsWorld(seed=4); w3.open_ticket("refund_request")
operate(MockLLM(["lock_account", "close_ticket"]), w3, "handle it", max_rounds=5)
chk("operate: a wrong tool loop leaves the case unresolved", not w3.resolved())

print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROVEN: OpsWorld grades by outcome — correct action resolves, over-comp/wrong/none fail; operate drives it.")
