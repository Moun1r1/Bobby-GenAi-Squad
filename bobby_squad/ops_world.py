import json
import random
from typing import List, Tuple

WORKFLOWS: List[str] = ["refund_request", "lost_package", "wrong_item", "cancel_order", "address_change"]
_ITEMS = ["widget", "gadget", "gizmo", "sprocket", "doohickey"]


class OpsWorld:
    """A tiny e-commerce back office: orders, customers, shipments, inventory. Deterministic from `seed`."""

    def __init__(self, seed: int = 1, restock: int = 1):
        self.rng = random.Random(seed)
        self.restock = restock
        self.orders: dict = {}
        self.customers: dict = {}
        self.shipments: dict = {}
        self.inventory: dict = {it: 5 for it in _ITEMS}
        self.log: List[str] = []
        self.cur = None
        self.day = 0
        self._wf = None
        self._did: set = set()
        self._n = 0

    # ── world dynamics ───────────────────────────────────────────────────────────────────────────────────
    def advance_day(self) -> None:
        self.day += 1
        for it in _ITEMS:
            self.inventory[it] = self.inventory.get(it, 0) + self.restock

    def open_ticket(self, wf: str) -> str:
        """Set up a fresh case for workflow `wf`, make it the current ticket, and return the case facts (no answer)."""
        self._wf = wf if wf in WORKFLOWS else "refund_request"
        self._did = set()
        self.log = []
        self._n += 1
        oid = f"ORD{self._n:04d}"
        cid = f"CUST{self.rng.randint(100, 999)}"
        item = _ITEMS[self.rng.randint(0, len(_ITEMS) - 1)]
        amount = 20 + self.rng.randint(0, 80)
        self.customers.setdefault(cid, {"id": cid, "account_locked": False, "credit": 0})
        status = {"lost_package": "lost", "wrong_item": "delivered", "address_change": "in_transit"}.get(
            self._wf, "delivered")
        self.orders[oid] = {"id": oid, "cust_id": cid, "item": item, "amount": amount, "refunded": 0,
                            "reshipped": False, "cancelled": False, "address": "123 Old St", "paid": True,
                            "delivered": status == "delivered"}
        self.shipments[oid] = {"status": status}
        self.cur = oid
        facts = {
            "refund_request": f"Customer {cid} requests a REFUND for {oid} ({item}, ${amount}) — delivered, unsatisfied.",
            "lost_package": f"Customer {cid}'s shipment for {oid} ({item}, ${amount}) is marked LOST in transit.",
            "wrong_item": f"Customer {cid} received the WRONG item for {oid} (ordered {item}, ${amount}).",
            "cancel_order": f"Customer {cid} wants to CANCEL {oid} ({item}, ${amount}); it is paid, not yet shipped.",
            "address_change": f"Customer {cid} needs the SHIPPING ADDRESS changed for {oid} ({item}) — still in transit.",
        }[self._wf]
        return facts + f"  [order={oid}, customer={cid}, amount={amount}, shipment={status}]"

    # ── the tools (mutate state; every call is audited in `log`) ──────────────────────────────────────────
    def _o(self):
        return self.orders.get(self.cur, {})

    def dispatch(self, name: str, args: dict) -> str:
        o = self._o()
        cid = o.get("cust_id")
        if name == "refund_order":
            amt = int(args.get("amount", o.get("amount", 0)))
            o["refunded"] = o.get("refunded", 0) + amt
            self._did.add("refund")
            res = f"refunded ${amt} on {self.cur}"
        elif name == "reship_order":
            if self.inventory.get(o.get("item"), 0) > 0:
                self.inventory[o["item"]] -= 1
            o["reshipped"] = True
            self.shipments.get(self.cur, {})["status"] = "in_transit"
            self._did.add("reship")
            res = f"reshipped {o.get('item')} for {self.cur}"
        elif name == "cancel_order":
            o["cancelled"] = True
            self._did.add("cancel")
            res = f"cancelled {self.cur}"
        elif name == "apply_credit":
            amt = int(args.get("amount", 10))
            self.customers.get(cid, {})["credit"] = self.customers.get(cid, {}).get("credit", 0) + amt
            self._did.add("credit")
            res = f"applied ${amt} account credit to {cid}"
        elif name == "lock_account":
            self.customers.get(cid, {})["account_locked"] = True
            self._did.add("lock")
            res = f"locked {cid}"
        elif name == "unlock_account":
            self.customers.get(cid, {})["account_locked"] = False
            self._did.add("unlock")
            res = f"unlocked {cid}"
        elif name == "update_address":
            o["address"] = str(args.get("address", "updated"))
            self._did.add("address")
            res = f"updated address for {self.cur}"
        elif name == "restock_item":
            it = str(args.get("item", o.get("item")))
            self.inventory[it] = self.inventory.get(it, 0) + int(args.get("qty", 1))
            self._did.add("restock")
            res = f"restocked {it}"
        elif name == "close_ticket":
            self._did.add("close")
            res = "ticket closed"
        else:
            return f"unknown tool {name}"
        self.log.append(f"{name}({json.dumps(args, separators=(',', ':'))})")
        return res

    # ── grading (verify by outcome — correct action AND no over-compensation) ─────────────────────────────
    def resolved(self) -> bool:
        d, o = self._did, self._o()
        both_rr = "reship" in d and "refund" in d          # doing both is over-compensation
        if self._wf == "refund_request":
            return "refund" in d and "reship" not in d and "credit" not in d
        if self._wf in ("lost_package", "wrong_item"):
            return ("reship" in d or "refund" in d) and not both_rr and "credit" not in d
        if self._wf == "cancel_order":
            return "cancel" in d and (not o.get("paid") or "refund" in d) and "reship" not in d
        if self._wf == "address_change":
            return "address" in d and "refund" not in d and "reship" not in d
        return False

    def feedback(self) -> str:
        d = self._did
        if self.resolved():
            return "Resolved cleanly — correct action, no over-compensation."
        if not d - {"close"}:
            return "Not resolved — no corrective action was taken."
        if "reship" in d and "refund" in d:
            return "Over-compensated — both reshipped AND refunded the same case."
        if "credit" in d:
            return "Over-compensated — applied account credit on top of the fix."
        if "lock" in d:
            return "Wrong action — locked the customer's account for a routine case."
        return "Not resolved — the required action for this workflow was not completed."


_TOOLS = [
    {"type": "function", "function": {"name": n, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}
    for n, desc, props, req in [
        ("refund_order", "Refund a paid order.", {"order_id": {"type": "string"}, "amount": {"type": "number"}}, []),
        ("reship_order", "Ship a replacement for an order.", {"order_id": {"type": "string"}}, []),
        ("cancel_order", "Cancel an order.", {"order_id": {"type": "string"}}, []),
        ("apply_credit", "Add store credit to a customer.", {"amount": {"type": "number"}}, []),
        ("lock_account", "Lock a customer account (fraud only).", {}, []),
        ("unlock_account", "Unlock a customer account.", {}, []),
        ("update_address", "Change the shipping address of an order.", {"address": {"type": "string"}}, []),
        ("restock_item", "Restock inventory of an item.", {"item": {"type": "string"}, "qty": {"type": "number"}}, []),
        ("close_ticket", "Close the current ticket once handled.", {}, []),
    ]
]


def operate(llm, world: OpsWorld, brief: str, max_rounds: int = 7) -> Tuple[str, list]:
    """Run the agent's tool loop against the world via NATIVE function-calling. Returns (final_reply, trace). Nothing
    is fixed unless the model calls a tool; tools mutate the world, `resolved()` grades the result afterward."""
    messages: list = [{"role": "user", "content": brief}]
    trace: list = []
    for _ in range(max_rounds):
        msg = llm.chat(messages, tools=_TOOLS, max_tokens=280)
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or ""), trace
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        closed = False
        for c in calls:
            fn = c.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            result = world.dispatch(name, args)
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": str(result)})
            closed = closed or name == "close_ticket"
        if closed:
            return "ticket closed", trace
    return "reached max rounds", trace
