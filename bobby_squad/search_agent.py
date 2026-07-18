from .planning import extract_json


class HypothesisSearcher:
    def __init__(self, llm, search_fn, experiment_fn=None, patience=4, min_rounds=2):
        self.llm = llm                      # an LLM callable
        self.search = search_fn             # (query, k) -> [memory strings]   (the world's real knowledge)
        self.experiment = experiment_fn     # (concept) -> {degradation, trials, ...}  — a real intervention IN the world
        self.patience = patience
        self.min_rounds = min_rounds

    def _ablation_target(self, claim, evidence):
        o = extract_json(self.llm([{"role": "user", "content":
            f"Claim:\n  {claim}\nEvidence:\n- " + "\n- ".join(evidence[:6]) +
            "\n\nName the ONE concept whose memory, if ablated from the world, would most test this claim. "
            'ONLY JSON: {"concept":"<concept>"}'}], max_tokens=40))
        return (o.get("concept") or "").strip()

    def _probe_query(self, claim):
        o = extract_json(self.llm([{"role": "user", "content":
            f"You are a skeptic trying to REFUTE this claim:\n  {claim}\n\nWrite ONE search query over a technical "
            "knowledge base whose results would most likely CONTRADICT the claim or expose it as already-known / "
            'not-novel. ONLY JSON: {"query":"<query>"}'}], max_tokens=60))
        return (o.get("query") or claim).strip()

    def _judge(self, claim, evidence):
        o = extract_json(self.llm([{"role": "user", "content":
            f"Claim under test:\n  {claim}\n\nEvidence retrieved from the knowledge world:\n- " +
            "\n- ".join(evidence[:8]) +
            "\n\nJudge SKEPTICALLY — default to REFUTED unless the evidence positively supports the claim. Decide:\n"
            "  verdict: SUPPORTED | REFUTED | INCONCLUSIVE\n  novel: true if the idea is NOT already present in the "
            "evidence\n  stable: true if this verdict would not change with more of the same evidence\n"
            "  gain_proof: one line naming what a real experiment would have to MEASURE to actually prove it\n"
            'ONLY JSON: {"verdict":"..","novel":true,"stable":true,"why":"..","gain_proof":".."}'}], max_tokens=200))
        return o

    def test(self, hypothesis):
        """Patiently test one hypothesis. Returns a grounded verdict + the probe trace + the required gain-proof."""
        claim = f"{hypothesis.get('name') or hypothesis.get('primitive')}: {hypothesis.get('mechanism', '')}".strip()
        evidence, trace, j, exp = [], [], {}, None
        for rnd in range(self.patience):
            evidence += [h for h in self.search(self._probe_query(claim), 5) if h not in evidence]
            if self.experiment and exp is None:                # run a CONTROLLED in-world ablation proof
                c = self._ablation_target(claim, evidence)
                if c:
                    exp = self.experiment(c)
                    evidence.append(f"[controlled in-world ablation of '{c}'] privileged-node degradation "
                                    f"{exp.get('treatment', 0)} vs background-node control {exp.get('control', 0)} "
                                    f"→ effect {exp.get('effect', 0)} (sign-stable {exp.get('sign_stable', 0)}, "
                                    f"{exp.get('trials', 0)} trials) → PROVEN={exp.get('proven', False)}")
            j = self._judge(claim, evidence)
            trace.append(j.get("verdict", "INCONCLUSIVE"))
            if j.get("stable") and rnd + 1 >= self.min_rounds:
                break
        # the DATA decides the top verdict: a controlled proof outranks the LLM's read
        data_proven = bool(exp and exp.get("proven"))
        verdict = "PROVEN-IN-WORLD" if data_proven else (trace[-1] if trace else "INCONCLUSIVE")
        return {"claim": claim, "verdict": verdict, "data_proven": data_proven,
                "novel": bool(j.get("novel")), "rounds": len(trace), "trace": trace, "experiment": exp,
                "why": j.get("why", ""), "gain_proof": j.get("gain_proof", ""), "evidence": evidence[:6]}
