#!/usr/bin/env python3
"""VLLM_SPEC_ADAPTIVE=1: pick the draft depth per step from live acceptance.

Rides on vLLM's dynamic speculative decoding (the scheduler already passes a
per-step `num_spec_tokens_to_schedule` to the drafter); upstream only keys it
on batch size. Here each request keeps an EMA estimate of p, the chance that
a draft token is accepted given the previous one was (geometric model,
censored correctly: a step with n accepted out of k is n successes and, if
n < k, one failure). The step picks the k from VLLM_SPEC_ADAPTIVE_KS that
maximizes expected tokens per unit time,

    E(k) = (1 - p^(k+1)) / (1 - p),    T(k) = T0 + c * k,

with T0, c from VLLM_SPEC_ADAPTIVE_COST (ms; fitted to measured step times).
The step shares one k across the batch and verifies N * (k + 1) tokens, so
with N streams it maximizes the batch's expected tokens per unit time,

    sum_i E_i(k) / (T0 + c * (N * (k + 1) - 1)),

which is the formula above at N = 1. This cost fits the measured code-decode
rates at N = 1, 2, 4 (45.0, 29.4, 20.0 tok/s per stream at k = 7) within 7%;
VLLM_SPEC_ADAPTIVE_BATCH=0 keeps the single-stream cost and the batch's mean p.
The idea is Strata's cost-driven draft controller. Needs
num_speculative_tokens_per_batch_size in --speculative-config, which turns
the dynamic path on. The V1 runner then forces PIECEWISE cudagraphs; keeping
FULL_AND_PIECEWISE instead was measured worse (Polish prose 15.7 vs 22.8
tok/s): the shorter verify steps lose their graphs.

GLM-5.3-Flash (patch 15): the same controller on the V2 runner, which keeps
FULL decode cudagraphs - it captures one uniform-decode graph per k listed in
num_speculative_tokens_per_batch_size, so VLLM_SPEC_ADAPTIVE_KS must only
name k values that table lists (boot.sh builds both from ADAPTIVE_KS).
Ported from deploy/gb10-mimo/patches/patch_adaptive_k.py.

  python3 patch_adaptive_k.py <scheduler.py>
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
if "VLLM_SPEC_ADAPTIVE" in s:
    sys.exit(0)

# 1) controller state, next to the upstream lookup
old = '''        self.dynamic_sd_lookup: list[int] | None = None
'''
new = '''        self.dynamic_sd_lookup: list[int] | None = None
        self._adapt_k = _AdaptiveK.from_env()
'''
assert s.count(old) == 1, "dynamic_sd_lookup init anchor moved"
s = s.replace(old, new)

# 2) the per-step choice
old = '''            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
'''
new = '''            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
            if self._adapt_k is not None:
                num_spec_tokens_to_schedule = min(
                    num_spec_tokens_to_schedule,
                    self._adapt_k.choose(num_scheduled_tokens.keys()),
                )
'''
assert s.count(old) == 1, "dynamic_sd_lookup use anchor moved"
s = s.replace(old, new)

# 3) observe acceptance
old = '''                num_rejected = num_draft_tokens - num_accepted
'''
new = '''                num_rejected = num_draft_tokens - num_accepted
                if self._adapt_k is not None and not output_is_stale:
                    self._adapt_k.observe(req_id, num_draft_tokens, num_accepted)
'''
assert s.count(old) == 1, "num_rejected anchor moved"
s = s.replace(old, new)

# 4) the controller itself, appended at module level
s += '''

class _AdaptiveK:
    """Per-request EMA of the conditional draft acceptance p -> best k."""

    def __init__(self, ks, t0, c, decay, batch):
        self.ks = sorted(ks)
        self.t0, self.c, self.decay, self.batch = t0, c, decay, batch
        self.state: dict[str, list[float]] = {}  # req_id -> [succ, fail]
        self.prior = [3.0, 1.0]  # carried over from recent requests
        self.hist = dict.fromkeys(self.ks, 0)

    @classmethod
    def from_env(cls):
        import os

        if os.environ.get("VLLM_SPEC_ADAPTIVE", "0") != "1":
            return None
        ks = [int(x) for x in os.environ.get("VLLM_SPEC_ADAPTIVE_KS", "1,3,7").split(",")]
        t0, c = (
            float(x) for x in os.environ.get("VLLM_SPEC_ADAPTIVE_COST", "70,5.3").split(",")
        )
        decay = float(os.environ.get("VLLM_SPEC_ADAPTIVE_DECAY", "0.8"))
        batch = os.environ.get("VLLM_SPEC_ADAPTIVE_BATCH", "1") == "1"
        logger.info("adaptive draft depth: k in %s, T(k) = %.1f + %.2f k, decay %.2f, "
                    "batch-aware %s", ks, t0, c, decay, batch)
        return cls(ks, t0, c, decay, batch)

    def observe(self, req_id, k, n):
        st = self.state.get(req_id)
        if st is None:
            if len(self.state) > 256:
                self.state.clear()
            st = self.state[req_id] = list(self.prior)
        d = self.decay
        st[0] = d * st[0] + n
        st[1] = d * st[1] + (1.0 if n < k else 0.0)
        self.prior = list(st)

    def p_of(self, req_id):
        st = self.state.get(req_id, self.prior)
        return (st[0] + 0.5) / (st[0] + st[1] + 1.0)

    @staticmethod
    def expected(p, k):
        return (k + 1) if p >= 0.9999 else (1 - p ** (k + 1)) / (1 - p)

    def choose(self, req_ids):
        ps = [self.p_of(r) for r in req_ids]
        if not ps:
            return self.ks[-1]
        p = sum(ps) / len(ps)
        n = len(ps) if self.batch else 1
        best, best_rate = self.ks[-1], -1.0
        for k in self.ks:
            if self.batch:
                e = sum(self.expected(q, k) for q in ps)
            else:
                e = self.expected(p, k)
            rate = e / (self.t0 + self.c * (n * (k + 1) - 1))
            if rate > best_rate:
                best, best_rate = k, rate
        self.hist[best] += 1
        if sum(self.hist.values()) % 500 == 0:
            logger.info("adaptive draft depth: steps per k %s (p now %.2f)", self.hist, p)
        return best
'''
p.write_text(s)
compile(s, str(p), "exec")
print("scheduler: VLLM_SPEC_ADAPTIVE controller added")
