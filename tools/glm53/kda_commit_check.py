#!/usr/bin/env python3
"""Patch 16 check: kda_commit (log + replay) against the stock KDA spec kernel.

Runs several chained verify steps for 2 rows with random acceptance and
compares, bit for bit, the verify outputs and the state slot the next step
reads (plus the block-boundary slot the align postprocess copies), for every
replay FMA form (VLLM_KDA_COMMIT_REPLAY). Then times both paths with the
state rotated through DRAM. Server STOPPED, serving image:

  docker run ... -v kda_commit.py:<vllm>/model_executor/layers/mamba/ops/kda_commit.py
  python3 kda_commit_check.py [--slots 256] [--bench]
"""
import argparse

import torch

from vllm.model_executor.layers.mamba.ops.kda_commit import KdaCommit
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

H, D, TM, R = 64, 128, 8, 2
dev = "cuda"


def inputs(T, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=g, dtype=torch.bfloat16)  # noqa: E731
    return r(1, T, H, D), r(1, T, H, D), r(1, T, H, D), r(1, T, H, D), r(1, T, H)


def params(slots):
    torch.manual_seed(0)
    a_log = torch.randn(H, device=dev) * 0.1
    g_bias = torch.randn(H * D, device=dev) * 0.1
    st0 = torch.randn(slots, H, D, D, device=dev) * 0.05
    return a_log, g_bias, st0


def kw(cu, idx, acc, a_log, g_bias):
    return dict(use_qk_l2norm_in_kernel=True, cu_seqlens=cu, ssm_state_indices=idx,
                num_accepted_tokens=acc, sigmoid_beta=True, a_log=a_log,
                g_bias=g_bias, compute_gate=True, lower_bound=-5.0)


def check(slots, steps):
    a_log, g_bias, st0 = params(slots)
    st_ref, st_new = st0.clone(), st0.clone()
    KdaCommit._ptrs, KdaCommit._ptr_table = {}, None
    idx = torch.tensor([[1 + j for j in range(TM)], [9 + j for j in range(TM)]],
                       dtype=torch.int32, device=dev)
    cu = torch.tensor([0, TM, 2 * TM], dtype=torch.int32, device=dev)
    acc = torch.ones(R, dtype=torch.int32, device=dev)
    idx_map = torch.arange(R, dtype=torch.int32, device=dev)
    gen = torch.Generator().manual_seed(1)
    computed = torch.tensor([100, 205], dtype=torch.int32, device=dev)
    bad = 0
    for step in range(steps):
        q, k, v, g, b = inputs(R * TM, 10 + step)
        o_ref, _ = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=b, initial_state=st_ref,
                                       **kw(cu, idx, acc, a_log, g_bias))
        KdaCommit.set_spec_rows([0, 1])
        o_new = KdaCommit.verify(0, q, k, v, g, b, st_new, cu, idx, acc, None,
                                 a_log, g_bias, -5.0)
        new_acc = torch.randint(1, TM + 1, (R,), generator=gen).to(torch.int32).to(dev)
        computed += new_acc
        KdaCommit.materialize(idx_map, new_acc, computed, 16)
        torch.cuda.synchronize()
        eq_o = torch.equal(o_ref, o_new)
        msg = [f"step {step}: out {'==' if eq_o else '!='}"]
        bad += not eq_o
        for r in range(R):
            na = int(new_acc[r])
            s = int(idx[r, na - 1])
            d = (st_ref[s] - st_new[s]).abs().max().item()
            bad += d != 0
            msg.append(f"row{r} acc {na} slot {s} " + ("==" if d == 0 else f"!= {d:.3g}"))
            new_c = int(computed[r])
            run = new_c - na + 1
            al = new_c // 16 * 16
            if al >= run and al - run != na - 1:
                s2 = int(idx[r, al - run])
                eq2 = torch.equal(st_ref[s2], st_new[s2])
                bad += not eq2
                msg.append(f"(boundary slot {s2} {'==' if eq2 else '!='})")
        print("  ".join(msg))
        acc = new_acc
    return bad


def bench(slots):
    a_log, g_bias, st = params(slots)
    KdaCommit._ptrs, KdaCommit._ptr_table = {}, None
    n = (slots - 1) // TM
    q, k, v, g, b = inputs(TM, 99)
    cu1 = torch.tensor([0, TM], dtype=torch.int32, device=dev)
    acc1 = torch.full((1,), 5, dtype=torch.int32, device=dev)
    idx_map = torch.zeros(1, dtype=torch.int32, device=dev)
    computed = torch.full((1,), 100, dtype=torch.int32, device=dev)
    rows = [torch.arange(1 + i * TM, 1 + (i + 1) * TM, dtype=torch.int32,
                         device=dev).view(1, TM) for i in range(n)]

    def ref(i):
        fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=b, initial_state=st,
                            **kw(cu1, rows[i % n], acc1, a_log, g_bias))

    def verify_only(i):
        KdaCommit.verify(0, q, k, v, g, b, st, cu1, rows[i % n], acc1, None,
                         a_log, g_bias, -5.0)

    def new(i):
        KdaCommit.set_spec_rows([0])
        verify_only(i)
        KdaCommit.materialize(idx_map, acc1, computed, 0)

    for name, f in (("stock (8 state stores)", ref), ("verify + commit", new),
                    ("verify only", verify_only)):
        for i in range(10):
            f(i)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for i in range(200):
            f(i)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) / 200 * 1000
        print(f"{name:24s} {us:7.1f} us/layer  x31 = {us * 31 / 1000:.2f} ms/step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--modes", default="0,1,2")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    KdaCommit.num_layers = 1
    KdaCommit.set_max_rows(R)
    for mode in (int(m) for m in a.modes.split(",")):
        KdaCommit.replay_mode = mode
        print(f"--- replay mode {mode}")
        bad = check(a.slots, a.steps)
        print(f"BIT-EXACT mode {mode}" if bad == 0 else f"MISMATCHES mode {mode}: {bad}")
    if a.bench:
        bench(a.slots)


if __name__ == "__main__":
    main()
