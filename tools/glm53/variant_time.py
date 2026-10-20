#!/usr/bin/env python3
"""Same-work timing of cubin variants: grid.y = PAIRS, scattered live pairs
(the production launch). Bit-compares every variant against the first.
usage: variant_time.py DIR [DIR ...]   (first = reference, e.g. /cubit-share)
"""
import sys

import torch

sys.path.insert(0, "/w")
import loop_check as lc  # noqa: E402

dirs = sys.argv[1:]
for tier, N, K in (("w2", 4096, 4096), ("w2", 4096, 2048),
                   ("w4", 4096, 4096), ("w4", 4096, 2048)):
    name = f"moe_{tier}_mm_k{K}_a32.cubin"
    fns = [lc.load(f"{d}/{name}", f"moe_{tier}_mm") for d in dirs]
    print(f"== {tier} K={K} regs {[r for _, r in fns]}", flush=True)
    for real in (12, 40, 64):
        case = lc.Case(tier, N, K, real, seed=real * 7 + K)
        cs = [case.c() for _ in fns]
        ds = [case.desc(c, False) for c in cs]
        for (fn, _), d in zip(fns, ds):
            lc.launch(fn, case, d, lc.PAIRS, lc.PAIRS, lc.PAIRS)
        torch.cuda.synchronize()
        same = [torch.equal(cs[0].view(torch.int16), c.view(torch.int16))
                for c in cs[1:]]
        # interleave repeats so clock/thermal drift hits all variants alike
        ts = [[] for _ in fns]
        for _ in range(3):
            for i, ((fn, _), d) in enumerate(zip(fns, ds)):
                ts[i].append(lc.timeit(
                    lambda: lc.launch(fn, case, d, lc.PAIRS, lc.PAIRS, lc.PAIRS),
                    200))
        print(f"  real={real:2d} bitexact {same} | "
              + " | ".join(f"{d.split('/')[-1]} {min(t):7.1f}"
                           for d, t in zip(dirs, ts)), flush=True)
