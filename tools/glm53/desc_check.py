#!/usr/bin/env python3
"""_desc_build_kernel COMPACT=True vs the scattered layout: every tier table
must hold exactly the live entries of the scattered table, in pair order,
followed by m_rows = 0. Run inside the serving image (needs vllm + triton):
  python3 desc_check.py /w/ep/moe_w2_cubit.loop.py
"""
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("mwc", sys.argv[1])
mwc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mwc)

E, bad = 288, 0
for pairs, npost_pairs, frac4 in ((64, 40, 0.3), (64, 64, 0.5), (64, 3, 0.0),
                                  (232, 45, 0.25), (64, 0, 0.3), (64, 64, 1.0)):
    g = torch.Generator(device="cuda").manual_seed(pairs + npost_pairs)
    eids = torch.randint(0, E, (pairs,), device="cuda", dtype=torch.int32,
                         generator=g)
    slot = torch.where(torch.rand(E, device="cuda", generator=g) < frac4,
                       torch.arange(E, device="cuda", dtype=torch.int32),
                       torch.full((E,), -1, device="cuda", dtype=torch.int32))
    npost = torch.tensor([npost_pairs * 4], device="cuda", dtype=torch.int32)
    out = {}
    for compact in (False, True):
        d = torch.full((4, pairs, 6), -123, dtype=torch.int64, device="cuda")
        mwc._desc_build_kernel[(1,)](
            eids, npost, slot, d,
            1 << 20, 2 << 20, 3 << 20, 4 << 20, 5 << 20, 6 << 20,
            7 << 30, 8 << 30, 9 << 30, 10 << 30, 11 << 30,
            1000, 100, 2000, 200, 4096, 777,
            4096, 512, 8192, 2048, 256, 8192,
            E, pairs, pairs * 6, 4, BLOCK=256, COMPACT=compact)
        torch.cuda.synchronize()
        out[compact] = d.cpu()
    for gi in range(4):
        sc, cp = out[False][gi], out[True][gi]
        live = sc[sc[:, 5] > 0]
        n = live.shape[0]
        ok = torch.equal(cp[:n], live) and bool((cp[n:, 5] == 0).all())
        bad += not ok
        print(f"pairs={pairs} live={npost_pairs} fp4={frac4} table {gi}: "
              f"{n} live -> {'OK' if ok else 'FAIL'}")
print("DESC OK" if not bad else f"DESC {bad} FAILURES")
