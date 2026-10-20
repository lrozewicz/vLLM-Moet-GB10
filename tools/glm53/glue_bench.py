#!/usr/bin/env python3
"""MoE decode glue: production ops vs the fused kernels of moe_w2_cubit.fuse.py.
Bit-compares a1/as1 (valid slots), the seen flags and a2/as2, and times both
in a CUDA graph. Run in the serving image, server STOPPED:
  python3 glue_bench.py /w/moe_w2_cubit.fuse.py [T]"""
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("mwc", sys.argv[1])
mwc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mwc)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size)

T = int(sys.argv[2]) if len(sys.argv) > 2 else 8
H, I, E, TOPK, MB, LIMIT = 4096, 2048, 288, 8, 4, 7.0
torch.manual_seed(T)
dev = "cuda"


def bench(fn, it=400):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(20):
            fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it // 20):
        gr.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000


x = (torch.randn(T, H, device=dev) * torch.rand(T, 1, device=dev) * 3).to(torch.bfloat16)
ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
sorted_ids, eb, npost = moe_align_block_size(ids, MB, E)
slots = sorted_ids.numel()
rows_cap = slots + 4
xq = torch.zeros(rows_cap, H, dtype=torch.float8_e4m3fn, device=dev)
xs = torch.zeros(rows_cap, H // 32, device=dev)
pad_row = rows_cap - 1


def mk():
    return (torch.zeros(slots, H, dtype=torch.float8_e4m3fn, device=dev),
            torch.zeros(slots, H // 32, device=dev),
            torch.zeros(E, dtype=torch.int32, device=dev))


a_o, s_o, seen_o = mk()
a_n, s_n, seen_n = mk()


def old():
    mwc._quant_a32(x, xq[:T], xs[:T], 128)
    valid = sorted_ids < T * TOPK
    rows = torch.where(valid, sorted_ids // TOPK, torch.full_like(sorted_ids, pad_row))
    torch.index_select(xq.view(torch.uint8), 0, rows, out=a_o.view(torch.uint8))
    torch.index_select(xs, 0, rows, out=s_o)
    seen_o.index_fill_(0, ids.view(-1).long(), 1)


def new():
    mwc._gather_quant_kernel[(slots, H // 128)](
        x, sorted_ids, a_n, s_n, seen_n, eb, T * TOPK, TOPK, x.stride(0),
        a_n.stride(0), s_n.stride(0), MBLOCK=MB, G=128, MARK=True, num_warps=1)


old()
new()
torch.cuda.synchronize()
valid = (sorted_ids < T * TOPK)
v = valid.nonzero().squeeze(1)
ok_a = torch.equal(a_o[v].view(torch.uint8), a_n[v].view(torch.uint8))
ok_s = torch.equal(s_o[v], s_n[v])
ok_seen = torch.equal(seen_o, seen_n)
print(f"T={T} slots={slots} valid={v.numel()}: a1 {ok_a} as1 {ok_s} seen {ok_seen}")
t_old, t_new = bench(old), bench(new)
print(f"  gather+quant+seen: old {t_old:6.1f} us new {t_new:6.1f} us  -> x42 saves "
      f"{(t_old - t_new) * 42 / 1000:.2f} ms/step")

c13 = (torch.randn(slots, 2 * I, device=dev) * 2).to(torch.bfloat16)
act = torch.empty(slots, I, dtype=torch.bfloat16, device=dev)
a2o = torch.zeros(slots, I, dtype=torch.float8_e4m3fn, device=dev)
s2o = torch.zeros(slots, I // 32, device=dev)
a2n, s2n = torch.zeros_like(a2o), torch.zeros_like(s2o)


def old2():
    mwc._silu_and_mul_clamp_fp32(act, c13, LIMIT)
    mwc._quant_a32(act, a2o, s2o, 128)


def new2():
    mwc._silu_quant_kernel[(slots, I // 128)](
        c13, a2n, s2n, I, c13.stride(0), a2n.stride(0), s2n.stride(0), LIMIT,
        G=128, num_warps=1)


old2()
new2()
torch.cuda.synchronize()
ok2a = torch.equal(a2o.view(torch.uint8), a2n.view(torch.uint8))
ok2s = torch.equal(s2o, s2n)
nd = int((a2o.view(torch.uint8) != a2n.view(torch.uint8)).sum())
print(f"  silu+quant: a2 {ok2a} ({nd} bytes differ) as2 {ok2s}")
t_old2, t_new2 = bench(old2), bench(new2)
print(f"  silu+quant: old {t_old2:6.1f} us new {t_new2:6.1f} us  -> x42 saves "
      f"{(t_old2 - t_new2) * 42 / 1000:.2f} ms/step")
print("GLUE BITEXACT" if all((ok_a, ok_s, ok_seen, ok2a, ok2s)) else "GLUE MISMATCH")
