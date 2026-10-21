#!/usr/bin/env python3
"""How much of the KDA spec-decode recurrent kernel is the per-token state
write? GLM-5.3-Flash: 64 heads x 128 x 128 fp32 state (4 MiB per layer per
request); the spec path stores the state after every draft token (k + 1 =
8 slots) so the next step can start from whichever position was accepted.

  (a) production: T = 8 tokens, 8 state slots written
  (b) same, only the last slot valid (index 0 = NULL skips the store)
  (c) T = 16 tokens, only the last slot written - the shape of a "replay the
      accepted tokens of the previous step, then this step's" kernel that
      commits one state per step
Server STOPPED, serving image:  python3 kda_state_bench.py"""
import torch

from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

H, D = 64, 128
dev = "cuda"
torch.manual_seed(0)
n_slots = 4096 // 4  # 1024 state slots = 4 GiB
state = torch.randn(n_slots, H, D, D, device=dev) * 0.01
a_log = torch.randn(H, device=dev) * 0.1
g_bias = torch.randn(H * D, device=dev) * 0.1


def run(T, valid):
    q = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    acc = torch.ones(1, dtype=torch.int32, device=dev)
    sets = []
    for r in range(16):  # rotate slots so the state comes from DRAM
        base = 1 + (r * 64) % (n_slots - 64)
        idx = torch.arange(base, base + T, dtype=torch.int32, device=dev).view(1, T)
        if valid == "last":
            idx[:, :-1] = 0
            idx[0, 0] = base  # the initial state is read from slot [num_accepted - 1] = 0
            idx[0, -1] = base + 1
        sets.append(idx)

    def f(i):
        fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                            use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                            ssm_state_indices=sets[i % 16], num_accepted_tokens=acc,
                            sigmoid_beta=True, a_log=a_log, g_bias=g_bias,
                            compute_gate=True, lower_bound=-5.0)
    for i in range(10):
        f(i)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for i in range(200):
        f(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 200 * 1000


ta = run(8, "all")
tb = run(8, "last")
tc = run(16, "last")
print(f"(a) T=8, 8 state writes: {ta:6.1f} us  ({(9 * 4) / ta * 1e3 * 1.048576:.0f} GB/s for 36 MiB)")
print(f"(b) T=8, 1 state write : {tb:6.1f} us")
print(f"(c) T=16, 1 state write: {tc:6.1f} us  (replay + step, upper bound of the commit scheme)")
print(f"per step at 31 KDA layers: (a) {ta * 31 / 1000:.2f} ms, (c) {tc * 31 / 1000:.2f} ms "
      f"-> ceiling of the saving {(ta - tc) * 31 / 1000:.2f} ms")
