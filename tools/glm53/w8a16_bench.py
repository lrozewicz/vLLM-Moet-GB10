"""Tune the W8A16 kernel of deploy/gb10-glm53/patches/glm53_w8a16.py: for each
shape, the best (BN, SPLIT_K, num_warps, num_stages) at M = 8 and 16 against
BF16 F.linear and FP8 Marlin, with the relative error against
F.linear(x, dequant(q, s)). Run it in the serving image, with the server
STOPPED (it needs a few GiB of GPU memory):

    docker run --rm --gpus all --entrypoint python3 -v "$PWD":/w -w /w vllm-moet-glm53:dflash2 \
      tools/glm53/w8a16_bench.py "kda_qkv:24576:4096,kda_o:4096:8192,lm_head:154880:4096"

Results on GB10 (2026-10-20) are the _CFG table in glm53_w8a16.py.
"""
import itertools
import os
import sys

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../deploy/gb10-glm53/patches"))
from glm53_w8a16 import w8a16_small_m as w8a16  # noqa: E402


def quant_block(w, b=128):
    n, k = w.shape
    blk = w.float().view(n // b, b, k // b, b)
    s = blk.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / 448.0
    return (blk / s).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, k), s.view(n // b, k // b).contiguous()


def bench(fn, it=200):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it // 20):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000


torch.manual_seed(0)
CFGS = list(itertools.product((16, 32, 64, 128), (1, 2, 4, 8), (2, 4, 8), (3, 4, 5, 6)))
shapes = [(a, int(b), int(c)) for a, b, c in (x.split(":") for x in sys.argv[1].split(","))]
for name, n, k in shapes:
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    q, s = quant_block(w)
    L = torch.nn.Module()
    L.output_size_per_partition, L.input_size_per_partition = n, k
    L.weight = torch.nn.Parameter(q.clone(), requires_grad=False)
    L.weight_scale_inv = torch.nn.Parameter(s.clone(), requires_grad=False)
    L.weight_block_size = [128, 128]
    L.orig_dtype = torch.bfloat16
    prepare_fp8_layer_for_marlin(L, size_k_first=False)
    wdq = (q.float().view(n // 128, 128, k // 128, 128) * s.view(n // 128, 1, k // 128, 1)).view(n, k).to(torch.bfloat16)
    del w
    for m in (8, 16):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x, wdq).float()
        t_bf = bench(lambda: F.linear(x, wdq))
        t_m = bench(lambda: apply_fp8_marlin_linear(x, L.weight, L.weight_scale_inv, L.workspace, n, k, None))
        best = None
        for cfg in CFGS:
            if n // cfg[0] * cfg[1] < 48 or (k // 128) < cfg[1]:
                continue
            try:
                out = w8a16(x, q, s, cfg)
                err = ((out.float() - ref).norm() / ref.norm()).item()
                if err > 2e-2:
                    print("  BAD", cfg, err)
                    continue
                t = bench(lambda: w8a16(x, q, s, cfg))
            except Exception as e:  # noqa: BLE001
                print("  FAIL", cfg, repr(e)[:120])
                continue
            if best is None or t < best[0]:
                best = (t, cfg, err)
        t, cfg, err = best
        print(f"{name:10} M={m:3}  BF16 {t_bf:7.1f} us  Marlin {t_m:7.1f} us ({n*k/t_m/1e3:4.0f} GB/s)  "
              f"Triton {t:7.1f} us ({n*k/t/1e3:4.0f} GB/s) cfg={cfg} err={err:.1e}  vs Marlin x{t_m/t:4.2f}", flush=True)
    del L, q, wdq
    torch.cuda.empty_cache()
