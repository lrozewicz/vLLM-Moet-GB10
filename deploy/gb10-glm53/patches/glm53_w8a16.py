# SPDX-License-Identifier: Apache-2.0
"""Block-128 FP8 W8A16 for the BF16 linears of GLM-5.3-Flash on GB10.

The GLM-5.3-Flash FP8 release keeps ~11 GB of its weights in BF16: the q/k/v
(fused into in_proj_qkvbfg_a) and o_proj of the 34 KDA layers and the
lm_head; the NoPE overlay additionally dequantizes the MLA q_b_proj / o_proj
of the 11 full-attention layers to BF16 on load, and the DFlash2 drafter is
BF16 throughout. Decode on GB10 is bandwidth bound, and these GEMMs were 46%
of a decode step (nsys, 2026-10-19).

VLLM_GLM53_W8A16_RE=<regex> selects modules by name (see glm53_w8a16_select,
called from model_loader/utils.py:process_weights_after_loading for the
target model and the drafter). Each one is requantized after load to
block-128 FP8 e4m3 (the checkpoint's own format) and kept as ONE plain
row-major copy:

* small batches (decode, speculative verify; M < VLLM_GLM53_W8A16_PREFILL_M)
  run the Triton kernel below: BF16 activations, FP32 accumulation, at the
  same ~230 GB/s as FP8 Marlin, which would need a second, repacked copy;
* larger batches (prefill) dequantize the weight to BF16 (bit-exact, one
  kernel) and run the stock BF16 GEMM - a W8A16 GEMM is 3x slower there.

Activations stay BF16 (W8A16, not W8A8): community runs found BF16
activations matter for long-context recall (local-inference-lab vllm#1027).
The KDA gate rows of in_proj_qkvbfg_a (b, f_a, g_a) stay BF16 for the same
reason, as a second small GEMM.
"""
import os
import re

import torch
import torch.nn.functional as F
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

PREFILL_M = int(os.getenv("VLLM_GLM53_W8A16_PREFILL_M", "128"))


def quant_block128(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    blk = w.float().view(n // 128, 128, k // 128, 128)
    scale = blk.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / 448.0
    q = (blk / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(n, k), scale.view(n // 128, k // 128).contiguous()


@triton.jit
def _w8a16_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, K, n_kb, stride_xm, stride_ym,
                  BM: tl.constexpr, BN: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    m_mask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    s_row = s_ptr + (pid_n * BN // 128) * n_kb
    for kb in range(pid_k, n_kb, SPLIT_K):
        k0 = kb * 128
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :],
                    mask=m_mask[:, None], other=0.0)
        w = tl.load(q_ptr + offs_n[:, None] * K + (k0 + offs_k)[None, :])
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * tl.load(s_row + kb)
    y = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        tl.atomic_add(y, acc, mask=m_mask[:, None], sem="relaxed")


# (BN, SPLIT_K, num_warps, num_stages) per (N, K), tuned on GB10 at M = 8 and
# 16 (tools/glm53/w8a16_bench.py); every entry is within 2% of FP8 Marlin or
# faster.
_CFG: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (24576, 4096): (32, 1, 4, 4),    # KDA q|k|v, drafter gate_up
    (4096, 8192): (32, 1, 4, 5),     # KDA o_proj
    (154880, 4096): (32, 1, 4, 4),   # lm_head
    (4096, 16384): (32, 1, 4, 5),    # MLA o_proj
    (16384, 1536): (16, 1, 4, 3),    # MLA q_b_proj
    (6144, 4096): (32, 1, 4, 5),     # drafter qkv
    (4096, 4096): (64, 2, 4, 3),     # drafter o_proj
    (4096, 12288): (32, 8, 4, 6),    # drafter down_proj
    (4096, 20480): (32, 8, 2, 5),    # drafter fc
}


def w8a16_small_m(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor,
                  cfg: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    m, k = x.shape
    n = q.shape[0]
    bn, split_k, warps, stages = cfg or _CFG.get((n, k), (32, 1, 4, 4))
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    if split_k == 1:
        y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    else:
        y = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    _w8a16_kernel[(n // bn, split_k, triton.cdiv(m, bm))](
        x, q, s, y, m, k, k // 128, x.stride(0), y.stride(0),
        BM=bm, BN=bn, SPLIT_K=split_k, num_warps=warps, num_stages=stages)
    return y if split_k == 1 else y.to(x.dtype)


@triton.jit
def _dequant_kernel(q_ptr, s_ptr, o_ptr, n_kb, K):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    r = pn * 128 + tl.arange(0, 128)[:, None]
    c = pk * 128 + tl.arange(0, 128)[None, :]
    off = r * K + c
    v = tl.load(q_ptr + off).to(tl.float32) * tl.load(s_ptr + pn * n_kb + pk)
    tl.store(o_ptr + off, v.to(tl.bfloat16))


def dequant_bf16(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    out = torch.empty((n, k), dtype=torch.bfloat16, device=q.device)
    _dequant_kernel[(n // 128, k // 128)](q, s, out, k // 128, k, num_warps=8)
    return out


def _glm53_w8a16_linear(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor,
                        prefill_m: int) -> torch.Tensor:
    if x.size(0) >= prefill_m:
        return F.linear(x, dequant_bf16(q, s))
    return w8a16_small_m(x, q, s)


def _glm53_w8a16_linear_fake(x, q, s, prefill_m):
    return x.new_empty((x.size(0), q.size(0)))


direct_register_custom_op(
    op_name="glm53_w8a16_linear",
    op_func=_glm53_w8a16_linear,
    mutates_args=[],
    fake_impl=_glm53_w8a16_linear_fake,
)


class Glm53W8A16Method(LinearMethodBase):
    """Takes over a BF16 linear (or lm_head) whose weight is already loaded:
    process_weights_after_loading quantizes it and frees the BF16 matrix.
    The first fp8_rows output rows go to FP8, the rest stay BF16."""

    def __init__(self, base, fp8_rows: int | None = None) -> None:
        self.base = base
        self.fp8_rows = fp8_rows

    def create_weights(self, *args, **kwargs):
        return self.base.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        w = layer.weight.data
        n_all, k = w.shape
        rows = self.fp8_rows or n_all
        assert rows % 128 == 0 and k % 128 == 0, (rows, k)
        q, scale = quant_block128(w[:rows])
        tail = w[rows:].clone() if rows < n_all else None
        layer.w8a16_weight = q
        layer.w8a16_scale = scale
        layer.w8a16_bf16_tail = None if tail is None else nn.Parameter(tail, requires_grad=False)
        # keep a 1-element placeholder for code that only inspects dtype/device
        layer.weight = nn.Parameter(w.new_empty(1), requires_grad=False)

    def apply(self, layer: nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = torch.ops.vllm.glm53_w8a16_linear(x2, layer.w8a16_weight, layer.w8a16_scale, PREFILL_M)
        if layer.w8a16_bf16_tail is not None:
            y = torch.cat([y, F.linear(x2, layer.w8a16_bf16_tail)], dim=-1)
        if bias is not None:
            y = y + bias
        return y.reshape(*shape[:-1], y.shape[-1])


def _fp8_rows(name: str, module: nn.Module, n: int) -> int | None:
    if name.endswith("in_proj_qkvbfg_a"):
        # q | k | v -> FP8; b | f_a | g_a (the KDA gates) stay BF16
        return sum(module.output_partition_sizes[:3])
    return n


def glm53_w8a16_select(model: nn.Module) -> None:
    pattern = os.getenv("VLLM_GLM53_W8A16_RE", "")
    if not pattern:
        return
    rx = re.compile(pattern)
    picked, left = [], []
    for name, module in model.named_modules():
        qm = getattr(module, "quant_method", None)
        w = getattr(module, "weight", None)
        if not isinstance(qm, (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)):
            continue
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16:
            continue
        if isinstance(qm, UnquantizedEmbeddingMethod) and not name.endswith("lm_head"):
            continue  # input embeddings are a gather, not a GEMM
        rows = _fp8_rows(name, module, w.shape[0])
        if rx.search(name) and rows % 128 == 0 and w.shape[1] % 128 == 0:
            module.quant_method = Glm53W8A16Method(qm, rows)
            picked.append(f"{name}{tuple(w.shape)}")
        else:
            left.append(f"{name}{tuple(w.shape)}")
    logger.info("glm53_w8a16: %d modules -> FP8 W8A16 (%s): %s", len(picked), pattern,
                ", ".join(picked))
    logger.info("glm53_w8a16: %d BF16 linears left: %s", len(left), ", ".join(left))
