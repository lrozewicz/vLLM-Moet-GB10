# SPDX-License-Identifier: Apache-2.0
"""KDA spec-decode verify without the per-draft-token state stores (patch 16).

The stock spec path (fla fused_recurrent_gated_delta_rule_fwd_kernel with
ssm_state_indices[n, t]) stores the full fp32 recurrent state after EVERY
verified token, so the next step can start from whichever position the
rejection sampler accepts. On GLM-5.3-Flash that is 64 x 128 x 128 x 4 B
= 4 MiB per token per layer, 8 tokens x 31 KDA layers = ~1 GB of DRAM writes
per decode step (~6 ms of a ~130 ms step), almost all of it thrown away.

Here the verify kernel (``kda_spec_verify``) stores only the rank-1 update of
every token instead: the L2-normalised key, the decay exp(g) and the
beta-scaled delta v' (3 x 32 KiB per token per layer). After sampling,
``KdaCommit.materialize`` (called from the model state's postprocess_state,
before the align-mode copies) replays the accepted tokens on top of the
step's initial state and stores ONE state, at the slot the stock kernel would
have filled for the accepted position, plus the block-boundary position the
align postprocess copies when the step crosses a mamba block. Everything
downstream (precopy, align postprocess, prefix caching, the next verify step
reading ssm_state_indices[n, num_accepted - 1]) sees exactly the memory the
stock kernel leaves behind for the slots it reads.

The replay repeats the stock kernel's state arithmetic (h *= exp(g); h +=
v' k^T, fp32) on the stored operands, so the committed state is bit-exact.

Switch: VLLM_KDA_COMMIT=1 (the layer patch and the model-state patch check
it; with 0 neither imports this module).
"""
import os

import torch

from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_KDA_COMMIT", "0") == "1"


@triton.jit(do_not_specialize=["T"])
def _kda_spec_verify_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    a_log,
    g_bias,
    dlog,  # fp32 [R, TMAX, 3, HV, K] of this layer
    hdr,  # int32 [R, TMAX + 2] of this layer
    flags,  # int32 [R] of this layer
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TMAX: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    SIGMOID_BETA: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    # Same arithmetic as fused_recurrent_gated_delta_rule_fwd_kernel with
    # IS_VARLEN, IS_CONTINUOUS_BATCHING, IS_SPEC_DECODING, IS_KDA,
    # COMPUTE_GATE, SAFE_GATE, INPLACE_FINAL_STATE and NK == 1; only the
    # per-token state store is replaced by the rank-1 log.
    i_v, i_nh = tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int64),
        tl.load(cu_seqlens + i_n + 1).to(tl.int64),
    )
    T = eos - bos
    if T == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_beta = beta + bos * HV + i_hv
    p_gk = g + (bos * HV + i_hv) * K + o_k
    b_a_log = tl.exp(tl.load(a_log + i_h).to(tl.float32))
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    i_t0 = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t0).to(
        tl.int64
    )
    if state_idx <= 0:
        return
    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Row header for the commit: the slot row, the initial-state slot, T.
    if (i_v == 0) & (i_hv == 0):
        o_t = tl.arange(0, TMAX)
        row = tl.load(
            ssm_state_indices + i_n * stride_indices_seq + o_t,
            mask=o_t < T,
            other=0,
        )
        tl.store(hdr + i_n * (TMAX + 2) + o_t, row.to(tl.int32))
        tl.store(hdr + i_n * (TMAX + 2) + TMAX, state_idx.to(tl.int32))
        tl.store(hdr + i_n * (TMAX + 2) + TMAX + 1, T.to(tl.int32))
        tl.store(flags + i_n, 1)

    p_log = dlog + (i_n * TMAX * 3 * HV + i_hv) * K
    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_gk = tl.load(p_gk).to(tl.float32)
        b_gk += tl.load(g_bias + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
        b_e = exp(b_gk)  # the stock kernel's `exp` (op.exp)
        b_h *= b_e[None, :]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_beta = tl.load(p_beta).to(tl.float32)
        if SIGMOID_BETA:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # rank-1 log: k and exp(g) once per head, v' per V tile
        p_t = p_log + i_t * 3 * HV * K
        if i_v == 0:
            tl.store(p_t + o_k, b_k, mask=mask_k)
            tl.store(p_t + HV * K + o_k, b_e, mask=mask_k)
        tl.store(p_t + 2 * HV * K + o_v, b_v, mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_gk += HV * K
        p_beta += HV


@triton.jit
def _kda_commit_kernel(
    state_ptrs,  # int64 [L]: data_ptr of every layer's recurrent state
    dlog,  # fp32 [L, R, TMAX, 3, HV, K]
    hdr,  # int32 [L, R, TMAX + 2]
    flags,  # int32 [L, R]
    spec_rows,  # int32 [R]: batch row of verify row r
    n_spec,
    idx_mapping,  # int32 [num_reqs]: batch row -> request slot
    num_accepted,  # int32 [max_reqs]
    num_computed,  # int32 [max_reqs], post-step
    R: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    TMAX: tl.constexpr,
    stride_state: tl.constexpr,
    ALIGN_BLOCK: tl.constexpr,  # mamba block size in align mode, else 0
    REPLAY_MODE: tl.constexpr,
):
    i_v, i_hv, i_lr = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l = i_lr // R
    i_r = i_lr % R
    if i_r >= n_spec:
        return
    if tl.load(flags + i_l * R + i_r) == 0:
        return
    b = tl.load(spec_rows + i_r)
    req = tl.load(idx_mapping + b)
    if req < 0:
        return
    p_hdr = hdr + (i_l * R + i_r) * (TMAX + 2)
    n_tok = tl.load(p_hdr + TMAX + 1)
    na = tl.minimum(tl.maximum(tl.load(num_accepted + req), 1), n_tok)
    src = tl.load(p_hdr + TMAX).to(tl.int64)

    extra = -1
    if ALIGN_BLOCK > 0:
        new_computed = tl.load(num_computed + req)
        running = new_computed - na + 1
        aligned = (new_computed // ALIGN_BLOCK) * ALIGN_BLOCK
        if aligned >= running:
            extra = aligned - running

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    base = tl.load(state_ptrs + i_l).to(tl.pointer_type(tl.float32))
    off_h = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, K], dtype=tl.float32)
    b_h += tl.load(base + src * stride_state + off_h)

    p_log = dlog + ((i_l * R + i_r) * TMAX * 3 * HV + i_hv) * K
    for i_t in range(0, na):
        p_t = p_log + i_t * 3 * HV * K
        b_k = tl.load(p_t + o_k)
        b_e = tl.load(p_t + HV * K + o_k)
        b_v = tl.load(p_t + 2 * HV * K + o_v)
        if REPLAY_MODE == 1:
            b_h = tl.fma(b_v[:, None], b_k[None, :], b_h * b_e[None, :])
        elif REPLAY_MODE == 2:
            b_h = tl.fma(b_h, b_e[None, :], b_v[:, None] * b_k[None, :])
        else:
            b_h *= b_e[None, :]
            b_h += b_v[:, None] * b_k[None, :]
        if (i_t == extra) & (i_t != na - 1):
            slot = tl.load(p_hdr + i_t).to(tl.int64)
            if slot > 0:
                tl.store(base + slot * stride_state + off_h, b_h)
    slot = tl.load(p_hdr + na - 1).to(tl.int64)
    if slot > 0:
        tl.store(base + slot * stride_state + off_h, b_h)


class KdaCommit:
    """Process-wide buffers shared by all KDA layers."""

    num_layers = 0  # counted at layer construction
    R = 0
    TMAX = 0
    HV = K = V = 0
    dlog: torch.Tensor | None = None
    hdr: torch.Tensor | None = None
    flags: torch.Tensor | None = None
    _ptrs: dict[int, int] = {}
    _stride: int | None = None
    _ptr_table: torch.Tensor | None = None
    _spec_rows_cpu: torch.Tensor | None = None
    _spec_rows: torch.Tensor | None = None
    _logged = False
    replay_mode = int(os.environ.get("VLLM_KDA_COMMIT_REPLAY", "1"))

    @classmethod
    def register_layer(cls) -> int:
        cls.num_layers += 1
        return cls.num_layers - 1

    @classmethod
    def ensure_buffers(cls, R: int, TMAX: int, HV: int, K: int, V: int, device):
        if cls.dlog is not None:
            return
        assert K == V, "kda_commit packs k / exp(g) / v' with one stride"
        cls.R, cls.TMAX, cls.HV, cls.K, cls.V = R, TMAX, HV, K, V
        L = cls.num_layers
        cls.dlog = torch.zeros(L, R, TMAX, 3, HV, K, dtype=torch.float32, device=device)
        cls.hdr = torch.zeros(L, R, TMAX + 2, dtype=torch.int32, device=device)
        cls.flags = torch.zeros(L, R, dtype=torch.int32, device=device)
        cls._spec_rows_cpu = torch.zeros(R, dtype=torch.int32).pin_memory()
        cls._spec_rows = torch.zeros(R, dtype=torch.int32, device=device)
        logger.info(
            "kda_commit: %d KDA layers, %d rows x %d tokens, log %.1f MiB "
            "(replaces %d per-token state stores per layer)",
            L, R, TMAX, cls.dlog.numel() * 4 / 2**20, TMAX,
        )

    @classmethod
    def note_state(cls, layer: int, state: torch.Tensor):
        ptr = state.data_ptr()
        if cls._ptrs.get(layer) != ptr:
            cls._ptrs[layer] = ptr
            st = state.stride(0)
            assert cls._stride in (None, st), "KDA state strides differ by layer"
            cls._stride = st
            cls._ptr_table = None

    @classmethod
    def verify(
        cls,
        layer: int,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        out,
        a_log,
        g_bias,
        lower_bound,
    ):
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[-1]
        N = len(cu_seqlens) - 1
        TMAX = ssm_state_indices.size(-1)
        cls.ensure_buffers(max(cls._max_rows, N), TMAX, HV, K, V, k.device)
        assert N <= cls.R and TMAX <= cls.TMAX, (N, cls.R, TMAX, cls.TMAX)
        cls.note_state(layer, initial_state)
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        g, beta = g.contiguous(), beta.contiguous()
        if out is None:
            o = torch.empty_like(k)
        else:
            assert out.shape == k.shape and out.dtype == k.dtype
            assert out.is_contiguous()
            o = out
        _kda_spec_verify_kernel[(1, triton.cdiv(V, BV), N * HV)](
            q, k, v, g, beta, o, initial_state, cu_seqlens, ssm_state_indices,
            num_accepted_tokens, a_log.reshape(-1).contiguous(),
            g_bias.reshape(-1).contiguous(), cls.dlog[layer], cls.hdr[layer],
            cls.flags[layer], K**-0.5, T,
            H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, TMAX=cls.TMAX,
            stride_init_state_token=initial_state.stride(0),
            stride_indices_seq=ssm_state_indices.stride(0),
            USE_QK_L2NORM_IN_KERNEL=True,
            SIGMOID_BETA=True,
            LOWER_BOUND=lower_bound,
            num_warps=1,
            num_stages=3,
        )
        return o

    _max_rows = 1

    @classmethod
    def set_max_rows(cls, n: int):
        cls._max_rows = max(cls._max_rows, int(n))

    @classmethod
    def set_spec_rows(cls, rows):
        """Batch rows that run the verify kernel in the coming forward, in
        order (GDN keeps the spec rows' relative order)."""
        cls._rows = list(rows)

    _rows: list = []

    @classmethod
    def materialize(cls, idx_mapping, num_accepted, num_computed, align_block: int):
        n = len(cls._rows)
        if cls.dlog is None or n == 0:
            return
        assert n <= cls.R, (n, cls.R)
        cls._spec_rows_cpu[:n] = torch.tensor(cls._rows, dtype=torch.int32)
        cls._spec_rows[:n].copy_(cls._spec_rows_cpu[:n], non_blocking=True)
        cls._rows = []
        if cls._ptr_table is None:
            assert len(cls._ptrs) == cls.num_layers, (len(cls._ptrs), cls.num_layers)
            cls._ptr_table = torch.tensor(
                [cls._ptrs[i] for i in range(cls.num_layers)],
                dtype=torch.int64,
                device=cls.dlog.device,
            )
        BV = 8
        grid = (cls.V // BV, cls.HV, cls.num_layers * cls.R)
        _kda_commit_kernel[grid](
            cls._ptr_table, cls.dlog, cls.hdr, cls.flags, cls._spec_rows,
            n, idx_mapping, num_accepted, num_computed,
            R=cls.R, HV=cls.HV, K=cls.K, V=cls.V, BV=BV, TMAX=cls.TMAX,
            stride_state=cls._stride, ALIGN_BLOCK=align_block,
            REPLAY_MODE=cls.replay_mode,
            num_warps=1,
        )
        cls.flags.zero_()
        if not cls._logged:
            cls._logged = True
            logger.info("kda_commit: first commit (%d layers)", cls.num_layers)
