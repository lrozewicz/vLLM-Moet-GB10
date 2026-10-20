#!/usr/bin/env python3
"""PATCH 14: the decode MoE launch without empty pairs, plus the glue fusion.

PROBLEM. A decode forward launches moe_w2_mm / moe_w4_mm with grid.y = the
moe_align table length (64 entries at T=8 with k=7), but each tier only has
its own live pairs in it: the rest load their desc and EXIT. On GB10 an
empty pair costs ~0.5 us (w2) / ~0.8 us (w4) per launch, 4-5 ms per step.

PATCH (with the looped cubins from kernels/cubins-sm120-loop, built by
kernels/gen/sched_loopify.py from the shipped ones, schedule unchanged):
  * _desc_build_kernel gets COMPACT: each tier's live pairs are written to
    the front of its table (cumsum rank), the tail gets m_rows = 0;
  * _launch passes the two new params (n_pairs, y_stride) and launches
    grid.y = VLLM_MOE_W2_LOOP_GRID ("16,12" = w2, w4) on the decode path;
    the cubins walk y = ctaid.y + k * y_stride and stop at the first empty
    entry. Cubins without the params ignore them (the driver reads 5);
  * VLLM_MOE_W2_FUSE_GLUE=1: one Triton kernel quantizes each slot's token
    row straight from x (UE8M0, group 128) and sets the FP4 tier's seen
    flags, replacing the token quant, the rows where/floordiv, two
    index_selects, the scale copy and the index_fill; a second one fuses the
    fp32-clamped SwiGLU with the a2/as2 quant.
Bit-exact with the unpatched path (GPU checks: tools/glm53/loop_check.py,
desc_check.py, glue_bench.py). Measured on GB10 (code / prose step):
140.6 / 137.6 ms -> 131.5 / 130.5 ms, code 39.2 -> 42.2 tok/s, HumanEval
154 -> 156. Empty VLLM_MOE_W2_LOOP_GRID / FUSE_GLUE=0 = old behaviour.

Applies after patch_store_skip_read (23 anchored hunks, generated from the
production file and replayed in order; every anchor must match exactly once).
"""
import ast
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "VLLM_MOE_W2_LOOP_GRID" in src:
    print("already patched")
    sys.exit(0)

HUNKS = [
    ("_PREFILL_T = int(os.getenv(\"VLLM_MOE_W2_PREFILL_T\", \"96\"))\n\n",
     "_PREFILL_T = int(os.getenv(\"VLLM_MOE_W2_PREFILL_T\", \"96\"))\n# Decode grid-stride launch (cubins built with MOEW2_LOOP=1): the desc build\n# compacts each tier's live pairs to the front of its table and the w2/w4\n# GEMMs launch grid.y = G instead of grid.y = pairs, so the empty pairs\n# (m_rows == 0, ~0.5/0.8 us each per launch on GB10) cost nothing.\n# \"G_w2,G_w4\" (e.g. \"16,8\"); empty = off (one CTA row per table entry).\n_LOOP_GRID = tuple(int(x) for x in os.getenv(\n    \"VLLM_MOE_W2_LOOP_GRID\", \"\").split(\",\") if x.strip())\nassert len(_LOOP_GRID) in (0, 2), \"VLLM_MOE_W2_LOOP_GRID wants G_w2,G_w4\"\n\n"),
    ("        logger.info(\"moe_w2_cubit: cubins loaded: %s\", sorted(_fns))\n        return True\n",
     "        logger.info(\"moe_w2_cubit: cubins loaded: %s\", sorted(_fns))\n        if _LOOP_GRID:\n            logger.info(\"moe_w2_cubit: decode grid-stride launch, grid.y \"\n                        \"w2=%d w4=%d (compacted desc tables)\", *_LOOP_GRID)\n        return True\n"),
    ("import triton.language as tl\n\n\n@triton.jit\n",
     "import triton.language as tl\n\n# Decode glue fusion (VLLM_MOE_W2_FUSE_GLUE=1; UE8M0 group-128 activations,\n# the production a32 setting): one kernel quantizes each slot's token row\n# straight from x into a1/as1 (replacing the token quant, the rows\n# where/full_like/floordiv, two index_selects and the scale broadcast copy)\n# and sets the FP4 tier's seen flags (index_fill); a second one does the\n# fp32-clamped SwiGLU -> BF16 -> a2/as2 quant in one pass. Bit-identical to\n# per_token_group_quant_fp8(use_ue8m0=True) on the valid slots.\n_FUSE_GLUE = os.getenv(\"VLLM_MOE_W2_FUSE_GLUE\", \"0\") == \"1\"\n\n\n@triton.jit\ndef _ue8m0_quant_store(v, q_ptr, s_ptr, offs, sub, G: tl.constexpr):\n    amax = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-10)\n    sc = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))\n    q = tl.minimum(tl.maximum(v / sc, -448.0), 448.0)\n    tl.store(q_ptr + offs, q.to(q_ptr.dtype.element_ty))\n    tl.store(s_ptr + sub, tl.zeros([G // 32], tl.float32) + sc)\n\n\n@triton.jit\ndef _gather_quant_kernel(x_ptr, sid_ptr, a_ptr, as_ptr, seen_ptr, eb_ptr,\n                         n_routes, top_k, stride_x, stride_a, stride_as,\n                         MBLOCK: tl.constexpr, G: tl.constexpr,\n                         MARK: tl.constexpr):\n    s = tl.program_id(0)\n    g = tl.program_id(1)\n    sid = tl.load(sid_ptr + s)\n    valid = sid < n_routes\n    row = tl.where(valid, sid // top_k, 0).to(tl.int64)\n    offs = g * G + tl.arange(0, G)\n    v = tl.load(x_ptr + row * stride_x + offs, mask=(offs >= 0) & valid,\n                other=0.0).to(tl.float32)\n    sub = g * (G // 32) + tl.arange(0, G // 32)\n    _ue8m0_quant_store(v, a_ptr + s.to(tl.int64) * stride_a, as_ptr + s.to(tl.int64) * stride_as,\n                       offs, sub, G)\n    if MARK:\n        e = tl.load(eb_ptr + s // MBLOCK)\n        one = tl.full((1,), 1, dtype=seen_ptr.dtype.element_ty)\n        tl.store(seen_ptr + e + tl.arange(0, 1), one,\n                 mask=(tl.arange(0, 1) == 0) & valid & (g == 0))\n\n\n@triton.jit\ndef _silu_quant_kernel(c_ptr, a_ptr, as_ptr, inter, stride_c, stride_a,\n                       stride_as, limit, G: tl.constexpr):\n    s = tl.program_id(0).to(tl.int64)\n    g = tl.program_id(1)\n    offs = g * G + tl.arange(0, G)\n    gate = tl.load(c_ptr + s * stride_c + offs).to(tl.float32)\n    up = tl.load(c_ptr + s * stride_c + inter + offs).to(tl.float32)\n    gate = tl.minimum(gate, limit)\n    up = tl.clamp(up, -limit, limit)\n    glu = gate / (1.0 + tl.exp(-gate))\n    v = (glu * up).to(tl.bfloat16).to(tl.float32)\n    sub = g * (G // 32) + tl.arange(0, G // 32)\n    _ue8m0_quant_store(v, a_ptr + s * stride_a, as_ptr + s * stride_as,\n                       offs, sub, G)\n\n\n\n@triton.jit\n"),
    ("    n_experts, pairs, cap6, mblock,\n    BLOCK: tl.constexpr,\n):\n    \"\"\"All four moe desc tables in one launch (24 columns per pair).\n",
     "    n_experts, pairs, cap6, mblock,\n    BLOCK: tl.constexpr,\n    COMPACT: tl.constexpr = False,\n):\n    \"\"\"All four moe desc tables in one launch (24 columns per pair).\n"),
    ("    \"\"\"All four moe desc tables in one launch (24 columns per pair).\n\n",
     "    \"\"\"All four moe desc tables in one launch (24 columns per pair).\n\n    COMPACT (single program, pairs <= BLOCK): each tier's live pairs are\n    written to the FRONT of its tables in pair order, the tail gets m=0 \u2014\n    the looped cubins stop at the first empty entry. The c/a pointers stay\n    per pair, so results are bit-identical to the scattered layout.\n\n"),
    ("    bs2 = s2b + e * s2s\n    for gi in tl.static_range(4):\n",
     "    bs2 = s2b + e * s2s\n    if COMPACT:\n        l2 = mask & (m2 > 0)\n        l4 = mask & (m4 > 0)\n        r2 = tl.cumsum(l2.to(tl.int32), axis=0) - 1\n        r4 = tl.cumsum(l4.to(tl.int32), axis=0) - 1\n        n2 = tl.sum(l2.to(tl.int32), axis=0)\n        n4 = tl.sum(l4.to(tl.int32), axis=0)\n    for gi in tl.static_range(4):\n"),
    ("    for gi in tl.static_range(4):\n        d = d_ptr + gi * cap6 + p * 6\n        if gi == 0:\n            b, s, a, as_, c, m = p13b + e * p13s, bs13, a1, as1, c13, m2\n",
     "    for gi in tl.static_range(4):\n        d = d_ptr + gi * cap6 + p * 6\n        smask = mask\n        if COMPACT:\n            if gi < 2:\n                d = d_ptr + gi * cap6 + r2.to(tl.int64) * 6\n                smask = l2\n                tl.store(d_ptr + gi * cap6 + p * 6 + 5, tl.zeros_like(m2),\n                         mask=mask & (p >= n2))\n            else:\n                d = d_ptr + gi * cap6 + r4.to(tl.int64) * 6\n                smask = l4\n                tl.store(d_ptr + gi * cap6 + p * 6 + 5, tl.zeros_like(m4),\n                         mask=mask & (p >= n4))\n        if gi == 0:\n            b, s, a, as_, c, m = p13b + e * p13s, bs13, a1, as1, c13, m2\n"),
    ("                                  bs2, a2, as2, c2, m4)\n        tl.store(d + 0, a, mask=mask)\n        tl.store(d + 1, as_, mask=mask)\n        tl.store(d + 2, b, mask=mask)\n        tl.store(d + 3, s, mask=mask)\n        tl.store(d + 4, c, mask=mask)\n        tl.store(d + 5, m, mask=mask)\n\n",
     "                                  bs2, a2, as2, c2, m4)\n        tl.store(d + 0, a, mask=smask)\n        tl.store(d + 1, as_, mask=smask)\n        tl.store(d + 2, b, mask=smask)\n        tl.store(d + 3, s, mask=smask)\n        tl.store(d + 4, c, mask=smask)\n        tl.store(d + 5, m, mask=smask)\n\n"),
    ("def _launch(tier: str, K: int, desc: torch.Tensor, n_rows: int, pairs: int,\n            stream):\n    fn = _fns[(tier, K)]\n",
     "def _launch(tier: str, K: int, desc: torch.Tensor, n_rows: int, pairs: int,\n            stream, grid_y: int = 0):\n    \"\"\"grid_y > 0: grid-stride launch over a compacted table (looped cubins).\n    The n_pairs/y_stride params are always passed; one-pair-per-CTA cubins\n    declare only five and the driver ignores the rest.\"\"\"\n    fn = _fns[(tier, K)]\n"),
    ("    fn = _fns[(tier, K)]\n    args = [ctypes.c_uint64(desc.data_ptr()),\n",
     "    fn = _fns[(tier, K)]\n    gy = grid_y if grid_y > 0 else pairs\n    args = [ctypes.c_uint64(desc.data_ptr()),\n"),
    ("            ctypes.c_uint32(n_rows * 2),\n            ctypes.c_uint32(K // 128)]\n    argv = (ctypes.c_void_p * len(args))(\n",
     "            ctypes.c_uint32(n_rows * 2),\n            ctypes.c_uint32(K // 128),\n            ctypes.c_uint32(pairs),\n            ctypes.c_uint32(gy)]\n    argv = (ctypes.c_void_p * len(args))(\n"),
    ("        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])\n    _ck(_driver().cuLaunchKernel(fn, n_rows // 16, pairs, 1,\n                                 _nwarp_for_k(K) * 32, 1, 1, 0,\n",
     "        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])\n    _ck(_driver().cuLaunchKernel(fn, n_rows // 16, gy, 1,\n                                 _nwarp_for_k(K) * 32, 1, 1, 0,\n"),
    ("    pad_row = xq.shape[0] - 1\n    _quant_a32(x, xq[:T], ws[\"xs\"][:T], _G1)\n    valid = sorted_ids < T * top_k\n    rows = torch.where(valid, sorted_ids // top_k,\n                       torch.full_like(sorted_ids, pad_row))\n    torch.index_select(xq.view(torch.uint8), 0, rows,\n                       out=ws[\"a1\"][:slots].view(torch.uint8))\n    torch.index_select(ws[\"xs\"], 0, rows, out=ws[\"as1\"][:slots])\n\n",
     "    pad_row = xq.shape[0] - 1\n    fuse = (_FUSE_GLUE and not prefill and _G1 == 128 and _G2 == 128\n            and _A32_UE8M0 and not st.get(\"base\", False)\n            and st[\"K13\"] % 128 == 0 and st[\"K2\"] % 128 == 0\n            and _FUSED_UNPERMUTE)          # the legacy unpermute reads `valid`\n    fused_seen = False\n    if fuse:\n        tier0 = None\n        from vllm.model_executor.layers.quantization.utils import moe_w2_delta\n        tier0 = moe_w2_delta._TIER\n        fused_seen = (tier0 is not None and not moe_w2_delta._COUNT)\n        seen = tier0.seen[layer_key] if fused_seen else sorted_ids\n        _gather_quant_kernel[(slots, st[\"K13\"] // 128)](\n            x, sorted_ids, ws[\"a1\"], ws[\"as1\"], seen, expert_blocks,\n            T * top_k, top_k, x.stride(0), ws[\"a1\"].stride(0),\n            ws[\"as1\"].stride(0), MBLOCK=mblock, G=128, MARK=fused_seen,\n            num_warps=1)\n    else:\n        _quant_a32(x, xq[:T], ws[\"xs\"][:T], _G1)\n        valid = sorted_ids < T * top_k\n        rows = torch.where(valid, sorted_ids // top_k,\n                           torch.full_like(sorted_ids, pad_row))\n        torch.index_select(xq.view(torch.uint8), 0, rows,\n                           out=ws[\"a1\"][:slots].view(torch.uint8))\n        torch.index_select(ws[\"xs\"], 0, rows, out=ws[\"as1\"][:slots])\n\n"),
    ("    miss_rows = None\n    use_pf4 = False        # prefill-FP4 (resident AND base modes; _PREFILL_FP4)\n",
     "    miss_rows = None\n    g2 = g4 = 0            # grid-stride launch (compacted decode tables)\n    use_pf4 = False        # prefill-FP4 (resident AND base modes; _PREFILL_FP4)\n"),
    ("            pool_ptr = tier.pool.data_ptr()\n            moe_w2_delta.mark_seen(tier.seen[layer_key],\n                                   topk_ids.view(-1).long())\n        else:\n",
     "            pool_ptr = tier.pool.data_ptr()\n            if not fused_seen:\n                moe_w2_delta.mark_seen(tier.seen[layer_key],\n                                       topk_ids.view(-1).long())\n        else:\n"),
    ("        else:\n            _desc_build_kernel[(triton.cdiv(pairs, 256),)](\n",
     "        else:\n            compact = (bool(_LOOP_GRID) and not prefill and pairs <= 256\n                       and not moe_w2_delta.split_enabled())\n            _desc_build_kernel[(triton.cdiv(pairs, 256),)](\n"),
    ("                (st[\"K2\"] // 32) * 4, 2 * st[\"K13\"],\n                st[\"E\"], pairs, cap * 6, mblock, BLOCK=256)\n        if (tier is not None and (not prefill or use_pf4)\n",
     "                (st[\"K2\"] // 32) * 4, 2 * st[\"K13\"],\n                st[\"E\"], pairs, cap * 6, mblock, BLOCK=256,\n                COMPACT=compact)\n            if compact:\n                g2, g4 = _LOOP_GRID\n        if (tier is not None and (not prefill or use_pf4)\n"),
    ("        _afrag_repack(ws[\"a1\"], ws[\"a1f\"], pairs, st[\"K13\"])\n    _launch(w2tier, st[\"K13\"], d[0], st[\"N13\"], pairs, stream)\n    # split-FP4 dispatch: both residency modes fill ws[\"desc4s\"] (classic:\n",
     "        _afrag_repack(ws[\"a1\"], ws[\"a1f\"], pairs, st[\"K13\"])\n    _launch(w2tier, st[\"K13\"], d[0], st[\"N13\"], pairs, stream, g2)\n    # split-FP4 dispatch: both residency modes fill ws[\"desc4s\"] (classic:\n"),
    ("        else:\n            _launch(\"w4\", st[\"K13\"], d[2], st[\"N13\"], pairs, stream)\n    elif use_pf4:\n",
     "        else:\n            _launch(\"w4\", st[\"K13\"], d[2], st[\"N13\"], pairs, stream, g4)\n    elif use_pf4:\n"),
    ("    act = ws[\"act\"][:slots]\n    if st.get(\"swiglu_limit\") is None:\n        torch.ops._C.silu_and_mul(act, ws[\"c13\"][:slots])\n",
     "    act = ws[\"act\"][:slots]\n    if (fuse and st.get(\"swiglu_limit\") is not None and _SWIGLU_CLAMP_FP32\n            and not use_afrag):\n        _silu_quant_kernel[(slots, st[\"K2\"] // 128)](\n            ws[\"c13\"], ws[\"a2\"], ws[\"as2\"], st[\"K2\"], ws[\"c13\"].stride(0),\n            ws[\"a2\"].stride(0), ws[\"as2\"].stride(0),\n            float(st[\"swiglu_limit\"]), G=128, num_warps=1)\n    elif st.get(\"swiglu_limit\") is None:\n        torch.ops._C.silu_and_mul(act, ws[\"c13\"][:slots])\n"),
    ("    # requant here was one of the two activation-precision gaps vs native)\n    _quant_a32(act, ws[\"a2\"][:slots], ws[\"as2\"][:slots], _G2)\n    if use_afrag:\n",
     "    # requant here was one of the two activation-precision gaps vs native)\n    if not (fuse and st.get(\"swiglu_limit\") is not None and _SWIGLU_CLAMP_FP32\n            and not use_afrag):\n        _quant_a32(act, ws[\"a2\"][:slots], ws[\"as2\"][:slots], _G2)\n    if use_afrag:\n"),
    ("        _afrag_repack(ws[\"a2\"], ws[\"a2f\"], pairs, st[\"K2\"])\n    _launch(w2tier, st[\"K2\"], d[1], st[\"N2\"], pairs, stream)\n    if tier is not None and not prefill:\n",
     "        _afrag_repack(ws[\"a2\"], ws[\"a2f\"], pairs, st[\"K2\"])\n    _launch(w2tier, st[\"K2\"], d[1], st[\"N2\"], pairs, stream, g2)\n    if tier is not None and not prefill:\n"),
    ("        else:\n            _launch(\"w4\", st[\"K2\"], d[3], st[\"N2\"], pairs, stream)\n    elif use_pf4:\n",
     "        else:\n            _launch(\"w4\", st[\"K2\"], d[3], st[\"N2\"], pairs, stream, g4)\n    elif use_pf4:\n"),
]

for i, (old, new) in enumerate(HUNKS):
    n = src.count(old)
    if n != 1:
        sys.exit(f"patch_moe_loop: hunk {i} anchor found {n} times (expected 1)")
    src = src.replace(old, new)
ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print(f"patch_moe_loop: {len(HUNKS)} hunks applied")
