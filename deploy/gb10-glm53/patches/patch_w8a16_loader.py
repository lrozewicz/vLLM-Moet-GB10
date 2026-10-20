#!/usr/bin/env python3
"""PATCH 13: block-128 FP8 W8A16 for selected BF16 linears (GB10 decode).

WHY. Decode on GB10 is bandwidth bound, and an nsys profile of one GLM-5.3-Flash
decode step (2026-10-19) showed 46% of it in BF16 GEMMs: the KDA q/k/v/o and
the lm_head that the FP8 release ships in BF16, the MLA q_b/o_proj the NoPE
overlay dequantizes to BF16 on load, and the BF16 DFlash2 drafter. Reading
them as FP8 halves the bytes.

HOW. The method, the kernels and the module selection live in a new file,
glm53_w8a16.py (mounted as model_executor/layers/quantization/utils/
glm53_w8a16.py). This patch only calls glm53_w8a16_select(model) at the top
of model_loader/utils.py:process_weights_after_loading - which runs for the
target model and for the drafter - so the selected modules get the W8A16
method before their process_weights_after_loading is called. The module is
imported only when VLLM_GLM53_W8A16_RE is set; unset = upstream behaviour.
"""
import ast
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[PATCH-W8A16]" in src:
    print("already patched")
    sys.exit(0)

OLD = """) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):"""

NEW = """) -> None:
    # [PATCH-W8A16] FP8 W8A16 for the BF16 linears named by VLLM_GLM53_W8A16_RE
    import os

    if os.environ.get("VLLM_GLM53_W8A16_RE"):
        from vllm.model_executor.layers.quantization.utils.glm53_w8a16 import (
            glm53_w8a16_select,
        )

        glm53_w8a16_select(model)
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):"""

if OLD not in src:
    print("anchor not found (process_weights_after_loading)", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD, NEW, 1)
ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print("patched, syntax OK")
