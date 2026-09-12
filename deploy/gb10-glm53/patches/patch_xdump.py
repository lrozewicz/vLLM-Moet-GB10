#!/usr/bin/env python3
"""PATCH 10 (optional): dump the MoE inputs for offline GPTQ calibration.

WHY. GPTQ needs, per layer, the Hessian of the expert inputs H = X^T X / T.
On this model every expert of a layer sees the same input (the routed token
hidden state), so one Hessian per layer serves all 288 experts - but X has to
come from the model actually running, with the real routing. This hook records
it from a live server; tools/glm53/gptq_build.py turns the dumps into planes.

WHAT. In moe_w2_cubit.py, right after the top-p routing filter, the hook
appends (x, topk_ids) for the selected layers to a host-side buffer and
flushes every VLLM_MOE_W2_XDUMP_FLUSH tokens to
$VLLM_MOE_W2_XDUMP_DIR/L{layer_key}_{n:03d}.pt  ({x: bf16 [T, H], topk_ids:
int32 [T, k]}). It records ONLY while the file $VLLM_MOE_W2_XDUMP_DIR/ARM
exists, so profiling and warm-up batches never pollute the statistics, and
never while a CUDA graph is being captured.

Off by default: with VLLM_MOE_W2_XDUMP_LAYERS empty it costs one falsy check
per MoE call. Production mounts the file WITHOUT this patch; boot.sh XDUMP=1
mounts the variant built by build_mounts.sh (moe_w2_cubit.xdump.py).

Usage (on top of the production moe_w2_cubit.py):
    python3 patch_xdump.py moe_w2_cubit.xdump.py
"""
import ast
import sys

MARK = "[PATCH-XDUMP]"

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if MARK in src:
    print("already patched")
    sys.exit(0)

OLD1 = """_afrag_ok = False
# Fused gather-reduce removes the enormous deterministic scatter buffer."""
NEW1 = """_afrag_ok = False

# [PATCH-XDUMP] MoE-input dump for offline GPTQ calibration (patch_xdump.py).
# VLLM_MOE_W2_XDUMP_LAYERS="5,20,38" (layer keys) arms it; it records only
# while the file $VLLM_MOE_W2_XDUMP_DIR/ARM exists (so profiling/warmup dummy
# batches never pollute the statistics) and never under cudagraph capture.
# Each flush writes {x: bf16 [T,H], topk_ids: int32 [T,k]} per layer.
# Off by default: an empty set costs one falsy check per MoE call.
_XDUMP_LAYERS = {int(s) for s in os.getenv("VLLM_MOE_W2_XDUMP_LAYERS", "").split(",")
                 if s.strip()}
_XDUMP_DIR = os.getenv("VLLM_MOE_W2_XDUMP_DIR", "/root/.cache/xdump")
_XDUMP_FLUSH = int(os.getenv("VLLM_MOE_W2_XDUMP_FLUSH", "32768"))
_xdump_buf: dict = {}
_xdump_n: dict = {}


def _xdump_record(layer_key: int, x: torch.Tensor, topk_ids: torch.Tensor) -> None:
    if not os.path.exists(os.path.join(_XDUMP_DIR, "ARM")):
        return
    buf = _xdump_buf.setdefault(layer_key, [])
    buf.append((x.detach().to(torch.bfloat16).cpu(),
                topk_ids.detach().to(torch.int32).cpu()))
    if sum(b[0].shape[0] for b in buf) >= _XDUMP_FLUSH:
        n = _xdump_n.get(layer_key, 0)
        torch.save({"x": torch.cat([b[0] for b in buf]),
                    "topk_ids": torch.cat([b[1] for b in buf])},
                   os.path.join(_XDUMP_DIR, f"L{layer_key}_{n:03d}.pt"))
        _xdump_n[layer_key] = n + 1
        buf.clear()
# Fused gather-reduce removes the enormous deterministic scatter buffer."""

OLD2 = """    topk_weights, topk_ids = _apply_topp(topk_weights, topk_ids)
    top_k = topk_ids.shape[1]"""
NEW2 = """    topk_weights, topk_ids = _apply_topp(topk_weights, topk_ids)
    if (_XDUMP_LAYERS and layer_key in _XDUMP_LAYERS
            and not torch.cuda.is_current_stream_capturing()):
        _xdump_record(layer_key, x, topk_ids)
    top_k = topk_ids.shape[1]"""

for old, new, name in ((OLD1, NEW1, "module globals"), (OLD2, NEW2, "routing call site")):
    n = src.count(old)
    if n != 1:
        print(f"ANCHOR NOT FOUND EXACTLY ONCE ({n}x): {name}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)

ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print("xdump hook installed (globals + routing call site), syntax OK")
