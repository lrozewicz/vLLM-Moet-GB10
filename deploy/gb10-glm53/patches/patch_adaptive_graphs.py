#!/usr/bin/env python3
"""PATCH 15b: decode cudagraphs for every adaptive draft depth (V2 runner).

With dynamic speculative decoding the V2 runner captures one uniform-decode
FULL graph per k that num_speculative_tokens_per_batch_size lists. The
adaptive controller (patch_adaptive_k.py, VLLM_SPEC_ADAPTIVE=1) picks k from
VLLM_SPEC_ADAPTIVE_KS regardless of the batch size, so every k it may pick
needs its graph - otherwise those steps fall back to piecewise graphs (on
MiMo/V1 that cost a third of the prose speed). This adds the KS values to the
captured decode query lengths. Inert unless VLLM_SPEC_ADAPTIVE=1.

  python3 patch_adaptive_graphs.py <v1/worker/gpu/cudagraph_utils.py>
"""
import ast
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "VLLM_SPEC_ADAPTIVE_KS" in src:
    print("already patched")
    sys.exit(0)
old = """            decode_query_lens = [
                x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
            ]
"""
new = old + """            # [PATCH 15b] graphs for every k the adaptive controller may pick
            import os as _os

            if _os.environ.get("VLLM_SPEC_ADAPTIVE", "0") == "1":
                _ks = _os.environ.get("VLLM_SPEC_ADAPTIVE_KS", "")
                decode_query_lens = sorted(
                    set(decode_query_lens)
                    | {int(x) + num_new_sampled_tokens_per_step
                       for x in _ks.split(",") if x.strip()}
                )
"""
if src.count(old) != 1:
    sys.exit("patch_adaptive_graphs: anchor (decode_query_lens) moved")
src = src.replace(old, new)
ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print("decode graphs for the adaptive KS added, syntax OK")
