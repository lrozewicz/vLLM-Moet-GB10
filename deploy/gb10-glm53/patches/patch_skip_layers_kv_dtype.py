#!/usr/bin/env python3
"""LATKA 9: dtype KV dla warstw pomijanych przez --kv-cache-dtype-skip-layers.

Galaz skip-quant w Attention.__init__ (model_executor/layers/attention/
attention.py) ustawia takim warstwom kv_cache_dtype = "auto", czyli bf16.
Dla draftera DFlash2 na GLM-5.3-Flash to 4096 B/token x 5 warstw = 20 MiB
na KAZDY blok puli KV (28% bloku 71,8 MiB). FlashInfer i Triton deklaruja
fp8_e4m3 dla zwyklych (nie-MLA) warstw - odrzucaly tylko fp8_ds_mla, ktore
canonicalizacja sparse-MLA narzuca calemu configowi.

Knob: VLLM_KV_SKIP_LAYERS_DTYPE (domyslnie "auto" = zachowanie bez latki).
"fp8_e4m3" -> KV draftera 2048 B/token, 10 MiB na blok puli, +17% blokow.

Uzycie: python3 patch_skip_layers_kv_dtype.py <attention.py>. Idempotentna.
"""
import ast
import sys

MARK = "[PATCH-SKIP-KV-DTYPE]"
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if MARK in src:
    print("juz zalatane")
    sys.exit(0)

OLD = """\
            if skip:
                kv_cache_dtype = "auto"
"""
NEW = """\
            if skip:
                # [PATCH-SKIP-KV-DTYPE] skipped layers may still use a plain
                # (non-MLA) fp8 cache when the backend takes it; default keeps
                # the stock behaviour (auto = model dtype).
                import os as _os

                kv_cache_dtype = _os.environ.get(
                    "VLLM_KV_SKIP_LAYERS_DTYPE", "auto"
                ).strip() or "auto"
"""
assert src.count(OLD) == 1, "kotwica galezi skip-quant nie znaleziona"
src = src.replace(OLD, NEW)
ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print("OK: LATKA 9 nalozona na", path)
