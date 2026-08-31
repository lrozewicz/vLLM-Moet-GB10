#!/usr/bin/env python3
"""LATKA 1: klamra SwiGLU dla ekspertow routowanych.

PROBLEM. moe_w2_cubit._activation_kwargs czyta limit przez
    limit = getattr(layer, "swiglu_limit", None)
Ten atrybut istnial na warstwie w vLLM 0.24, pod ktore pisano silnik W2.
W 0.29-dev limit mieszka w FusedMoEConfig (config.py:1324) i warstwa wystawia
go jako layer.moe_config.swiglu_limit - tak czytaja go wszystkie backendy
upstreamu (np. trtllm_nvfp4_moe.py:100 `getattr(moe_config,"swiglu_limit",None)`).

SKUTEK. getattr zwraca None, kod CICHO schodzi na goly `silu_and_mul`
(moe_w2_cubit.py:2697), mimo ze GLM-5.3-Flash deklaruje swiglu_limit=10.0
i byl trenowany z klamra. Upstream jawnie broni sie przed taka sytuacja
("backends that do not implement the clamp are filtered out ... so the oracle
cannot silently select one and drop the clamp"), ale silnik W2 podpina sie
pod Fp8MoEMethod i omija ten straznik.

Niesklamrowana SwiGLU na wagach 2-bitowych to dokladnie mechanizm produkujacy
rozpad na tresci o wysokiej entropii: szum kwantyzacji wzmacnia sie w bramce
i nic go nie ogranicza.

LATKA. Czytaj limit z warstwy, a gdy go tam nie ma - z moe_config. Logujemy
zrodlo, zeby nigdy wiecej nie zgadywac.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "moe_w2_cubit.py"
src = open(path, encoding="utf-8").read()

OLD = '    limit = getattr(layer, "swiglu_limit", None)\n'
NEW = '''    # [PATCH-SWIGLU] vLLM 0.24 trzymal limit na warstwie; 0.29-dev trzyma go
    # w FusedMoEConfig (layer.moe_config.swiglu_limit) - tak czytaja go
    # backendy upstreamu. Bez tego fallbacku getattr zwraca None i klamra
    # znika PO CICHU, mimo ze checkpoint deklaruje swiglu_limit=10.0.
    limit = getattr(layer, "swiglu_limit", None)
    _limit_src = "layer"
    if limit is None:
        _mc = getattr(layer, "moe_config", None)
        limit = getattr(_mc, "swiglu_limit", None)
        _limit_src = "moe_config"
    if limit is None:
        _qm = getattr(layer, "quant_method", None)
        limit = getattr(getattr(_qm, "moe", None), "swiglu_limit", None)
        _limit_src = "quant_method.moe"
    logger.warning_once(
        "moe_w2_cubit: routed-expert swiglu_limit=%s (zrodlo: %s)",
        limit, _limit_src if limit is not None else "BRAK - klamra WYLACZONA")
'''

if "[PATCH-SWIGLU]" in src:
    print("juz zalatane")
    sys.exit(0)
if OLD not in src:
    print("NIE ZNALEZIONO kotwicy:", repr(OLD), file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
open(path, "w", encoding="utf-8").write(src)
import ast
ast.parse(src)
print("zalatane, skladnia OK")
