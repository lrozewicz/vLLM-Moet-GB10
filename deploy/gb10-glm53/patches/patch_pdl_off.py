#!/usr/bin/env python3
"""LATKA 6: przelacznik do gaszenia PDL (Programmatic Dependent Launch).

PO CO. vLLM wlacza PDL dla capability >= 9 (`CudaPlatform.is_arch_support_pdl`
zwraca `major >= 9`). To sito napisano dla Hoppera; GB10 to sm_121, czyli
major 12, wiec PDL wchodzi rowniez tutaj - na krzemie, na ktorym nikt tego nie
walidowal.

GDZIE NAS TO DOTYKA. `glm5next/nvidia/kda.py:278` tworzy `FusedRMSNormGated`,
a ten (flash_linear_attention/ops/fused_norm_gate.py:266,285) startuje swoje
kernele Tritona z `launch_pdl=current_platform.is_arch_support_pdl()`. Mamy
34 warstwy KDA, wiec ten kernel biegnie 34 razy na token.

DLACZEGO TO PODEJRZANE. Wzorzec w kernelu to:

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

czyli zgoda na start nastepnika jest wystawiana na POCZATKU, zanim kernel
zapisze cokolwiek. To jest celowy idiom prefetchowy i jest bezpieczny tylko
dopoty, dopoki nastepnik nie czyta wynikow tego kernela przez wlasny gdc_wait.
Dwa niezalezne zespoly (barrydeen, tonyd2wild) zglosily na sm_12x wyscigi
w kernelach niosacych stan rekurencyjny KDA i pogasily PDL u siebie.

Czy realizuje sie u NAS - z lektury nie wynika. Dlatego to jest przelacznik
do A/B, a nie wylaczenie na stale: `VLLM_DISABLE_PDL=1` gasi, brak zmiennej
zachowuje zachowanie upstreamu.

HIPOTEZA DO OBALENIA: degeneracja polskiej prozy na golym 2-bit (zmierzone
4 z 6 probek rozsypanych) pochodzi z tego wyscigu, a nie z kwantyzacji. Jesli
tak, warstwa FP4 3 GiB maskowala bug i mozna ja odzyskac.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[PATCH-PDL-OFF]" in src:
    print("juz zalatane")
    sys.exit(0)

OLD = """    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9"""

NEW = """    def is_arch_support_pdl(cls) -> bool:
        # [PATCH-PDL-OFF] Sito `major >= 9` pisano dla Hoppera i lapie takze
        # sm_121, gdzie PDL nikt nie walidowal. Na naszej sciezce dotyka to
        # FusedRMSNormGated w 34 warstwach KDA. VLLM_DISABLE_PDL=1 gasi to
        # globalnie, zeby dalo sie zrobic A/B; bez zmiennej - jak upstream.
        import os
        if os.getenv("VLLM_DISABLE_PDL", "0") == "1":
            return False
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9"""

if OLD not in src:
    print("NIE ZNALEZIONO kotwicy is_arch_support_pdl", file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
open(path, "w", encoding="utf-8").write(src)
import ast
ast.parse(src)
print("zalatane, skladnia OK")
