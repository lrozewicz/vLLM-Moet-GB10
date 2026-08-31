#!/usr/bin/env python3
"""LATKA 7: pozwol trzymac stan rekurencyjny KDA w bf16 zamiast f32.

PO CO. Rozmiar strony attention na tym modelu nie jest wyborem - vLLM wyrownuje
go W GORE do strony mamby:

    Setting attention block size to 6912 tokens to ensure that attention page
    size is >= mamba page size

Strona 6912 tokenow jest zrodlem podatku, ktory ZMIERZYLISMY: agent placi
11-22 s TTFT na ture NIEZALEZNIE od tego, jak malo dokleja, z pila o okresie
3 tur (3 x 2067 ~ 6912). Prefix cache dziala w granulacji strony, wiec blok
z granica "cache / nowe" trzeba przeliczyc w calosci.

CO BLOKUJE. `MambaStateDtypeCalculator.kda_state_dtype` (mamba_utils.py)
zwraca `(state_dtype, torch.float32)` - drugi element, czyli stan rekurencyjny,
jest ZAHARDKODOWANY na f32 i NIE slucha ani `--mamba-cache-dtype`, ani
`--mamba-ssm-cache-dtype`. Sprawdzone w naszym obrazie: flagi CLI istnieja, ale
na tej sciezce sa martwe. Receptura SGLang dla tego modelu jawnie ustawia
`--mamba-ssm-dtype bfloat16`, wiec bf16 nie jest tu pomyslem z sufitu.

CO DAJE. Stan o polowe mniejszy -> strona mamby o polowe mniejsza -> vLLM
wyrownuje attention do ~3456 zamiast 6912 -> **podatek prefix-cache o polowe
mniejszy** (11-22 s -> 6-11 s na ture agenta). Plus troche pamieci.

RYZYKO, wypowiedziane wprost. To zmiana NUMERYCZNA w 34 z 45 warstw, na stanie,
ktory z definicji akumuluje sie przez caly kontekst - czyli dokladnie tam, gdzie
blad narasta. Musi byc zmierzona sonda jakosci, i to na dlugim kontekscie, a nie
tylko na krotkich promptach. Jest tez znany bug vLLM #27264 o malformacji
cache'u hybryd przy zawijaniu alokacji blokow (dotyczy f32, ale sygnalizuje, ze
ta sciezka bywa krucha).

Dlatego to jest PRZELACZNIK: `VLLM_KDA_SSM_DTYPE=bfloat16` wlacza, brak
zmiennej zachowuje f32 upstreamu.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[PATCH-KDA-SSM-DTYPE]" in src:
    print("juz zalatane")
    sys.exit(0)

OLD = """        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, torch.float32)"""

NEW = """        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        # [PATCH-KDA-SSM-DTYPE] Drugi element to stan rekurencyjny i upstream
        # przybija go do f32, ignorujac --mamba-ssm-cache-dtype. To on dyktuje
        # rozmiar strony mamby, do ktorej vLLM wyrownuje strone attention
        # (u nas 6912 tokenow), a ta z kolei jest zrodlem stalego podatku
        # prefix-cache na ture agenta. bf16 powinno zbic strone o polowe.
        # Zmiana NUMERYCZNA w 34 z 45 warstw - wlacznie tylko jawnie.
        import os
        _ssm = os.getenv("VLLM_KDA_SSM_DTYPE", "").strip().lower()
        if _ssm in ("bfloat16", "bf16"):
            return (state_dtype, torch.bfloat16)
        if _ssm in ("float16", "fp16"):
            return (state_dtype, torch.float16)
        return (state_dtype, torch.float32)"""

if OLD not in src:
    print("NIE ZNALEZIONO kotwicy kda_state_dtype", file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
open(path, "w", encoding="utf-8").write(src)
import ast
ast.parse(src)
print("zalatane, skladnia OK")
