#!/usr/bin/env python3
"""LATKA 2: pozwol warstwie bez sekcji FP4 ladowac sie z cache jako czyste 2-bit.

PROBLEM. Wlaczenie delta tier zmienia expected_sizes(want_fp4=True), wiec
warstwa, ktorej cache ma tylko plany 2-bit, jest MISS -> loader-skip sie
rozbraja -> requant z checkpointu. A requant JEDNEJ warstwy z FP4, gdy 40
innych jest juz rezydentnych, nie miesci sie w 121.6 GiB: zmierzone trzy razy,
zawsze smierc na warstwie 8 przy 0.4-0.8 GiB wolnego (hostguard, exit 137).

Powstaje zakleszczenie: zeby zbudowac brakujace sekcje FP4, trzeba miec
zaladowany model, a zeby zaladowac model, trzeba miec sekcje FP4.

LATKA. Zerwij zakleszczenie per warstwa: gdy pelny odczyt (z FP4) chybi,
sprobuj odczytu bez FP4. Warstwa serwuje wtedy z planow 2-bit i po prostu
nigdy nie jest promowana do FP4 - delta tier obejmuje pozostale. Przy 41 z 46
warstw to utrata ~11% pokrycia promocji, a nie utrata bootu.

Dwa miejsca: probe loader-skipu i wlasciwy odczyt cache.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[PATCH-PARTIAL-FP4]" in src:
    print("juz zalatane")
    sys.exit(0)

# --- 1. probe loader-skipu -------------------------------------------------
OLD1 = """        lidx = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
        if lidx is None or not _pc.cache_has_layer(
                lidx, _pc.expected_sizes(
                    E, N13, K13, N2, K2, want_fp4=moe_w2_delta.enabled())):
            return False"""
NEW1 = """        lidx = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
        if lidx is None:
            return False
        # [PATCH-PARTIAL-FP4] Warstwa bez sekcji FP4 nadal moze sie skipnac
        # jako czyste 2-bit - inaczej requant z checkpointu przy rezydentnych
        # sasiadach wywala boot (zmierzone: smierc na warstwie 8).
        if not _pc.cache_has_layer(lidx, _pc.expected_sizes(
                E, N13, K13, N2, K2, want_fp4=moe_w2_delta.enabled())):
            if not (moe_w2_delta.enabled() and _pc.cache_has_layer(
                    lidx, _pc.expected_sizes(
                        E, N13, K13, N2, K2, want_fp4=False))):
                return False
            logger.warning_once(
                "moe_w2: warstwa %d bez sekcji FP4 w cache - serwuje 2-bit "
                "bez promocji do FP4 (pozostale warstwy dzialaja normalnie)",
                lidx)"""

# --- 2. wlasciwy odczyt ----------------------------------------------------
OLD2 = """    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    cached = planes_cache.try_load(lidx, planes_cache.expected_sizes(
        E, N13, K13, N2, K2, want_fp4=tier is not None))
    if cached is None:
        return False"""
NEW2 = """    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    cached = planes_cache.try_load(lidx, planes_cache.expected_sizes(
        E, N13, K13, N2, K2, want_fp4=tier is not None))
    if cached is None and tier is not None:
        # [PATCH-PARTIAL-FP4] Brak sekcji FP4 dla TEJ warstwy: wczytaj same
        # plany 2-bit i nie stage'uj jej do puli. Reszta warstw bez zmian.
        cached = planes_cache.try_load(lidx, planes_cache.expected_sizes(
            E, N13, K13, N2, K2, want_fp4=False))
        if cached is not None:
            tier = None
            logger.warning_once(
                "moe_w2: warstwa %d serwowana jako czyste 2-bit (brak FP4 "
                "w cache) - bez promocji", lidx)
    if cached is None:
        return False"""

for old, new, name in ((OLD1, NEW1, "probe loader-skipu"),
                       (OLD2, NEW2, "odczyt cache")):
    if old not in src:
        print(f"NIE ZNALEZIONO kotwicy: {name}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)

open(path, "w", encoding="utf-8").write(src)
import ast
ast.parse(src)
print("zalatane oba miejsca, skladnia OK")
