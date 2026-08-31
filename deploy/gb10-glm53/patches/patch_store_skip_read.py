#!/usr/bin/env python3
"""LATKA 5: nie czytaj sekcji FP4 z cache'u planow, gdy warstwa jest juz
w trwalym pakiecie store'u.

PROBLEM. _consume_planes_cache zawsze wczytuje komplet szesciu sekcji, gdy
delta tier jest wlaczony - takze wtedy, gdy dana warstwa lezy juz w pliku
pakietu na NVMe. Ten odczyt jest bezuzyteczny: _stage_fp4_host -> store.
add_layer sprawdza `if layer_key in self._present: return` i wraca, NIC nie
robiac. Placimy za to dwa razy:

  * I/O: 3.38 GiB na warstwe (fp13 2.25 + fp2 1.13), czyli 145 GiB
    zbednego czytania z ADATA na kazdy boot 43 warstw;
  * PAMIEC: przejsciowy bufor CPU rosnie z 1.9 do 5.3 GiB na warstwe.

Drugie jest zabojcze na pamieci zunifikowanej. ZMIERZONE 2026-08-31: drugi
przebieg ladowania (glowica MTP, warstwa 45) startuje z 5.71 GiB wolnego,
wiec odkad warstwa 45 ma sekcje FP4, jej odczyt tego zapasu nie miesci -
hostguard ubija przy 0.5 GiB, exit 137. Przed dobudowaniem sekcji FP4 ta
sama warstwa czytala 1.9 GiB i boot przechodzil.

LATKA. Zapytaj store, czy ma juz te warstwe (`layer_key in store` - obie
implementacje backendu to definiuja) i jesli tak, wczytaj same plany 2-bit
oraz pomin staging. Semantycznie bez zmian, bo add_layer i tak byloby
no-opem, a warstwa pozostaje w pelni promowalna: pula czyta wiersze wprost
z pakietu przy promocji, nie z tego bufora.

Fail-closed: kazdy blad sondy oznacza "nie wiem", czyli zachowanie sprzed
latki (pelny odczyt).
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[PATCH-STORE-SKIP-READ]" in src:
    print("juz zalatane")
    sys.exit(0)

# --- 1. sonda obecnosci warstwy w pakiecie ---------------------------------
OLD_HELPER = """def _stage_fp4_host(tier, layer_key: int, fp13, sc13, fp2, sc2) -> None:"""
NEW_HELPER = '''def _fp4_already_packed(tier, layer_key: int) -> bool:
    """[PATCH-STORE-SKIP-READ] Czy warstwa jest juz w trwalym pakiecie?

    Jesli tak, jej sekcje FP4 sa kompletne na dysku i nie trzeba ich ani
    czytac z cache'u planow, ani stage'owac - store.add_layer wrocilby
    natychmiast. Fail-closed: przy jakimkolwiek bledzie zwroc False, czyli
    zachowaj pelny odczyt.
    """
    try:
        store = getattr(tier, "_store", None)
        return store is not None and int(layer_key) in store
    except Exception:  # noqa: BLE001 - sonda, nigdy nie blokuje ladowania
        return False


def _stage_fp4_host(tier, layer_key: int, fp13, sc13, fp2, sc2) -> None:'''

# --- 2. odczyt: pytaj o pakiet przed wyborem zestawu sekcji ----------------
OLD_READ = """    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    cached = planes_cache.try_load(lidx, planes_cache.expected_sizes(
        E, N13, K13, N2, K2, want_fp4=tier is not None))
    if cached is None and tier is not None:"""
NEW_READ = """    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    # [PATCH-STORE-SKIP-READ] Warstwa juz w pakiecie -> czytaj same plany
    # 2-bit. Oszczedza 3.38 GiB odczytu I przejsciowego bufora na warstwe;
    # bez tego drugi przebieg (glowica MTP) nie miesci sie w zapasie.
    need_fp4 = tier is not None and not _fp4_already_packed(tier, layer_key)
    if tier is not None and not need_fp4:
        logger.info_once(
            "moe_w2: sekcje FP4 juz w pakiecie - pomijam ich odczyt "
            "z cache'u planow (warstwa %d)", lidx)
    cached = planes_cache.try_load(lidx, planes_cache.expected_sizes(
        E, N13, K13, N2, K2, want_fp4=need_fp4))
    if cached is None and need_fp4:"""

# Galaz PARTIAL-FP4 musi zerowac takze need_fp4, inaczej ponizszy staging
# siegnie po cached["fp13"], ktorego w tym wariancie nie ma.
OLD_FALLBACK = """        if cached is not None:
            tier = None
            logger.warning_once("""
NEW_FALLBACK = """        if cached is not None:
            tier = None
            need_fp4 = False
            logger.warning_once("""

# --- 3. staging tylko wtedy, gdy faktycznie mamy co stage'owac -------------
OLD_STAGE = """    if tier is not None:
        _stage_fp4_host(tier, layer_key, cached["fp13"].view(E, -1),
                        sc13, cached["fp2"].view(E, -1), sc2)"""
NEW_STAGE = """    if tier is not None and need_fp4:
        _stage_fp4_host(tier, layer_key, cached["fp13"].view(E, -1),
                        sc13, cached["fp2"].view(E, -1), sc2)"""

for old, new, name in ((OLD_HELPER, NEW_HELPER, "sonda _fp4_already_packed"),
                       (OLD_READ, NEW_READ, "odczyt cache"),
                       (OLD_FALLBACK, NEW_FALLBACK, "galaz PARTIAL-FP4"),
                       (OLD_STAGE, NEW_STAGE, "staging")):
    if old not in src:
        print(f"NIE ZNALEZIONO kotwicy: {name}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)

open(path, "w", encoding="utf-8").write(src)
import ast
ast.parse(src)
print("zalatane cztery miejsca, skladnia OK")
