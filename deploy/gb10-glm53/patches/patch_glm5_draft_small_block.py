#!/usr/bin/env python3
"""LATKA 8: maly blok KV dla draftera DFlash2 na GLM-5.3-Flash przy TP=1.

KONTEKST. Overlay DFlash2 (tonyd2wild, patch_glm5_drafter_group.py) dodaje
do sciezki `_get_kv_cache_groups_glm5_next` osobna grupe KV dla piecu warstw
SWA draftera. Ma dwa tryby:

  * EXACT FIT - blok draftera przeskalowany tak, by jego strona byla RÓWNA
    stronie MLA; drafter jedzie wtedy na tensorach MLA (zero bajtow).
    Warunek: fit_block = mla_page / bajty_na_token_draftera musi byc
    wielokrotnoscia 64 i dzielic sie wzajemnie z blokiem MLA.
  * STANDALONE - drafter zachowuje swoj spec (blok = cache_config.block_size)
    i dostaje wlasne tensory rozmiaru draft_page * num_blocks.

NA JEDNYM GB10 EXACT FIT JEST ARYTMETYCZNIE NIEMOZLIWY. Strona MLA na tej
sciezce to 656 B/token (fp8_ds_mla z dopchanym latentem 576), a
656 = 16 * 41, gdzie 41 jest pierwsze. fit_block = blok * 41 / (2048/16)
przy fp8 albo blok * 41 / 256 przy bf16 - zeby bylo wielokrotnoscia 64,
blok musialby byc wielokrotnoscia 8192 (fp8) lub 16384 (bf16), a blok
podyktowany stanem KDA to 7168. Zostaje STANDALONE.

PROBLEM ZE STANDALONE JAK JEST. Blok draftera = blok modelu = 7168 tokenow,
wiec jedna strona draftera to 7168 * 2048 B = 14.7 MB na warstwe, razy 5
warstw = 73 MB doliczane do KAZDEGO bloku puli. Blok MLA kosztuje ~54 MB
(11 * 4.7 MB + indexer), wiec pojemnosc KV spada 2.35x: z 473K tokenow na
~200K. Przy oknie 200K to concurrency 1.0 - za malo.

LATKA. W trybie STANDALONE przeskaluj blok draftera na MALY blok: najwiekszy
dzielnik bloku MLA, ktory jest wielokrotnoscia 64 (kazdy calkowity rozmiar
bloku kernela SWA: 16/32/64 dzieli go bez reszty, wiec select_common_block_size
zawsze znajdzie czysty podzial) i nie przekracza 1024. Dla bloku 7168 to
1024 (7168 = 7 * 1024): strona 2 MiB na warstwe, +10 MiB na blok puli
(-16% pojemnosci zamiast -58%). Drafter z oknem 2048 trzyma
cdiv(2047 + in_flight, 1024) + 1 = 5 blokow na request; kazdy z nich to
jeden id z puli, czyli 7168 tokenow MLA - to drugi, mniejszy koszt.

Dlaczego mniejszy blok jest bezpieczny: widok tensora draftera jest CIAGLY
(brak page_size_padded), wiec podzial na bloki kernela jest poprawny dla
kazdego rozmiaru; scheduler liczy LCM blokow grup (7168, bo 1024 | 7168),
a hash prefix-cache GCD (1024) - dokladnie tak, jak w trybie exact-fit
overlayu, gdzie blok draftera (2304) tez rozni sie od bloku MLA (4608).

Knob: VLLM_GLM5_DRAFT_BLOCK=<int> wymusza konkretny blok (musi dzielic blok
MLA i byc wielokrotnoscia 16); pusty = automat opisany wyzej; 0 = zachowanie
overlayu bez zmian (blok = blok modelu).

Uzycie: python3 patch_glm5_draft_small_block.py <kv_cache_utils.py>
Kotwiczy sie w kodzie ZOSTAWIONYM przez patch_glm5_drafter_group.py, wiec
kolejnosc ma znaczenie. Idempotentna, fail-loud.
"""
import ast
import sys

MARK = "[PATCH-DRAFT-SMALL-BLOCK]"

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if MARK in src:
    print("juz zalatane")
    sys.exit(0)

OLD_STANDALONE = """\
            # STANDALONE: the drafter's geometry cannot exactly fill the MLA
            # page; keep its spec as-is and give its layers compact tensors
            # of their own (emitted in get_kv_cache_config_from_groups and
            # charged in the per-block cost).
            new_draft_specs = dict(draft_specs)
"""
NEW_STANDALONE = """\
            # STANDALONE: the drafter's geometry cannot exactly fill the MLA
            # page; its layers get compact tensors of their own (emitted in
            # get_kv_cache_config_from_groups and charged in the per-block
            # cost). [PATCH-DRAFT-SMALL-BLOCK]: with a SMALL block, so that
            # the per-pool-block charge is a few MiB instead of a full
            # model-block page per drafter layer (see patch docstring).
            small_block = _glm5_draft_small_block(mla_block)
            if small_block:
                new_draft_specs = {
                    name: replace(s, block_size=small_block)
                    for name, s in draft_specs.items()
                }
                logger.info(
                    "GLM5Next drafter group: standalone tensors with block "
                    "%d (MLA block %d, %d B/token, %.1f MiB/page/layer)",
                    small_block,
                    mla_block,
                    draft_bytes_per_token,
                    small_block * draft_bytes_per_token / 2**20,
                )
            else:
                new_draft_specs = dict(draft_specs)
"""
assert src.count(OLD_STANDALONE) == 1, "kotwica STANDALONE (overlay tonyd2wild) nie znaleziona"
src = src.replace(OLD_STANDALONE, NEW_STANDALONE)

OLD_DEF = "def _get_kv_cache_groups_glm5_next(\n"
NEW_DEF = '''def _glm5_draft_small_block(mla_block: int) -> int:
    """[PATCH-DRAFT-SMALL-BLOCK] Block size for a standalone drafter group.

    Largest divisor of the MLA block that is a multiple of 64 and at most
    1024 (env VLLM_GLM5_DRAFT_BLOCK overrides; 0 disables). Returns 0 when no
    such divisor exists, which keeps the overlay's original behaviour.
    """
    env = os.getenv("VLLM_GLM5_DRAFT_BLOCK", "").strip()
    if env:
        forced = int(env)
        if forced <= 0:
            return 0
        if mla_block % forced != 0 or forced % 16 != 0:
            raise ValueError(
                f"VLLM_GLM5_DRAFT_BLOCK={forced} must divide the MLA block "
                f"({mla_block}) and be a multiple of 16"
            )
        return forced
    cap = min(1024, mla_block)
    for cand in range(cap - cap % 64, 63, -64):
        if mla_block % cand == 0:
            return cand
    return 0


def _get_kv_cache_groups_glm5_next(
'''
assert src.count(OLD_DEF) == 1
src = src.replace(OLD_DEF, NEW_DEF)

ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print("OK: maly blok draftera (LATKA 8) nalozony na", path)
