#!/usr/bin/env python3
"""LATKA 9b: builder FlashInfer dla zwyklych warstw przy globalnym fp8_ds_mla.

Uzupelnienie LATKI 9. FlashInferMetadataBuilder.__init__ czyta
`cache_config.cache_dtype` (globalny, z komentarzem "cannot use
kv_cache_spec.dtype"), a canonicalizacja sparse-MLA ustawia go na
`fp8_ds_mla` - format, ktory get_dtype_for_flashinfer nie zna:

    ValueError: Unrecognized dtype: fp8_ds_mla   (boot 3, 02.09 -> 03.09)

Ten builder powstaje w tym modelu tylko dla warstw SWA draftera (warstwy MLA
maja wlasny backend sparse-MLA), a ich KV to zwykly fp8 e4m3 (LATKA 9). Mapa:
fp8_ds_mla -> fp8_e4m3, wylacznie w tym miejscu.

Uzycie: python3 patch_flashinfer_builder_dtype.py <flashinfer.py>. Idempotentna.
"""
import ast
import sys

MARK = "[PATCH-FI-BUILDER-DTYPE]"
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if MARK in src:
    print("juz zalatane")
    sys.exit(0)

OLD = "            self.cache_dtype = self.cache_config.cache_dtype\n"
NEW = """            self.cache_dtype = self.cache_config.cache_dtype
            # [PATCH-FI-BUILDER-DTYPE] the sparse-MLA canonicalisation stamps
            # fp8_ds_mla on the whole config; a plain (non-MLA) FlashInfer
            # layer stores ordinary fp8 e4m3 records.
            if self.cache_dtype == "fp8_ds_mla":
                self.cache_dtype = "fp8_e4m3"
"""
assert src.count(OLD) == 1, "kotwica cache_dtype w builderze nie znaleziona"
src = src.replace(OLD, NEW)
ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print("OK: LATKA 9b nalozona na", path)
