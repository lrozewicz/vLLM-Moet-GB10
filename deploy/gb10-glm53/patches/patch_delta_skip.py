#!/usr/bin/env python3
"""LATKA 3: nie promuj warstw, ktorych NIE MA w pakiecie FP4.

PROBLEM. DeltaTier iteruje kandydatow przez `for li in range(self.n_layers)` -
na slepo, bez sprawdzenia, czy warstwa w ogole ma wiersze w pakiecie. Gdy
kilka warstw serwuje jako czyste 2-bit (latka 2), ich wiersze w pliku pack sa
DZIURAMI (sparse). Promocja czyta smieci i konczy sie
`CUDA error: an illegal memory access was encountered` na pierwszej generacji.
Zmierzone: HTTP 500 na kazdym zadaniu, EngineDeadError.

LATKA. Store zna prawde o zawartosci (`__contains__` czyta sidecar z lista
zapisanych warstw). Filtrujemy nia OBA miejsca promocji. Warstwa bez FP4
zostaje na 2-bit zamiast wywracac silnik.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[PATCH-DELTA-SKIP]" in src:
    print("juz zalatane"); sys.exit(0)

OLD1 = """    def _promote(self, li, ei, slot):
        try:
            row = self._store.rows_for([(li, ei)])[0]"""
NEW1 = """    def _promote(self, li, ei, slot):
        # [PATCH-DELTA-SKIP] Warstwa bez wierszy w pakiecie = dziura w pliku.
        # Czytanie jej daje illegal memory access; zostawiamy ja na 2-bit.
        try:
            if li not in self._store:
                return
        except BaseException:
            pass
        try:
            row = self._store.rows_for([(li, ei)])[0]"""

OLD2 = """        if not candidates:
            return 0
        slots = self._take_slots_batch(len(candidates))"""
NEW2 = """        # [PATCH-DELTA-SKIP] jak wyzej, ale dla sciezki wsadowej
        try:
            candidates = [c for c in candidates
                          if (c[0] if isinstance(c, tuple) else c) in self._store]
        except BaseException:
            pass
        if not candidates:
            return 0
        slots = self._take_slots_batch(len(candidates))"""

for old, new, name in ((OLD1, NEW1, "_promote"), (OLD2, NEW2, "_promote_batch")):
    if old not in src:
        print(f"NIE ZNALEZIONO: {name}", file=sys.stderr); sys.exit(1)
    src = src.replace(old, new, 1)

open(path, "w", encoding="utf-8").write(src)
import ast; ast.parse(src)
print("zalatane oba miejsca promocji, skladnia OK")
