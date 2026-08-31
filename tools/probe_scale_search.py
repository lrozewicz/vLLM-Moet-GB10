#!/usr/bin/env python3
"""Ile realnie tracimy na heurystyce skali AbsMax w budowniczym planów 2-bit?

TEZA (research 2026-08-31, sekcja 2.1). Kwantyzator liczy skalę jako
`exp = round(log2(amax / 6.0))`, czyli normalizuje maksimum bloku 32 do ~6,
podczas gdy najwyższy poziom codebooka to **4**. Największa waga w bloku jest
więc odtwarzana z błędem ~-33%, a wagi bliskie zera (większość) jako +-1, czyli
z wielokrotnym przeszacowaniem. Skala jest jedynym wolnym parametrem tego
formatu i jest liczbą CAŁKOWITĄ (UE8M0 = potęga dwójki), więc jej "wyszukanie"
to przejrzenie kilku kandydatów wokół obecnego wyboru.

CO TO MIERZY. Dla kilku ekspertów jednej warstwy:
  * błąd wzgl. L2 przy obecnej skali (amax/6),
  * błąd przy skali wybranej per blok tak, by minimalizować L2,
  * ile bloków w ogóle zmienia wykładnik i o ile.

Format wyniku, liczba bitów, kernele i rozmiar cache'u **nie zmieniają się**
- to ta sama siatka {-4,-1,1,4} x 2^exp, inaczej ustawiona. Jeśli zysk jest
realny, jest darmowy przy serwowaniu i kosztuje tylko przebudowę cache'u.

Liczone na CPU w f64: wolno, ale nie dotyka GPU ani budżetu pamięci boksu,
więc można to puścić przy działającym serwerze.

Uzycie:  python3 probe_scale_search.py --layer 20 --experts 8
"""

import argparse
import json
import os
import re
import time

import torch
from safetensors import safe_open

# Siatka magnitud e2m1 i punkty środkowe - dokładnie jak w moe_w2_planes.
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                         dtype=torch.float64)
# mag e2m1 {0,.5,1,1.5,2,3,4,6} -> poziom {1,1,1,1,1,4,4,4}
_MAG_TO_LEVEL = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
                             dtype=torch.float64)


def quant_err(wb: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
    """Błąd L2 na blok dla zadanego wykładnika. wb: [B,32] f64, exp: [B]."""
    u = wb / torch.exp2(exp).unsqueeze(1)
    mag = torch.bucketize(u.abs(), _E2M1_MID)
    lvl = _MAG_TO_LEVEL[mag] * torch.sign(u)
    # sign(0) = 0, a kwantyzator mapuje zera na +-1; bierzemy +1 (bias tej
    # decyzji jest osobnym tematem, tu liczy się skala)
    lvl = torch.where(u == 0, torch.ones_like(lvl), lvl)
    rec = lvl * torch.exp2(exp).unsqueeze(1)
    return ((wb - rec) ** 2).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--radius", type=int, default=2,
                    help="ilu kandydatow wykladnika w kazda strone")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(os.path.join(a.model, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    pat = re.compile(rf"\.layers\.{a.layer}\.mlp\.experts\.0\.gate_proj\.weight$")
    pref = None
    for k in wmap:
        if pat.search(k):
            pref = k[:k.index(".mlp.experts.")] + ".mlp.experts."
            break
    if pref is None:
        raise SystemExit(f"nie ma warstwy {a.layer}")

    handles = {}

    def get(key):
        sh = wmap[key]
        if sh not in handles:
            handles[sh] = safe_open(os.path.join(a.model, sh),
                                    framework="pt", device="cpu")
        return handles[sh].get_tensor(key)

    tot_cur = tot_best = tot_pow = 0.0
    shifts: dict[int, int] = {}
    t0 = time.time()
    for e in range(a.experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = get(f"{pref}{e}.{proj}.weight")
            s = get(f"{pref}{e}.{proj}.weight_scale_inv")
            N, K = w.shape
            # dekwantyzacja bloku 128x128 -> f64, jak fp8_block_to_codes_scales
            wf = w.to(torch.float64)
            sf = s.to(torch.float64)
            sc = (sf.repeat_interleave(128, 0)[:N]
                    .repeat_interleave(128, 1)[:, :K])
            wd = (wf * sc).view(-1, 32)

            amax = wd.abs().amax(dim=1)
            exp0 = torch.where(amax > 0,
                               torch.round(torch.log2(amax / 6.0 + 1e-30)),
                               torch.full_like(amax, -127.0)).clamp(-127., 127.)
            err_cur = quant_err(wd, exp0)
            best = err_cur.clone()
            best_exp = exp0.clone()
            for d in range(-a.radius, a.radius + 1):
                if d == 0:
                    continue
                cand = (exp0 + d).clamp(-127., 127.)
                err = quant_err(wd, cand)
                better = err < best
                best = torch.where(better, err, best)
                best_exp = torch.where(better, cand, best_exp)

            tot_cur += float(err_cur.sum())
            tot_best += float(best.sum())
            tot_pow += float((wd ** 2).sum())
            for d, n in zip(*torch.unique(best_exp - exp0,
                                          return_counts=True)):
                shifts[int(d)] = shifts.get(int(d), 0) + int(n)
        print(f"  ekspert {e}: {time.time() - t0:.0f}s", flush=True)

    rel_cur = (tot_cur / tot_pow) ** 0.5
    rel_best = (tot_best / tot_pow) ** 0.5
    n_blocks = sum(shifts.values())
    print()
    print(f"warstwa {a.layer}, {a.experts} ekspertow, {n_blocks} blokow po 32")
    print(f"  rel-RMS obecna skala (amax/6) : {rel_cur:.6f}")
    print(f"  rel-RMS skala szukana         : {rel_best:.6f}")
    print(f"  redukcja bledu                : {100 * (1 - rel_best / rel_cur):.2f}%")
    print("  rozklad przesuniecia wykladnika:")
    for d in sorted(shifts):
        print(f"    {d:+d}: {shifts[d]:>10d}  ({100 * shifts[d] / n_blocks:5.2f}%)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(layer=a.layer, experts=a.experts,
                           rel_cur=rel_cur, rel_best=rel_best,
                           shifts={str(k): v for k, v in shifts.items()}), f,
                      indent=1)
        print(f"zapisano {a.out}")


if __name__ == "__main__":
    main()
