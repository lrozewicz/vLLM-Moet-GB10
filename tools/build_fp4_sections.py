#!/usr/bin/env python3
"""Dobuduj brakujace sekcje FP4 (fp13/fp2) w cache'u planow W2 - offline,
warstwa po warstwie, BEZ ladowania modelu.

PO CO. Cache planes-cache-fp4 ma komplet planow 2-bit dla wszystkich 43
warstw eksperckich, ale warstwy 8, 9 i 45 nie maja sekcji FP4 (layer{L}.
fp13.bin / fp2.bin). Warstwa bez sekcji FP4 serwuje jako czyste 2-bit i jest
wykluczona z promocji do puli FP4 - a 45 to glowica MTP, wiec traci na tym
takze akceptacja draftu. Zbudowanie tych sekcji nie kosztuje ANI BAJTA
pamieci przy serwowaniu: pula FP4 ma staly rozmiar (VLLM_MOE_W2_DELTA_GB),
zmienia sie tylko to, ile warstw moze sie o nia ubiegac.

DLACZEGO OSOBNY SKRYPT, A NIE BOOT. Requant warstwy z FP4, gdy 40 innych
jest juz rezydentnych, umiera przy 0.4-0.8 GiB wolnego (zmierzone trzy razy,
zawsze na warstwie 8) - stad zakleszczenie opisane w patch_partial_fp4.py.
Tutaj w pamieci siedzi JEDNA warstwa naraz (~7 GiB wag FP8 + ~5 GiB planow),
wiec 121.6 GiB to ocean.

POPRAWNOSC. Skrypt nie odtwarza kwantyzatora - importuje DOKLADNIE ten sam
modul moe_w2_planes z obrazu i te same trzy funkcje, ktorych uzywa
build_layer_planes_fp8 (fp8_block_to_codes_scales -> pack_fragment_major /
pack_scales / pack_fp4_fragment_major). Jedyne, co odtwarza, to zlozenie
w13 = [gate_proj; up_proj] z checkpointu, ktore normalnie robi loader
FusedMoE. I wlasnie to jest sprawdzane: dla kazdej warstwy przeliczamy TAKZE
plany 2-bit i porownujemy je bajt w bajt z tym, co juz lezy w cache'u.
Zgodnosc czterech sekcji 2-bit dowodzi, ze wejscie i pipeline sa identyczne,
wiec sekcje FP4 policzone przy okazji sa tymi samymi, ktore wyprodukowalby
boot. Rozjazd = przerwanie bez zapisu.

Uzycie (w kontenerze z obrazem serwera, model niezaladowany):
  python3 build_fp4_sections.py --layers 10 --dry-run   # kontrola na warstwie
                                                        # ktora MA juz FP4
  python3 build_fp4_sections.py --layers 8,9,45         # wlasciwa budowa
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open

PLANES_MODULE = ("/usr/local/lib/python3.12/dist-packages/vllm/"
                 "model_executor/layers/quantization/utils/moe_w2_planes.py")
CHUNK = 8               # ekspertow miedzy empty_cache() - jak w cubicie
HASH_CHUNK = 1 << 26    # 64 MiB


def load_planes_module(path):
    """Zaladuj moe_w2_planes PO SCIEZCE, bez importu pakietu vllm.

    Modul importuje tylko os i torch (sprawdzone), wiec nie potrzebuje
    inicjalizacji vllm - a jej pominiecie usuwa z tej sciezki cala warstwe
    wykrywania platformy i konfiguracji, ktora nie ma tu nic do roboty.
    """
    spec = importlib.util.spec_from_file_location("moe_w2_planes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(HASH_CHUNK)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_tensor(t):
    """sha256 bajtow tensora u8, kawalkami - bez kopii calosci na host."""
    h = hashlib.sha256()
    flat = t.detach().reshape(-1)
    for i in range(0, flat.numel(), HASH_CHUNK):
        h.update(flat[i:i + HASH_CHUNK].cpu().numpy().tobytes())
    return h.hexdigest()


class Shards:
    """Leniwe uchwyty do shardow safetensors, z mapy wag."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir,
                               "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.handles = {}

    def get(self, key):
        shard = self.weight_map[key]
        h = self.handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.model_dir, shard),
                          framework="pt", device="cpu")
            self.handles[shard] = h
        return h.get_tensor(key)

    def close(self):
        self.handles.clear()


def layer_prefix(weight_map, layer):
    """Prefiks kluczy ekspertow tej warstwy (checkpoint jest multimodalny,
    wiec stack jezykowy siedzi pod model.language_model.*)."""
    pat = re.compile(rf"\.layers\.{layer}\.mlp\.experts\.0\.gate_proj\.weight$")
    for k in weight_map:
        if pat.search(k):
            return k[:k.index(".mlp.experts.")] + ".mlp.experts."
    raise SystemExit(f"nie znalazlem ekspertow warstwy {layer} w indeksie")


def build_layer(planes, shards, layer, dev, n_experts, block_shape):
    """Zbuduj komplet szesciu sekcji dla jednej warstwy. Zwraca dict."""
    pref = layer_prefix(shards.weight_map, layer)
    out = None
    t0 = time.time()
    for e in range(n_experts):
        if e % CHUNK == 0:
            # Ten sam powod co w build_layer_planes_fp8: dequant FP8 jest
            # f64-ciezki i alokator w kolko dokłada nowe bloki zamiast
            # uzywac starych. Na pamieci zunifikowanej to nie jest kosmetyka.
            torch.cuda.empty_cache()
        g = shards.get(f"{pref}{e}.gate_proj.weight")
        u = shards.get(f"{pref}{e}.up_proj.weight")
        d = shards.get(f"{pref}{e}.down_proj.weight")
        gs = shards.get(f"{pref}{e}.gate_proj.weight_scale_inv")
        us = shards.get(f"{pref}{e}.up_proj.weight_scale_inv")
        ds = shards.get(f"{pref}{e}.down_proj.weight_scale_inv")
        # Zlozenie jak w loaderze FusedMoE: w1 (gate) na pozycji 0, w3 (up)
        # na pozycji I. Zgodnosc tego zalozenia weryfikuje pozniej
        # porownanie planow 2-bit z cache'em.
        w13 = torch.cat((g.to(dev, non_blocking=True),
                         u.to(dev, non_blocking=True)), dim=0)
        s13 = torch.cat((gs.to(dev, non_blocking=True),
                         us.to(dev, non_blocking=True)), dim=0)
        w2 = d.to(dev, non_blocking=True)
        s2 = ds.to(dev, non_blocking=True)
        if out is None:
            N13, K13 = w13.shape
            N2, K2 = w2.shape
            assert w13.dtype == torch.float8_e4m3fn, w13.dtype
            out = dict(
                planes13=torch.empty(n_experts, N13 * K13 // 4,
                                     dtype=torch.uint8, device=dev),
                sc13=torch.empty(n_experts, N13 * K13 // 32,
                                 dtype=torch.uint8, device=dev),
                planes2=torch.empty(n_experts, N2 * K2 // 4,
                                    dtype=torch.uint8, device=dev),
                sc2=torch.empty(n_experts, N2 * K2 // 32,
                                dtype=torch.uint8, device=dev),
                fp13=torch.empty(n_experts, N13 * K13 // 2,
                                 dtype=torch.uint8, device=dev),
                fp2=torch.empty(n_experts, N2 * K2 // 2,
                                dtype=torch.uint8, device=dev),
            )
            print(f"  warstwa {layer}: E={n_experts} w13=[{N13},{K13}] "
                  f"w2=[{N2},{K2}]", flush=True)
        codes, sbytes, nib = planes.fp8_block_to_codes_scales(
            w13, s13, block_shape=block_shape, want_nibbles=True)
        out["planes13"][e] = planes.pack_fragment_major(codes)
        out["sc13"][e] = planes.pack_scales(sbytes)
        out["fp13"][e] = planes.pack_fp4_fragment_major(nib)
        codes, sbytes, nib = planes.fp8_block_to_codes_scales(
            w2, s2, block_shape=block_shape, want_nibbles=True)
        out["planes2"][e] = planes.pack_fragment_major(codes)
        out["sc2"][e] = planes.pack_scales(sbytes)
        out["fp2"][e] = planes.pack_fp4_fragment_major(nib)
        if e and e % 64 == 0:
            free = torch.cuda.mem_get_info()[0] / 2**30
            print(f"    ekspert {e}/{n_experts}  {time.time() - t0:.0f}s  "
                  f"wolne {free:.1f} GiB", flush=True)
    torch.cuda.empty_cache()
    print(f"  policzone w {time.time() - t0:.0f}s", flush=True)
    return out


def write_part(cache_dir, layer, part, tensor):
    """Zapis atomowy: tmp + fsync + rename (jak writer w planes_cache)."""
    path = os.path.join(cache_dir, f"layer{layer}.{part}.bin")
    tmp = f"{path}.tmp.{os.getpid()}"
    flat = tensor.detach().reshape(-1)
    with open(tmp, "wb") as f:
        for i in range(0, flat.numel(), HASH_CHUNK):
            f.write(flat[i:i + HASH_CHUNK].cpu().numpy().tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--cache", default="/packs/planes-cache-fp4/v2/"
                                       "f03c405bb24af556422b/tp1-rank0")
    ap.add_argument("--layers", required=True,
                    help="lista warstw po przecinku, np. 8,9,45")
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--dry-run", action="store_true",
                    help="tylko policz i porownaj, nic nie zapisuj")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    meta_path = os.path.join(args.cache, "meta.json")
    meta = json.load(open(meta_path))
    print(f"meta: {meta}", flush=True)
    # Sekcje FP4 maja rozmiar n*k/2 tylko w trybie NIEdzielonym; przy
    # fp4_split ("w4q") to plany kwintalne 5/16 i caly ten skrypt liczylby
    # nie to, co trzeba.
    if meta.get("fp4_split"):
        raise SystemExit("cache jest w trybie split (w4q) - ten skrypt "
                         "buduje wylacznie pelne nibble (fp4_split=false)")
    if meta.get("quantizer_abi") != "tsym4-e8m0-fragmajor-a32-v2":
        raise SystemExit(f"nieznane ABI kwantyzatora: {meta}")
    if (meta.get("world"), meta.get("rank")) != (1, 0):
        raise SystemExit(f"ten skrypt zaklada TP=1 rank=0, a meta mowi {meta}")
    if os.environ.get("VLLM_MOE_W2_ZERO_MODE", "auto") != meta.get(
            "zero_mode", "auto"):
        raise SystemExit("VLLM_MOE_W2_ZERO_MODE inny niz w meta.json")

    planes = load_planes_module(PLANES_MODULE)
    if planes.zero_mode() != meta.get("zero_mode"):
        raise SystemExit(f"zero_mode {planes.zero_mode()} != {meta}")
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    qc = cfg.get("quantization_config") or cfg["text_config"][
        "quantization_config"]
    block_shape = tuple(qc["weight_block_size"])
    print(f"blok FP8: {block_shape}", flush=True)

    dev = torch.device("cuda")
    shards = Shards(args.model)
    failures = []
    for layer in layers:
        print(f"=== warstwa {layer} ===", flush=True)
        built = build_layer(planes, shards, layer, dev, args.experts,
                            block_shape)
        # 1. WERYFIKACJA: sekcje 2-bit musza sie zgadzac bajt w bajt z tym,
        #    co juz jest w cache'u. To jest caly dowod poprawnosci.
        ok = True
        for part in ("planes13", "sc13", "planes2", "sc2"):
            path = os.path.join(args.cache, f"layer{layer}.{part}.bin")
            if not os.path.exists(path):
                print(f"  {part}: BRAK w cache'u - pomijam porownanie")
                continue
            want = os.path.getsize(path)
            got = built[part].numel()
            if want != got:
                print(f"  {part}: ROZMIAR {got} != {want}")
                ok = False
                continue
            a, b = sha256_file(path), sha256_tensor(built[part])
            same = a == b
            ok &= same
            print(f"  {part}: {'ZGODNE' if same else 'ROZJAZD'} "
                  f"({got} B, {b[:16]})", flush=True)
        # 2. Sekcje FP4: jesli juz sa (warstwa kontrolna), porownaj takze je.
        for part in ("fp13", "fp2"):
            path = os.path.join(args.cache, f"layer{layer}.{part}.bin")
            if os.path.exists(path):
                a, b = sha256_file(path), sha256_tensor(built[part])
                same = a == b
                ok &= same
                print(f"  {part}: {'ZGODNE' if same else 'ROZJAZD'} "
                      f"(juz w cache'u, {b[:16]})", flush=True)
        if not ok:
            failures.append(layer)
            print(f"  warstwa {layer}: NIE ZAPISUJE (rozjazd)", flush=True)
        elif args.dry_run:
            print(f"  warstwa {layer}: dry-run, nic nie zapisuje", flush=True)
        else:
            for part in ("fp13", "fp2"):
                path = os.path.join(args.cache, f"layer{layer}.{part}.bin")
                if os.path.exists(path):
                    print(f"  {part}: juz jest, nie nadpisuje", flush=True)
                    continue
                t = time.time()
                p = write_part(args.cache, layer, part, built[part])
                print(f"  {part}: zapisane {os.path.getsize(p)} B "
                      f"w {time.time() - t:.0f}s", flush=True)
        del built
        torch.cuda.empty_cache()
    shards.close()
    if failures:
        print(f"ROZJAZD na warstwach: {failures}", file=sys.stderr)
        return 1
    print("gotowe", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
