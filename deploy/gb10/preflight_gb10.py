#!/usr/bin/env python3
"""
preflight_gx10.py — czy vLLM-Moet ma szansę ruszyć na GX10 (GB10 / sm_121)?

Odpowiada na jedno decydujące pytanie, zanim ściągniesz 167 GB checkpointu:

    czy sterownik CUDA załaduje ręcznie pisany cubin sm_120 na sm_121?

Cubiny w kernels/cubins-sm120/ to czysty SASS bez PTX-a — nie ma fallbacku
przez JIT. Albo driver przyjmie ELF-a z inną compute capability (rodzinowa
kompatybilność sm_120 -> sm_121), albo dostaniesz CUDA_ERROR_NO_BINARY_FOR_GPU
(209) i temat jest zamknięty w dwie minuty zamiast w weekend.

Reszta etapów to kontekst: co jeszcze trzeba będzie tknąć, ile pamięci
zostanie i czy poprawki SM120 z patcha w ogóle się aktywują na 12.1.

Skrypt NIE wymaga torcha ani vllm — te etapy same się pominą. Wymaga tylko
python3 i libcuda.so.1.

Użycie (w kontenerze vllm-moet):

    docker run --rm --gpus all --ipc host \
      -v /sciezka/do/preflight_gx10.py:/preflight.py:ro \
      --entrypoint python3 vllm-moet-sm120:v024 /preflight.py

albo na hoście, jeśli masz tam cubiny:

    python3 preflight_gx10.py --cubins /sciezka/do/kernels/cubins-sm120

Kody wyjścia:  0 = przejdź dalej   1 = GATE 1 padł (koniec tematu)
               2 = GATE 1 przeszedł, ale coś innego wymaga uwagi
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import struct
import sys

# ---------------------------------------------------------------- raport

_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


RESULTS: list[tuple[str, str, str]] = []   # (status, etap, komentarz)

PASS, FAIL, WARN, SKIP, INFO = "PASS", "FAIL", "WARN", "SKIP", "INFO"
_COLOR = {PASS: "32", FAIL: "31;1", WARN: "33", SKIP: "90", INFO: "36"}


def record(status: str, stage: str, note: str = "") -> None:
    RESULTS.append((status, stage, note))


def head(title: str) -> None:
    print()
    print(_c("1", f"── {title} " + "─" * max(0, 66 - len(title))))


def line(status: str, msg: str) -> None:
    print(f"  [{_c(_COLOR[status], status):>4}] {msg}" if not _TTY
          else f"  [{_c(_COLOR[status], status)}] {msg}")


# ---------------------------------------------------------------- CUDA driver

class Cuda:
    """Minimalne wiązanie driver API — bez torcha, bez cudart."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        L = self.lib
        L.cuInit.argtypes = [ctypes.c_uint]
        L.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                      ctypes.c_int]
        L.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int, ctypes.c_int]
        L.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                          ctypes.c_int]
        L.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.c_uint, ctypes.c_int]
        L.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(ctypes.c_size_t)]
        L.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_char_p]
        L.cuModuleUnload.argtypes = [ctypes.c_void_p]
        L.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.c_void_p, ctypes.c_char_p]
        L.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                         ctypes.c_int, ctypes.c_void_p]
        L.cuGetErrorName.argtypes = [ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_char_p)]

    def err(self, rc: int) -> str:
        p = ctypes.c_char_p()
        if self.lib.cuGetErrorName(rc, ctypes.byref(p)) == 0 and p.value:
            return f"{rc} ({p.value.decode()})"
        return str(rc)

    # atrybuty urządzenia
    ATTR_CC_MAJOR = 75
    ATTR_CC_MINOR = 76
    ATTR_MP_COUNT = 16
    ATTR_CLOCK_KHZ = 13
    ATTR_INTEGRATED = 18
    ATTR_UNIFIED_ADDR = 41
    # atrybuty funkcji
    FUNC_MAX_THREADS = 0
    FUNC_SHARED_BYTES = 1
    FUNC_NUM_REGS = 4
    FUNC_PTX_VERSION = 5
    FUNC_BINARY_VERSION = 6

    def dev_attr(self, attr: int, dev: int) -> int:
        v = ctypes.c_int()
        rc = self.lib.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)
        return v.value if rc == 0 else -1

    def func_attr(self, attr: int, fn: ctypes.c_void_p) -> int:
        v = ctypes.c_int()
        rc = self.lib.cuFuncGetAttribute(ctypes.byref(v), attr, fn)
        return v.value if rc == 0 else -1


# ---------------------------------------------------------------- ELF cubinu

def elf_info(path: str) -> dict:
    """e_flags + symbole FUNC z cubinu (ELF64 LE), bez cuobjdump."""
    with open(path, "rb") as f:
        b = f.read()
    out: dict = {"size": len(b), "ok": False, "syms": []}
    if len(b) < 64 or b[:4] != b"\x7fELF":
        out["error"] = "nie jest ELF"
        return out
    e_machine, = struct.unpack_from("<H", b, 18)
    e_shoff, = struct.unpack_from("<Q", b, 40)
    e_flags, = struct.unpack_from("<I", b, 48)
    e_shentsize, = struct.unpack_from("<H", b, 58)
    e_shnum, = struct.unpack_from("<H", b, 60)
    out.update(machine=e_machine, flags=e_flags,
               # bity 8..15 niosą numer SM w cubinach nvidii (0x78 == 120);
               # najlepsze przybliżenie bez nvdisasm — wynik cuModuleLoadData
               # niżej jest i tak rozstrzygający
               sm=(e_flags >> 8) & 0xFF, ok=True)

    sections = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        if o + 64 > len(b):
            break
        sh_type, = struct.unpack_from("<I", b, o + 4)
        sh_offset, = struct.unpack_from("<Q", b, o + 24)
        sh_size, = struct.unpack_from("<Q", b, o + 32)
        sh_link, = struct.unpack_from("<I", b, o + 40)
        sh_entsize, = struct.unpack_from("<Q", b, o + 56)
        sections.append((sh_type, sh_offset, sh_size, sh_link, sh_entsize))

    SHT_SYMTAB, STT_FUNC = 2, 2
    for sh_type, off, size, link, entsize in sections:
        if sh_type != SHT_SYMTAB or entsize < 24 or link >= len(sections):
            continue
        _, str_off, str_size, _, _ = sections[link]
        strtab = b[str_off:str_off + str_size]
        for k in range(size // entsize):
            so = off + k * entsize
            st_name, = struct.unpack_from("<I", b, so)
            st_info = b[so + 4]
            if (st_info & 0xF) != STT_FUNC or st_name == 0:
                continue
            end = strtab.find(b"\x00", st_name)
            name = strtab[st_name:end].decode("utf-8", "replace")
            if name and name not in out["syms"]:
                out["syms"].append(name)
    return out


# ---------------------------------------------------------------- etapy

def stage_env() -> None:
    head("A. Środowisko")
    mach = platform.machine()
    line(PASS if mach == "aarch64" else WARN,
         f"architektura hosta: {mach}"
         + ("" if mach == "aarch64" else "  (spodziewane aarch64 na GX10)"))
    record(PASS if mach == "aarch64" else WARN, "A. arch", mach)
    line(INFO, f"python {platform.python_version()}, "
               f"kernel {platform.release()}")

    try:
        import torch  # noqa: PLC0415
        line(INFO, f"torch {torch.__version__}, "
                   f"zbudowany z CUDA {torch.version.cuda}")
        alist = getattr(torch.cuda, "get_arch_list", lambda: [])()
        line(INFO, f"torch arch list: {', '.join(alist) or 'n/d'}")
        if alist and not any(a.startswith(("sm_120", "sm_121")) for a in alist):
            line(WARN, "torch nie ma sm_120 ani sm_121 na liście — jego własne "
                       "kernele mogą nie wystartować")
            record(WARN, "A. torch arch list", "brak sm_12x")
    except Exception as e:  # noqa: BLE001
        line(SKIP, f"torch niedostępny ({type(e).__name__}) — etapy D/E ograniczone")


def stage_device(cu: Cuda) -> tuple[int, int, int]:
    """Zwraca (cc_major, cc_minor, total_mem_bytes)."""
    head("B. GPU")
    rc = cu.lib.cuInit(0)
    if rc != 0:
        line(FAIL, f"cuInit: {cu.err(rc)}")
        record(FAIL, "B. cuInit", cu.err(rc))
        return (0, 0, 0)

    ver = ctypes.c_int()
    cu.lib.cuDriverGetVersion(ctypes.byref(ver))
    line(INFO, f"driver API {ver.value // 1000}.{(ver.value % 1000) // 10}")
    if ver.value < 13000:
        line(WARN, "sm_121 wymaga CUDA 13.0+; ten driver jest starszy")
        record(WARN, "B. wersja CUDA", f"{ver.value}")

    n = ctypes.c_int()
    cu.lib.cuDeviceGetCount(ctypes.byref(n))
    if n.value < 1:
        line(FAIL, "brak widocznych urządzeń CUDA")
        record(FAIL, "B. urządzenia", "0")
        return (0, 0, 0)

    dev = ctypes.c_int()
    cu.lib.cuDeviceGet(ctypes.byref(dev), 0)
    buf = ctypes.create_string_buffer(256)
    cu.lib.cuDeviceGetName(buf, 256, dev)
    name = buf.value.decode("utf-8", "replace")

    major = cu.dev_attr(Cuda.ATTR_CC_MAJOR, dev)
    minor = cu.dev_attr(Cuda.ATTR_CC_MINOR, dev)
    sms = cu.dev_attr(Cuda.ATTR_MP_COUNT, dev)
    integ = cu.dev_attr(Cuda.ATTR_INTEGRATED, dev)
    total = ctypes.c_size_t()
    cu.lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev)

    line(INFO, f"{name} — sm_{major}{minor}, {sms} SM, "
               f"{total.value / 2**30:.1f} GiB, "
               f"{'pamięć zunifikowana' if integ == 1 else 'pamięć dedykowana'}")

    if (major, minor) == (12, 1):
        line(PASS, "sm_121 potwierdzone — to jest przypadek, o który chodzi")
        record(PASS, "B. capability", "sm_121")
    elif (major, minor) == (12, 0):
        line(WARN, "sm_120 — to natywnie wspierana karta, GATE 1 przejdzie "
                   "trywialnie i nic nie udowodni")
        record(WARN, "B. capability", "sm_120 (nie GX10)")
    else:
        line(FAIL, f"sm_{major}{minor} — poza rodziną 12.x, cubiny nie mają szans")
        record(FAIL, "B. capability", f"sm_{major}{minor}")

    ctx = ctypes.c_void_p()
    rc = cu.lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    if rc != 0:
        line(FAIL, f"cuCtxCreate: {cu.err(rc)}")
        record(FAIL, "B. kontekst", cu.err(rc))
        return (major, minor, 0)
    line(PASS, "kontekst CUDA utworzony")
    return (major, minor, total.value)


# cubiny, których DS4-Flash faktycznie potrzebuje przy TP1 na GX10:
#   G1 (x -> w13):  K = hidden_size      = 4096
#   G2 (w2):        K = moe_intermediate = 2048
# (TP2 przesunąłby to na 2048/1024 — patrz --tp)
def critical_cubins(tp: int) -> list[str]:
    k13, k2 = 4096 // tp, 2048 // tp
    names = []
    for k in (k13, k2):
        names += [
            f"moe_w2_mm_k{k}_a32.cubin",         # baza 2-bit
            f"moe_w2_mm_mc4_k{k}_a32.cubin",     # decode, 4 CTA/SM
            f"moe_w2_mm_mc4afrag_k{k}_a32.cubin",  # prefill AFRAG
            f"moe_w4_mm_k{k}_a32.cubin",         # tier FP4 (delta cache)
            f"moe_w4q_mm_k{k}_a32.cubin",        # quintal split-FP4
        ]
    return names


def stage_cubins(cu: Cuda, cubin_dir: str, tp: int) -> bool:
    head("C. GATE 1 — ładowanie cubinów sm_120 (DECYDUJĄCY)")
    if not os.path.isdir(cubin_dir):
        line(FAIL, f"katalog cubinów nie istnieje: {cubin_dir}")
        line(INFO, "podaj go przez --cubins albo VLLM_MOE_W2_CUBIT_DIR")
        record(FAIL, "C. GATE 1", "brak cubinów")
        return False

    all_cubins = sorted(f for f in os.listdir(cubin_dir)
                        if f.endswith(".cubin"))
    if not all_cubins:
        line(FAIL, f"w {cubin_dir} nie ma żadnego .cubin")
        record(FAIL, "C. GATE 1", "katalog pusty")
        return False

    crit = critical_cubins(tp)
    missing = [c for c in crit if c not in all_cubins]
    line(INFO, f"{len(all_cubins)} cubinów w {cubin_dir}; "
               f"krytycznych dla DS4 przy TP{tp}: {len(crit)}")
    if missing:
        line(WARN, f"brak w katalogu: {', '.join(missing)}")
        record(WARN, "C. brakujące cubiny", ", ".join(missing))

    # --- ELF: co te pliki w ogóle deklarują
    probe = elf_info(os.path.join(cubin_dir, all_cubins[0]))
    if probe.get("ok"):
        line(INFO, f"ELF {all_cubins[0]}: e_flags=0x{probe['flags']:08x} "
                   f"-> deklarowane sm_{probe['sm']}   symbole: "
                   f"{', '.join(probe['syms']) or 'brak symtab'}")
        if probe["sm"] == 120:
            line(INFO, "czyste sm_120, zero PTX-a — poniżej sprawdzamy, czy "
                       "driver mimo to je przyjmie")

    ok, bad, funcs_ok = [], [], []
    for fname in all_cubins:
        path = os.path.join(cubin_dir, fname)
        with open(path, "rb") as f:
            data = f.read()
        mod = ctypes.c_void_p()
        rc = cu.lib.cuModuleLoadData(ctypes.byref(mod), data)
        if rc != 0:
            bad.append((fname, cu.err(rc)))
            continue
        ok.append(fname)

        # moduł się załadował — sprawdź, czy da się wyciągnąć kernel
        info = elf_info(path)
        for sym in info.get("syms", []):
            fn = ctypes.c_void_p()
            rc2 = cu.lib.cuModuleGetFunction(ctypes.byref(fn), mod,
                                             sym.encode())
            if rc2 == 0:
                regs = cu.func_attr(Cuda.FUNC_NUM_REGS, fn)
                smem = cu.func_attr(Cuda.FUNC_SHARED_BYTES, fn)
                binv = cu.func_attr(Cuda.FUNC_BINARY_VERSION, fn)
                funcs_ok.append((fname, sym, regs, smem, binv))
        cu.lib.cuModuleUnload(mod)

    if bad:
        f0, e0 = bad[0]
        line(FAIL, f"cuModuleLoadData padło na {len(bad)}/{len(all_cubins)} "
                   f"cubinach — np. {f0}: {e0}")
        if "NO_BINARY" in e0:
            line(INFO, "CUDA_ERROR_NO_BINARY_FOR_GPU = driver nie akceptuje "
                       "ELF-a sm_120 na tym GPU.")
            line(INFO, "Bez PTX-a nie ma fallbacku. vLLM-Moet na tym pudełku "
                       "wymagałby przebudowy cubinów pod sm_121 przez cubit — "
                       "to nie jest 'mały hack'.")
        record(FAIL, "C. GATE 1", f"{len(bad)} cubinów odrzuconych: {e0}")
        return False

    line(PASS, f"cuModuleLoadData: {len(ok)}/{len(all_cubins)} cubinów "
               f"załadowanych bez błędu")
    crit_ok = [c for c in crit if c in ok]
    line(PASS if len(crit_ok) == len(crit) else WARN,
         f"z tego krytycznych dla DS4/TP{tp}: {len(crit_ok)}/{len(crit)}")

    if not funcs_ok:
        line(WARN, "moduły się ładują, ale nie udało się pobrać żadnego "
                   "kernela — sprawdź symtab (cubin bez .symtab?)")
        record(WARN, "C. GATE 1", "load OK, cuModuleGetFunction bez wyniku")
        return True

    line(PASS, f"cuModuleGetFunction + atrybuty OK dla {len(funcs_ok)} kerneli")
    for fname, sym, regs, smem, binv in funcs_ok[:4]:
        line(INFO, f"  {sym:<12} w {fname:<34} regs={regs} smem={smem}B "
                   f"binary_version={binv}")
    if any(r == 64 for _, _, r, _, _ in funcs_ok):
        line(INFO, "  regcount 64 = zakładane 4 CTA/SM — zgodne z MANIFEST.md")

    print()
    line(PASS, _c("1", "GATE 1 ZALICZONY: sm_120 -> sm_121 działa na tym driverze."))
    record(PASS, "C. GATE 1", f"{len(ok)} cubinów, {len(funcs_ok)} kerneli")
    return True


def stage_guards() -> None:
    head("D. Twarde strażniki != (12, 0) w zainstalowanym vLLM")
    try:
        import vllm  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        line(SKIP, f"vllm nie do zaimportowania ({type(e).__name__}) — "
                   "uruchom w kontenerze vllm-moet, żeby to sprawdzić")
        record(SKIP, "D. strażniki", "brak vllm")
        return

    root = os.path.dirname(vllm.__file__)
    line(INFO, f"vllm {getattr(vllm, '__version__', '?')} w {root}")

    import re  # noqa: PLC0415
    pat = re.compile(r"!=\s*\(12,\s*0\)|\(12,\s*0\)\s*!=")
    hits: list[tuple[str, int, str]] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    for i, ln in enumerate(f, 1):
                        if pat.search(ln):
                            hits.append((os.path.relpath(p, root), i,
                                         ln.strip()))
            except OSError:
                continue

    if not hits:
        line(WARN, "nie znalazłem żadnego strażnika — patch nie jest "
                   "nałożony albo wersja się rozjechała")
        record(WARN, "D. strażniki", "0 trafień")
        return

    line(WARN, f"{len(hits)} miejsc do tknięcia, żeby wpuścić sm_121:")
    for rel, i, txt in hits:
        line(INFO, f"  {rel}:{i}  {txt[:78]}")
    line(INFO, "kluczowy: quantization/utils/moe_w2_cubit.py -> _ensure_ready() "
               "(odmawia wprost dla SM121/GB10)")
    line(INFO, "zamiana na `cap[0] != 12` wpuszcza całą rodzinę 12.x")
    record(WARN, "D. strażniki", f"{len(hits)} do zmiany")


def stage_family() -> None:
    head("E. Czy poprawki SM120 z patcha aktywują się na 12.1")
    try:
        from vllm.platforms import current_platform  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        line(SKIP, f"brak vllm.platforms ({type(e).__name__})")
        record(SKIP, "E. family check", "brak vllm")
        return

    fn = getattr(current_platform, "is_device_capability_family", None)
    if fn is None:
        line(WARN, "is_device_capability_family nie istnieje w tej wersji vLLM "
                   "— patch odwołuje się do niej, więc coś się nie zgadza")
        record(WARN, "E. family check", "brak metody")
        return

    try:
        res = bool(fn(120))
    except Exception as e:  # noqa: BLE001
        line(WARN, f"wywołanie rzuciło {type(e).__name__}: {e}")
        record(WARN, "E. family check", "wyjątek")
        return

    cap = current_platform.get_device_capability()
    line(INFO, f"capability wg vLLM: {cap}")
    if res:
        line(PASS, "is_device_capability_family(120) == True  -> poprawki SM120 "
                   "(DeepGEMM raw f32 scales, wyłączony cluster launch w "
                   "cooperative_topk, ścieżki sparse-MLA) WŁĄCZAJĄ SIĘ SAME")
        record(PASS, "E. family check", "True")
    else:
        line(FAIL, "is_device_capability_family(120) == False  -> poprawki SM120 "
                   "MILCZĄCO SIĘ NIE AKTYWUJĄ")
        line(INFO, "to gorszy przypadek niż twarda odmowa: dostaniesz "
                   "'broken-as-shipped' zachowanie v0.24.0 bez komunikatu "
                   "(NaN-y w einsumie o_proj, 'invalid argument' w topk)")
        record(FAIL, "E. family check", "False — poprawki nieaktywne")


def stage_stack() -> None:
    head("F. Zależności stosu")
    for mod, label in (("deep_gemm", "DeepGEMM"),
                       ("flashinfer", "flashinfer")):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            line(PASS, f"{label} {ver}  ({getattr(m, '__file__', '?')})")
            record(PASS, f"F. {label}", str(ver))
        except Exception as e:  # noqa: BLE001
            line(WARN, f"{label} nie importuje się: {type(e).__name__}: {e}")
            record(WARN, f"F. {label}", type(e).__name__)

    # sparse-MLA flashinfera budowany jest z 12.0f (family) -> powinien
    # przejść na 121; nvfp4_ds_mla ext jest 120a (device) -> nie przejdzie
    jit = os.path.expanduser("~/.cache/flashinfer")
    if os.path.isdir(jit):
        found = []
        for dirpath, _d, files in os.walk(jit):
            for fn in files:
                if fn.endswith(".so") and "sparse_mla" in fn:
                    found.append(os.path.join(dirpath, fn))
        line(PASS if found else WARN,
             f"prekompilowany sparse-MLA w cache: {len(found)} .so")
    else:
        line(SKIP, f"brak {jit} — sparse-MLA pójdzie przez JIT przy pierwszym starcie")

    ext = os.getenv("VLLM_NVFP4_DS_MLA_EXT_DIR", "/opt/nvfp4-ds-mla")
    if os.path.isdir(ext):
        line(INFO, f"{ext} istnieje — zbudowany dla sm_120a (device-specific). "
                   "Na sm_121 NIE wystartuje, ale dotyczy tylko "
                   "--kv-cache-dtype nvfp4. Zostań przy fp8.")


def stage_budget(total_bytes: int, cu: Cuda) -> None:
    head("G. Budżet pamięci dla DeepSeek-V4-Flash-0731")
    if total_bytes <= 0:
        line(SKIP, "nie znam rozmiaru pamięci urządzenia")
        return

    free = ctypes.c_size_t()
    tot = ctypes.c_size_t()
    cu_free = None
    if cu.lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(tot)) == 0:
        cu_free = free.value / 2**30
        line(INFO, f"cuMemGetInfo: wolne {cu_free:.1f} GiB "
                   f"z {tot.value / 2**30:.1f} GiB")
    avail = cu_free if cu_free is not None else total_bytes / 2**30

    # Na pamieci ZUNIFIKOWANEJ cuMemGetInfo raportuje MemFree, ktory NIE liczy page cache'u —
    # a ten jest odzyskiwalny. Po duzym `docker pull`/buildzie potrafi trzymac 30 GiB i budzic
    # falszywy alarm "nie zmiesci sie". Wlasciwa liczba to MemAvailable z /proc/meminfo.
    meminfo = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                meminfo[k] = int(v.split()[0]) / 2**20   # kB -> GiB
    except OSError:
        pass
    mem_avail = meminfo.get("MemAvailable")
    if mem_avail is not None:
        cached = meminfo.get("Cached", 0) + meminfo.get("Buffers", 0)
        line(INFO, f"/proc/meminfo: MemAvailable {mem_avail:.1f} GiB "
                   f"(w tym ~{cached:.1f} GiB odzyskiwalnego page cache'u)")
        if cu_free is not None and mem_avail - cu_free > 5:
            line(INFO, f"  -> biore MemAvailable ({mem_avail:.0f} GiB), nie MemFree "
                       f"({cu_free:.0f} GiB): roznica to page cache, jadro go odda")
        avail = max(avail, mem_avail)

    # 43 warstwy x 256 ekspertów x 3 x 4096 x 2048 = 277e9 parametrów
    # routed experts; plany 2-bit to kody + skale UE8M0 blok-32 ~ 2.25 bit/elem
    routed_params = 43 * 256 * 3 * 4096 * 2048
    planes_gib = routed_params * (2.25 / 8) / 2**30
    dense_gib = 10.0        # attention fp8 + shared expert + embeddingi + lm_head
    dspark_gib = 3.0        # moduł DSpark + głowa MTP (szacunek, 0731 > preview)
    need = planes_gib + dense_gib + dspark_gib

    line(INFO, f"plany 2-bit (routed experts, {routed_params/1e9:.0f}B param): "
               f"~{planes_gib:.0f} GiB")
    line(INFO, f"dense/attention/embeddingi: ~{dense_gib:.0f} GiB   "
               f"DSpark+MTP: ~{dspark_gib:.0f} GiB")
    line(INFO, f"razem rezydentnie: ~{need:.0f} GiB   |   dostępne: "
               f"~{avail:.0f} GiB   |   zapas: {avail - need:+.0f} GiB")

    if avail - need > 12:
        line(PASS, "mieści się z zapasem na KV + pulę FP4 + grafy CUDA")
        record(PASS, "G. budżet", f"zapas {avail - need:.0f} GiB")
    elif avail - need > 0:
        line(WARN, "mieści się, ale ciasno — pula FP4 (VLLM_MOE_W2_DELTA_GB) "
                   "i okno kontekstu będą walczyć o resztę")
        record(WARN, "G. budżet", f"zapas {avail - need:.0f} GiB")
    else:
        line(FAIL, "nie mieści się rezydentnie — potrzebny base cache "
                   "(VLLM_MOE_W2_BASE_CACHE_GB), co na pamięci zunifikowanej "
                   "nie ma sensu wydajnościowego")
        record(FAIL, "G. budżet", f"brakuje {need - avail:.0f} GiB")

    line(INFO, "Uwaga do pamięci zunifikowanej: --gpu-memory-utilization liczy "
               "się od CAŁYCH 128 GB, z których żyje też system. Zacznij od "
               "0.80–0.85, nie od 0.95 jak w recipe dla kart dedykowanych.")
    line(INFO, "Ładowanie: STREAM-BUILD (VLLM_MOE_W2_STREAM_BUILD=1, domyślne) "
               "trzyma szczyt stagingu na ~1 warstwie (~6 GB) zamiast całego "
               "checkpointu — na 128 GB to warunek konieczny, sprawdź w logu.")


def stage_next_steps(gate1: bool) -> None:
    head("H. Co dalej")
    if not gate1:
        line(INFO, "GATE 1 padł — dalsze kroki nie mają sensu. Jedyna droga to "
                   "przebudowa cubinów pod sm_121 przez cubit "
                   "(github.com/kacper-daftcode/cubit), czyli inna skala pracy.")
        return
    for i, s in enumerate((
        "usuń strażnika w moe_w2_cubit.py::_ensure_ready "
        "(cap != (12,0)  ->  cap[0] != 12) — lista miejsc w etapie D",
        "odpal op-testy BEZ checkpointu: "
        "python3 tools/test_moe_w2_forward.py  (poprawność kerneli na 12.1)",
        "potem tools/test_moe_w2_planes.py i tools/test_store_backends.py",
        "dopiero teraz ściągaj checkpoint (~167 GB) i startuj z "
        "--kv-cache-dtype fp8, MTP k=2, małym oknem (8–16K) na pierwszy boot",
        "zmierz decode i porównaj z szacunkiem 25–45 tok/s; jeśli jesteś "
        "znacząco niżej, to nie kwantyzacja tylko scheduling SASS-ów na 48 SM",
        "dopiero na końcu próbuj --speculative-config method=dspark "
        "num_speculative_tokens=7 (backport jest w patchu)",
    ), 1):
        line(INFO, f"{i}. {s}")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Preflight vLLM-Moet na GX10 (GB10 / sm_121)")
    ap.add_argument("--cubins",
                    default=os.getenv("VLLM_MOE_W2_CUBIT_DIR", "/cubit-share"),
                    help="katalog z cubinami sm_120 (domyślnie /cubit-share)")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel size, którego zamierzasz użyć (1)")
    args = ap.parse_args()

    print(_c("1", "\npreflight_gx10 — vLLM-Moet na GB10/sm_121"))
    print(_c("90", "cel: rozstrzygnąć GATE 1 zanim ściągniesz 167 GB\n"))

    stage_env()

    try:
        cu = Cuda()
    except OSError as e:
        head("B. GPU")
        line(FAIL, f"nie mogę załadować libcuda.so.1: {e}")
        line(INFO, "uruchom w kontenerze z --gpus all")
        record(FAIL, "B. libcuda", str(e))
        summary(False)
        return 1

    _major, _minor, total = stage_device(cu)
    gate1 = stage_cubins(cu, args.cubins, args.tp)
    stage_guards()
    stage_family()
    stage_stack()
    stage_budget(total, cu)
    stage_next_steps(gate1)
    return summary(gate1)


def summary(gate1: bool) -> int:
    head("PODSUMOWANIE")
    width = max((len(s) for _, s, _ in RESULTS), default=20)
    for status, stage, note in RESULTS:
        print(f"  {_c(_COLOR[status], status):>4}  {stage:<{width}}  "
              f"{_c('90', note)}")
    fails = [r for r in RESULTS if r[0] == FAIL]
    warns = [r for r in RESULTS if r[0] == WARN]
    print()
    if not gate1:
        print(_c("31;1", "  GATE 1 NIE PRZESZEDŁ — cubiny sm_120 nie ładują się "
                         "na tym GPU. Stop."))
        return 1
    if fails:
        print(_c("31;1", f"  GATE 1 OK, ale {len(fails)} etapów padło — "
                         "przeczytaj etapy D/E, zanim pójdziesz dalej."))
        return 2
    if warns:
        print(_c("33", f"  GATE 1 OK, {len(warns)} ostrzeżeń — spodziewane "
                       "(strażniki trzeba zdjąć ręcznie). Idź do etapu H."))
        return 0
    print(_c("32;1", "  Wszystko czysto. Idź do etapu H."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
