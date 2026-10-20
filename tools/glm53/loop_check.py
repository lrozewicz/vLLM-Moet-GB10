#!/usr/bin/env python3
"""Looped (grid-stride, MOEW2_LOOP=1) moe_w2_mm / moe_w4_mm vs the production
cubins: bit-exactness and decode-launch time.

Production launch: grid.y = PAIRS table entries, `real` of them live at
scattered positions (m_rows = 4), the rest m_rows = 0.
Looped launch: the same live entries COMPACTED to the front (zero tail),
grid.y = G, n_pairs = PAIRS, y_stride = G.
Also checks the looped cubin on the UNcompacted table with G = PAIRS
(the fallback the server uses when it cannot compact).

Run with the server STOPPED, in a separate container:
  OLD_DIR=/cubit-share NEW_DIR=/w/loop python3 loop_check.py
"""
import ctypes
import os

import numpy as np
import torch

OLD = os.environ.get("OLD_DIR", "/cubit-share")
NEW = os.environ.get("NEW_DIR", "/w/loop")
PAIRS = int(os.environ.get("PAIRS", "64"))
E_ALL = 288
cu = ctypes.CDLL("libcuda.so.1")
torch.cuda.init()
torch.zeros(1, device="cuda")
cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
    ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what}: CUresult {r}")


def load(path, kernel):
    mod = ctypes.c_void_p()
    ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()), "load " + path)
    fn = ctypes.c_void_p()
    ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, kernel.encode()), "getfn")
    regs = ctypes.c_int()
    ck(cu.cuFuncGetAttribute(ctypes.byref(regs), 4, fn), "attr")  # NUM_REGS
    return fn, regs.value


def nwarp(k):
    nb = k // 128
    cap = 8 if k >= 1024 else 4
    for n in range(min(cap, nb), 0, -1):
        if nb % n == 0:
            return n
    return 1


class Case:
    def __init__(self, tier, N, K, real, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        pb = N * K // (4 if tier == "w2" else 2)
        sb = N * K // 32
        self.N, self.K = N, K
        self.planes = torch.randint(0, 255, (E_ALL, pb), dtype=torch.uint8,
                                    device="cuda", generator=g)
        self.scales = torch.randint(118, 130, (E_ALL, sb), dtype=torch.uint8,
                                    device="cuda", generator=g)
        rows = PAIRS * 4
        self.a = (torch.randn(rows, K, device="cuda", generator=g) * 0.5
                  ).to(torch.float8_e4m3fn)
        self.a_s = torch.rand(rows, K // 32, device="cuda", generator=g) + 0.5
        rng = np.random.default_rng(seed)
        self.live = sorted(rng.choice(PAIRS, size=real, replace=False))
        self.ex = rng.choice(E_ALL, size=PAIRS, replace=False)
        self.m = {p: int(rng.integers(1, 5)) for p in self.live}

    def entry(self, p, c, live=True):
        e = int(self.ex[p])
        return [self.a[p * 4].data_ptr(), self.a_s[p * 4].data_ptr(),
                self.planes[e].data_ptr(), self.scales[e].data_ptr(),
                c[p * 4].data_ptr(), self.m[p] if live else 0]

    def desc(self, c, compact):
        d = np.zeros((PAIRS, 6), dtype=np.uint64)
        if compact:
            for i, p in enumerate(self.live):
                d[i] = self.entry(p, c)
        else:
            for p in range(PAIRS):
                d[p] = self.entry(p, c, p in self.m)
        return torch.from_numpy(d.view(np.int64)).cuda()

    def c(self):
        return torch.full((PAIRS * 4, self.N), -7.0, dtype=torch.bfloat16,
                          device="cuda")


def launch(fn, case, desc, grid_y, n_pairs, stride):
    K, N = case.K, case.N
    args = [ctypes.c_uint64(desc.data_ptr()), ctypes.c_uint32(K),
            ctypes.c_uint32(K // 64), ctypes.c_uint32(N * 2),
            ctypes.c_uint32(K // 128), ctypes.c_uint32(n_pairs),
            ctypes.c_uint32(stride)]
    argv = (ctypes.c_void_p * 7)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p)
                                   for x in args])
    ck(cu.cuLaunchKernel(fn, N // 16, grid_y, 1, nwarp(K) * 32, 1, 1, 0, None,
                         argv, None), "launch")


def timeit(f, iters=300):
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def main():
    shapes = [("w2", 4096, 4096), ("w2", 4096, 2048),
              ("w4", 4096, 4096), ("w4", 4096, 2048)]
    bad = 0
    for tier, N, K in shapes:
        name = f"moe_{tier}_mm_k{K}_a32.cubin"
        fo, ro = load(f"{OLD}/{name}", f"moe_{tier}_mm")
        fn, rn = load(f"{NEW}/{name}", f"moe_{tier}_mm")
        print(f"== {tier} N={N} K={K}: regs old {ro} new {rn}", flush=True)
        for real in (1, 4, 12, 24, 40, 64):
            case = Case(tier, N, K, real, seed=real * 7 + K)
            c_old, c_new, c_fb = case.c(), case.c(), case.c()
            launch(fo, case, case.desc(c_old, False), PAIRS, PAIRS, PAIRS)
            launch(fn, case, case.desc(c_fb, False), PAIRS, PAIRS, PAIRS)
            res = []
            for G in (4, 8, 16):
                c_new.fill_(-7.0)
                launch(fn, case, case.desc(c_new, True), G, PAIRS, G)
                torch.cuda.synchronize()
                # compacted rows land in the same c rows (c ptr per pair)
                res.append(torch.equal(c_old.view(torch.int16),
                                       c_new.view(torch.int16)))
            torch.cuda.synchronize()
            fb = torch.equal(c_old.view(torch.int16), c_fb.view(torch.int16))
            nz = int((c_old != -7.0).sum())
            ok = all(res) and fb and nz > 0
            bad += not ok
            d_old = case.desc(c_old, False)
            d_new = case.desc(c_new, True)
            t_old = timeit(lambda: launch(fo, case, d_old, PAIRS, PAIRS, PAIRS))
            t_fb = timeit(lambda: launch(fn, case, d_old, PAIRS, PAIRS, PAIRS))
            ts = {G: timeit(lambda: launch(fn, case, d_new, G, PAIRS, G))
                  for G in (4, 8, 12, 16, 24)}
            best = min(ts, key=ts.get)
            print(f"  real={real:2d} bitexact={'OK' if ok else 'FAIL'} "
                  f"(G4/8/16 {res}, fallback {fb}, written {nz}) | "
                  f"old {t_old:6.1f} us, loop-uncompacted {t_fb:6.1f} | "
                  + " ".join(f"G{G}:{t:6.1f}" for G, t in ts.items())
                  + f" | best G{best} saves {t_old - ts[best]:5.1f} us",
                  flush=True)
    print("ALL BITEXACT" if not bad else f"{bad} FAILURES")


if __name__ == "__main__":
    main()
