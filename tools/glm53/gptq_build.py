#!/usr/bin/env python3
"""Build a GLM-5.3-Flash planes-cache variant whose 2-bit w13 codes come from GPTQ.

Same format, same scales, same kernels - only the choice of code per weight
changes. Round-to-nearest (RTN) picks, for every weight independently, the
nearest of s * {-4, -1, 1, 4}. GPTQ picks them column by column and pushes
each column's rounding error onto the columns not yet quantized, weighted by
the inverse Hessian of the layer's real inputs, so the errors cancel in the
OUTPUT instead of adding up. On this model every expert of a layer sees the
same input stream, so one Hessian per layer serves all 288 experts.

Measured on the reference box (42 layers, code calibration): w13 output error
on held-out calibration tokens -22.0% on average (median -21.0%, range
-13.8 ... -41.8%).

Output:
  <dst>/v2/<hash>/tp1-rank0/
      layer{L}.planes13.bin   NEW (GPTQ codes, pack_fragment_major)
      everything else         symlinks to <src> (sc13 = the SAME scales)
      meta.json               copy (the loader compares meta and sizes)
  <dst>/gptq_build_summary.json   per-layer RTN vs GPTQ output error

Scales are taken from the EXISTING sc13 (inverse of pack_scales), so the grid
is bit-for-bit the production one. Self-check per layer: production RTN
(fp8_block_to_codes_scales, f64) of one expert, packed by this script, must
give EXACTLY the bytes in <src>/planes13 - otherwise abort.

Hessian: from the MoE-input dumps of patches/patch_xdump.py (boot.sh XDUMP=1);
every fifth 2048-token block is held out for the RTN-vs-GPTQ evaluation.

Run inside the serving image with the server STOPPED (unified memory: the
build needs the GPU and ~20 GiB):
  docker run --rm --gpus all --entrypoint python3 \\
    -v $MODEL_DIR:/model:ro -v $PACKS_DIR:/packs -v $NVME_DIR/xdump:/x:ro \\
    -v $REPO/tools:/w:ro vllm-moet-glm53:dflash2 /w/glm53/gptq_build.py \\
    --src /packs/planes-cache-fp4 --dst /packs/planes-cache-gptq-code --xdump /x
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

# build_fp4_sections.py lives one directory up (tools/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_fp4_sections import Shards, layer_prefix, load_planes_module  # noqa: E402

PLANES = ("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
          "layers/quantization/utils/moe_w2_planes.py")
FIRST_DENSE = 3                 # first_k_dense_replace: layer key k = model layer k + 3
E, N13, K13 = 288, 4096, 4096
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0])


def rank_dir(root):
    ds = glob.glob(os.path.join(root, "v2", "*", "tp1-rank0"))
    if len(ds) != 1:
        raise SystemExit(f"expected exactly one v2/*/tp1-rank0 in {root}: {ds}")
    return ds[0]


def unpack_scales(packed, N, KS):
    """inverse of moe_w2_planes.pack_scales: [N*KS] -> [N, KS]"""
    return packed.view(N // 16, KS, 16).transpose(1, 2).reshape(N, KS)


def codes_of(u):
    """u = w/s -> index into LEVELS [-4,-1,1,4] (threshold 2.5)"""
    a = u.abs() < 2.5
    return torch.where(u < 0, torch.where(a, 1, 0), torch.where(a, 2, 3)).to(torch.uint8)


def gptq_codes(W, S, U, blocksize=128):
    """GPTQ on the fixed grid S*{-4,-1,1,4}; W, S [R, K] f32 (already permuted
    by act-order), U = upper Cholesky factor of the inverse Hessian (f32).
    Returns uint8 codes."""
    R, K = W.shape
    C = torch.empty(R, K, dtype=torch.uint8, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.empty_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            s = S[:, i1 + i]
            c = codes_of(w / s)
            C[:, i1 + i] = c
            q = LEVELS.to(W.device)[c.long()] * s
            err = (w - q) / U1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * U1[i, i:].unsqueeze(0)
            E1[:, i] = err
        W[:, i2:] -= E1 @ U[i1:i2, i2:]
    return C


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="/model")
    ap.add_argument("--src", required=True, help="RTN planes cache (with FP4 sections)")
    ap.add_argument("--dst", required=True, help="new planes cache directory")
    ap.add_argument("--xdump", default="/x", help="directory with L{key}_*.pt dumps")
    ap.add_argument("--keys", default="all", help="layer keys (e.g. 3,17) or all")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=32, help="experts per GPTQ batch")
    ap.add_argument("--eval-experts", type=int, default=4)
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--abs-links", action="store_true",
                    help="absolute symlinks to src (use when dst is on another disk)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    planes = load_planes_module(PLANES)
    src = rank_dir(a.src)
    dst = os.path.join(a.dst, os.path.relpath(src, a.src))
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(src, "meta.json"), os.path.join(dst, "meta.json"))
    shards = Shards(a.model_dir)

    have = sorted({int(os.path.basename(p)[1:].split("_")[0])
                   for p in glob.glob(os.path.join(a.xdump, "L*_*.pt"))})
    keys = have if a.keys == "all" else [int(k) for k in a.keys.split(",")]
    print(f"[build] src {src}\n[build] dst {dst}\n[build] keys with a Hessian: {keys}", flush=True)
    sz13 = N13 * K13 // 4
    summary = {}
    t_all = time.time()
    for key in keys:
        t0 = time.time()
        L = key + FIRST_DENSE
        files = sorted(glob.glob(os.path.join(a.xdump, f"L{key}_*.pt")))
        x = torch.cat([torch.load(f, map_location="cpu")["x"] for f in files])
        ids = torch.cat([torch.load(f, map_location="cpu")["topk_ids"] for f in files])
        T = x.shape[0]
        test = (torch.arange(T) // 2048) % a.test_every == a.test_every - 1
        xtr = x[~test].to(dev, torch.float32)
        H = (xtr.T @ xtr).double() / xtr.shape[0]
        del xtr
        H += a.damp * torch.mean(torch.diag(H)) * torch.eye(K13, dtype=H.dtype, device=dev)
        perm = torch.argsort(torch.diag(H), descending=True)
        Hp = H[perm][:, perm]
        U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hp)),
                                  upper=True).float()
        inv = torch.argsort(perm)

        sc_all = torch.from_numpy(np.fromfile(os.path.join(src, f"layer{L}.sc13.bin"),
                                              dtype=np.uint8)).view(E, -1)
        pl_src = np.fromfile(os.path.join(src, f"layer{L}.planes13.bin"), dtype=np.uint8)
        assert pl_src.size == E * sz13, (pl_src.size, E * sz13)
        out = np.empty(E * sz13, dtype=np.uint8)
        pref = layer_prefix(shards.weight_map, L)

        def load_w13(e):
            ws = []
            for nm in ("gate_proj", "up_proj"):
                w = shards.get(f"{pref}{e}.{nm}.weight")
                s = shards.get(f"{pref}{e}.{nm}.weight_scale_inv")
                ws.append((w, s))
            w8 = torch.cat((ws[0][0], ws[1][0]), 0).to(dev)
            s8 = torch.cat((ws[0][1], ws[1][1]), 0).to(dev)
            sf = s8.float().repeat_interleave(128, 0)[:N13].repeat_interleave(128, 1)[:, :K13]
            return w8, s8, w8.float() * sf

        # self-check: production RTN of expert 0 -> our packer -> the bytes in src
        w8, s8, _ = load_w13(0)
        c_prod, sb_prod, _ = planes.fp8_block_to_codes_scales(w8, s8, block_shape=(128, 128))
        if not torch.equal(planes.pack_scales(sb_prod).cpu(), sc_all[0]):
            raise SystemExit(f"L{L}: RTN scales != sc13 in the cache - aborting")
        if not np.array_equal(planes.pack_fragment_major(c_prod).cpu().numpy(), pl_src[:sz13]):
            raise SystemExit(f"L{L}: packing != planes13 in the cache - aborting")

        counts = torch.bincount(ids.flatten().long().clamp(min=0), minlength=E)
        ev_set = set(torch.argsort(counts, descending=True)[:a.eval_experts].tolist())
        ev = {}
        mism = None
        for b0 in range(0, E, a.batch):
            bl = list(range(b0, min(E, b0 + a.batch)))
            Ws, Ss = [], []
            for e in bl:
                _, _, W = load_w13(e)
                sc = unpack_scales(sc_all[e], N13, K13 // 32).to(dev)
                Ws.append(W)
                Ss.append(torch.exp2(sc.float() - 127.0).repeat_interleave(32, 1))
            W = torch.cat(Ws)[:, perm].contiguous()
            S = torch.cat(Ss)[:, perm].contiguous()
            W_orig = torch.cat(Ws)
            del Ws, Ss
            if mism is None:   # our RTN path vs production (expert 0)
                c_rtn0 = codes_of(W_orig[:N13] / S[:N13][:, inv])
                mism = float((c_rtn0 != c_prod).float().mean())
            C = gptq_codes(W, S, U)[:, inv]
            for j, e in enumerate(bl):
                ce = C[j * N13:(j + 1) * N13]
                out[e * sz13:(e + 1) * sz13] = planes.pack_fragment_major(ce).cpu().numpy()
                if e in ev_set:
                    sel = (ids == e).any(1) & test
                    Xt = x[sel].to(dev, torch.float32)
                    We = W_orig[j * N13:(j + 1) * N13]
                    Se = S[j * N13:(j + 1) * N13][:, inv]
                    Qg = LEVELS.to(dev)[ce.long()] * Se
                    Qr = LEVELS.to(dev)[codes_of(We / Se).long()] * Se
                    Y = Xt @ We.T
                    ev[e] = (float(torch.linalg.norm(Xt @ (Qr - We).T) / torch.linalg.norm(Y)),
                             float(torch.linalg.norm(Xt @ (Qg - We).T) / torch.linalg.norm(Y)),
                             int(sel.sum()))
            del W, S, W_orig, C
        tmp = os.path.join(dst, f"layer{L}.planes13.bin.tmp")
        out.tofile(tmp)
        os.replace(tmp, os.path.join(dst, f"layer{L}.planes13.bin"))
        r = sum(v[0] for v in ev.values()) / max(1, len(ev))
        g = sum(v[1] for v in ev.values()) / max(1, len(ev))
        changed = float((out != pl_src).mean())
        summary[L] = {"out13_rtn": r, "out13_gptq": g, "bytes_changed": changed,
                      "rtn_code_mismatch": mism, "tokens": T}
        print(f"[build] L{L} (key {key}): {T} tok, out13 RTN {r:.4f} -> GPTQ {g:.4f} "
              f"({100 * (g / r - 1):+.1f}%), bytes changed {100 * changed:.1f}%, "
              f"RTN mismatch {mism:.2e}, {time.time() - t0:.0f} s", flush=True)
        del x, ids, out, pl_src

    # symlinks for everything not built here (incl. planes13 of layers without a Hessian)
    n_link = 0
    for p in sorted(glob.glob(os.path.join(src, "layer*.bin"))):
        q = os.path.join(dst, os.path.basename(p))
        if os.path.lexists(q):
            continue
        # --abs-links: dst on another disk than src - a relative link would
        # resolve against the REAL dst directory and miss; an absolute
        # container path (/packs/...) works in every container with the same
        # mounts (it dangles on the host, but the host never reads it)
        os.symlink(p if a.abs_links else os.path.relpath(p, dst), q)
        n_link += 1
    json.dump(summary, open(os.path.join(a.dst, "gptq_build_summary.json"), "w"), indent=1)
    print(f"[build] DONE: {len(summary)} GPTQ layers, {n_link} symlinks, "
          f"{time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
