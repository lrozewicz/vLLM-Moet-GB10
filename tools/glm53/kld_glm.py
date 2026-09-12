#!/usr/bin/env python3
"""KLD harness for GLM-5.3-Flash on vLLM: teacher forcing through prompt logprobs.

Why: a dozen temperature-1.0 generations cannot tell "1 failure in 12" from
"2 in 9". This measures quality deterministically on a fixed corpus instead:
every position of every chunk, the full next-token distribution.

Metric: at every corpus position, the next-token distribution of config TEST
is compared with config REF. vLLM returns only the top-K, so KL is computed on
the reference's top-K plus one "rest of the vocabulary" bucket. A token that is
missing from the test's top-K gets the test's K-th logprob - an upper bound on
q, so KL is slightly UNDERestimated, equally for every comparison. Next to KL:
top-1 agreement and the perplexity of the actual tokens (PPL says in which
direction a change moved the model; KL only says how far).

Modes:
  prep         tokenize the corpus -> chunks.json, inside the vLLM image
               (tokenizer from /model), no server needed:
                 kld_glm.py prep --model-dir /model --out chunks.json \\
                   --corpus code /c/code.txt 16 --corpus en /c/en.txt 10
  prep-server  the same through /tokenize of a RUNNING server (no second copy
               of the tokenizer in memory while the model is served);
               --spread spaces the chunks evenly over the file
  dump         chunk by chunk as token ids with prompt_logprobs=K -> .npz
               (the server needs --max-logprobs >= K; boot.sh MAX_LOGPROBS=64):
                 VLLM_API_KEY=... kld_glm.py dump --chunks chunks.json --out R.npz --tag R
  compare      REF.npz TEST.npz [TEST2.npz ...] -> per-domain table (+ --json)

Requests with prompt_logprobs do not read the prefix cache, so two dumps in
one boot are independent - dump the same config twice to get the noise floor.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
KEY = os.environ.get("VLLM_API_KEY", "")


# --------------------------------------------------------------------- prep
def cmd_prep(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    chunks = []
    for dom, path, n in a.corpus:
        text = open(path, encoding="utf-8").read()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n = int(n)
        have = len(ids) // a.chunk_len
        if have < n:
            print(f"[prep] {dom}: only {have} full chunks of {n}", file=sys.stderr)
            n = have
        for i in range(n):
            chunks.append({"domain": dom, "idx": i,
                           "ids": ids[i * a.chunk_len:(i + 1) * a.chunk_len]})
        print(f"[prep] {dom}: {len(text)} chars, {len(ids)} tokens "
              f"({len(text) / max(1, len(ids)):.2f} chars/token), {n} x {a.chunk_len}")
    json.dump({"chunk_len": a.chunk_len, "model_dir": a.model_dir,
               "chunks": chunks}, open(a.out, "w"))
    print(f"[prep] wrote {len(chunks)} chunks -> {a.out}")


# --------------------------------------------------------------------- dump
def _req(path, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if KEY:
        req.add_header("Authorization", "Bearer " + KEY)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _metrics_line(prefix):
    try:
        txt = _req("/metrics", timeout=30).decode()
    except Exception as e:  # noqa: BLE001
        return f"(metrics: {e})"
    return [ln for ln in txt.splitlines() if ln.startswith(prefix)]


def cmd_dump(a):
    import numpy as np
    spec = json.load(open(a.chunks))
    chunks = spec["chunks"]
    if a.limit:
        chunks = chunks[:a.limit]
    model = json.loads(_req("/v1/models", timeout=30))["data"][0]["id"]
    K = a.k
    domains = sorted({c["domain"] for c in chunks})
    all_ids, pos, cidx, dom = [], [], [], []
    topk_ids, topk_lp, act = [], [], []
    t_start = time.time()
    for ci, c in enumerate(chunks):
        ids = c["ids"]
        T = len(ids)
        t0 = time.time()
        r = json.loads(_req("/v1/completions", {
            "model": model, "prompt": ids, "max_tokens": 1,
            "temperature": 0.0, "prompt_logprobs": K}))
        dt = time.time() - t0
        plp = r["choices"][0]["prompt_logprobs"]
        assert len(plp) == T, (len(plp), T)
        ti = np.full((T, K), -1, np.int32)
        tl = np.full((T, K), -1e9, np.float32)
        ac = np.full(T, np.nan, np.float32)
        for t in range(1, T):
            d = plp[t]
            if not d:
                continue
            rows = []
            for k, v in d.items():
                tid = int(k)
                if tid == ids[t]:
                    ac[t] = v["logprob"]
                rk = v.get("rank")
                if rk is not None and rk <= K:
                    rows.append((rk, tid, v["logprob"]))
            rows.sort()
            for j, (_, tid, lp) in enumerate(rows[:K]):
                ti[t, j] = tid
                tl[t, j] = lp
        all_ids.append(np.asarray(ids, np.int32))
        pos.append(np.arange(T, dtype=np.int32))
        cidx.append(np.full(T, ci, np.int32))
        dom.append(np.full(T, domains.index(c["domain"]), np.int8))
        topk_ids.append(ti)
        topk_lp.append(tl)
        act.append(ac)
        nll = -np.nanmean(ac[1:])
        print(f"[dump {a.tag}] {ci + 1}/{len(chunks)} {c['domain']}#{c['idx']} "
              f"{T} tok {dt:.1f} s ({T / dt:.0f} tok/s) ppl {math.exp(nll):.3f}",
              flush=True)
    meta = {"tag": a.tag, "k": K, "chunk_len": spec["chunk_len"],
            "domains": domains, "model": model, "base": BASE,
            "seconds": round(time.time() - t_start, 1),
            "env": {k: os.environ.get(k, "") for k in (a.env or [])},
            "cache_config_info": _metrics_line("vllm:cache_config_info")}
    np.savez_compressed(
        a.out, ids=np.concatenate(all_ids), pos=np.concatenate(pos),
        chunk=np.concatenate(cidx), dom=np.concatenate(dom),
        topk_ids=np.concatenate(topk_ids), topk_lp=np.concatenate(topk_lp),
        act=np.concatenate(act), meta=json.dumps(meta))
    print(f"[dump {a.tag}] wrote {a.out} in {meta['seconds']} s")


# ------------------------------------------------------------------ compare
def _kl_block(pid, plp, qid, qlp):
    import numpy as np
    match = pid[:, :, None] == qid[:, None, :]            # [n, K, K]
    has = match.any(2)
    idx = match.argmax(2)
    lq = np.take_along_axis(qlp, idx, 1)
    lq = np.where(has, lq, qlp[:, -1:])                   # outside the test top-K -> K-th
    p = np.exp(plp.astype(np.float64))
    kl_top = (p * (plp - lq)).sum(1)
    p_rest = np.clip(1.0 - p.sum(1), 0.0, 1.0)
    q_rest = np.clip(1.0 - np.exp(lq.astype(np.float64)).sum(1), 1e-12, 1.0)
    tail = np.where(p_rest > 1e-9,
                    p_rest * np.log(np.clip(p_rest, 1e-12, 1.0) / q_rest), 0.0)
    return kl_top + tail, pid[:, 0] == qid[:, 0]


def compare(ref_path, test_path, skip):
    import numpy as np
    R = np.load(ref_path)
    Q = np.load(test_path)
    if not np.array_equal(R["ids"], Q["ids"]):
        raise SystemExit(f"{test_path}: different tokens than {ref_path}")
    mr = json.loads(str(R["meta"]))
    mq = json.loads(str(Q["meta"]))
    valid = ((R["pos"] >= skip) & (R["topk_ids"][:, 0] >= 0)
             & (Q["topk_ids"][:, 0] >= 0)
             & ~np.isnan(R["act"]) & ~np.isnan(Q["act"]))
    sel = np.nonzero(valid)[0]
    kl = np.empty(len(sel))
    top1 = np.empty(len(sel), bool)
    B = 4096
    for s in range(0, len(sel), B):
        ix = sel[s:s + B]
        kl[s:s + B], top1[s:s + B] = _kl_block(
            R["topk_ids"][ix], R["topk_lp"][ix],
            Q["topk_ids"][ix], Q["topk_lp"][ix])
    kl = np.maximum(kl, 0.0)
    dom = R["dom"][sel]
    chunk = R["chunk"][sel]
    out = {"ref": mr["tag"], "test": mq["tag"], "skip": skip, "domains": {}}
    for di, dname in enumerate(mr["domains"]):
        m = dom == di
        if not m.any():
            continue
        chunks = sorted(set(chunk[m].tolist()))
        per_chunk = [float(kl[m & (chunk == c)].mean()) for c in chunks]
        out["domains"][dname] = {
            "n": int(m.sum()),
            "kld": float(kl[m].mean()),
            # standard error from the spread between chunks (positions within
            # a chunk are correlated, so they are not independent samples)
            "kld_se": float(np.std(per_chunk, ddof=1) / math.sqrt(len(per_chunk)))
            if len(per_chunk) > 1 else float("nan"),
            "kld_p99": float(np.percentile(kl[m], 99)),
            "top1": float(top1[m].mean()),
            "ppl_ref": float(math.exp(-R["act"][sel][m].mean())),
            "ppl_test": float(math.exp(-Q["act"][sel][m].mean())),
            "per_chunk": [round(x, 5) for x in per_chunk],
        }
    return out


def cmd_compare(a):
    rows = [compare(a.ref, t, a.skip) for t in a.tests]
    print(f"REF = {a.ref}  (positions >= {a.skip})")
    print(f"{'test':<22}{'domain':<8}{'KLD':>9}{'+-SE':>8}{'p99':>8}"
          f"{'top-1':>8}{'PPL ref':>9}{'PPL test':>9}")
    for r in rows:
        for d, v in r["domains"].items():
            print(f"{r['test']:<22}{d:<8}{v['kld']:>9.5f}{v['kld_se']:>8.5f}"
                  f"{v['kld_p99']:>8.3f}{100 * v['top1']:>7.2f}%"
                  f"{v['ppl_ref']:>9.4f}{v['ppl_test']:>9.4f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"-> {a.json}")


def cmd_prep_server(a):
    model = json.loads(_req("/v1/models", timeout=30))["data"][0]["id"]
    chunks = []
    for dom, path, n in a.corpus:
        text = open(path, encoding="utf-8").read()
        ids, pos = [], 0
        while pos < len(text):
            end = min(len(text), pos + a.piece)
            if end < len(text):
                nl = text.rfind("\n", pos, end)
                end = nl + 1 if nl > pos else end
            r = json.loads(_req("/tokenize", {"model": model, "prompt": text[pos:end],
                                              "add_special_tokens": False}))
            ids.extend(r["tokens"])
            pos = end
        n = min(int(n), len(ids) // a.chunk_len)
        if a.spread and n > 1:
            step = (len(ids) - a.chunk_len) // (n - 1)
            starts = [i * step for i in range(n)]
        else:
            starts = [i * a.chunk_len for i in range(n)]
        for i, s in enumerate(starts):
            chunks.append({"domain": dom, "idx": i, "ids": ids[s:s + a.chunk_len]})
        print(f"[prep-server] {dom}: {len(text)} chars, {len(ids)} tokens, "
              f"{n} x {a.chunk_len}{' (spread)' if a.spread else ''}")
    json.dump({"chunk_len": a.chunk_len, "model_dir": f"server:{model}",
               "chunks": chunks}, open(a.out, "w"))
    print(f"[prep-server] wrote {len(chunks)} chunks -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep-server")
    p.add_argument("--corpus", nargs=3, action="append", required=True,
                   metavar=("DOMAIN", "FILE", "N_CHUNKS"))
    p.add_argument("--chunk-len", type=int, default=4096)
    p.add_argument("--piece", type=int, default=60000,
                   help="characters per /tokenize call")
    p.add_argument("--spread", action="store_true")
    p.add_argument("--out", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--model-dir", default="/model")
    p.add_argument("--corpus", nargs=3, action="append", required=True,
                   metavar=("DOMAIN", "FILE", "N_CHUNKS"))
    p.add_argument("--chunk-len", type=int, default=4096)
    p.add_argument("--out", required=True)
    p = sp.add_parser("dump")
    p.add_argument("--chunks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--env", nargs="*", help="environment variables to record in meta")
    p = sp.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("tests", nargs="+")
    p.add_argument("--skip", type=int, default=256,
                   help="skip the first positions of each chunk (too little context)")
    p.add_argument("--json")
    a = ap.parse_args()
    {"prep": cmd_prep, "prep-server": cmd_prep_server, "dump": cmd_dump,
     "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
