#!/usr/bin/env python3
"""Measure speculative decode throughput and per-position acceptance on a live
vLLM server - works for any method (MTP, DFlash2); the name is historical.

Reports, per request: TTFT, pure decode tok/s, and the acceptance profile taken
from the /metrics delta around that single request (so positions are exact, not
averaged over the server's lifetime). Every "code" number in
docs/models/glm-5.3-flash.md is the mean of the three code prompts below at
temperature 0, reasoning_effort low, 600 tokens, `--repeats 2 --warmup`.

Compare configurations on the SAME prompt set and warm server only: acceptance
depends on content far more than on configuration, and boot-to-boot variance
is ~8% on this box.

Usage:  VLLM_API_KEY=... ./bench_mtp.py --tag R --repeats 2 --warmup
"""
import argparse
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("VLLM_MODEL", "glm-5.3-flash")

# Fixed, deterministic prompt set. Two shapes, because MTP acceptance is
# strongly content-dependent: code is highly predictable (long runs of
# boilerplate the drafter nails), prose much less so.
PROMPTS = [
    ("code-bisect",
     "Write a Python function `bisect_left(a, x, lo=0, hi=None)` that mirrors the "
     "standard library's behaviour, then a short unit test with three cases. "
     "Code only, no commentary."),
    ("code-lru",
     "Implement an LRU cache class in Python with get/put in O(1) using a dict and "
     "a doubly linked list. Include type hints and a docstring for each method."),
    ("code-sql",
     "Write a PostgreSQL query that, for each customer, returns their id, name, "
     "number of orders in 2025 and total spend, including customers with zero "
     "orders, sorted by spend descending. Then explain the join choice in two "
     "sentences."),
    ("prose-explain",
     "Explain in about 250 words why speculative decoding speeds up LLM inference "
     "on a memory-bandwidth-bound machine, and what determines the point at which "
     "adding more draft tokens stops helping."),
]

SPEC_KEYS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def _get(path):
    req = urllib.request.Request(BASE + path)
    if KEY:
        req.add_header("Authorization", "Bearer " + KEY)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def metrics():
    """Scrape the speculative-decode counters into a flat dict."""
    out = {"pos": {}}
    for line in _get("/metrics").splitlines():
        if line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        name, _, value = line.partition(" ")
        head, _, labels = name.partition("{")
        try:
            v = float(value)
        except ValueError:
            continue
        if head in SPEC_KEYS:
            out[head] = v
        elif head == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            for part in labels.rstrip("}").split(","):
                k, _, val = part.partition("=")
                if k == "position":
                    out["pos"][int(val.strip('"'))] = v
    return out


def run_one(prompt, max_tokens, effort):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if KEY:
        req.add_header("Authorization", "Bearer " + KEY)

    t0 = time.perf_counter()
    t_first = None
    t_last = t0
    usage = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                # "reasoning" is this vLLM's field (NOT "reasoning_content" -
                # verified on the stream); the thinking phase is decode too, so
                # it must be inside the timing window, or its tokens count in
                # the numerator but its time not in the denominator (+2-7%).
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks += 1
    if t_first is None:
        raise RuntimeError("no content streamed back")
    n_out = usage["completion_tokens"] if usage else chunks
    # Decode rate excludes prefill: first token lands at t_first, the remaining
    # n_out-1 tokens are produced between t_first and t_last.
    decode_s = t_last - t_first
    return {
        "ttft_s": t_first - t0,
        "decode_s": decode_s,
        "out_tokens": n_out,
        "prompt_tokens": usage["prompt_tokens"] if usage else None,
        "decode_tps": (n_out - 1) / decode_s if decode_s > 0 else 0.0,
        "wall_tps": n_out / (t_last - t0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--out", default=None)
    ap.add_argument("--warmup", action="store_true",
                    help="throw away one short request first")
    args = ap.parse_args()

    if args.warmup:
        print("warmup...", flush=True)
        run_one("Say OK.", 32, args.effort)

    rows = []
    print(f"{'prompt':<16} {'rep':>3} {'out':>5} {'ttft':>6} {'dec tok/s':>10} "
          f"{'AL':>5} {'per-position acceptance'}", flush=True)
    for name, prompt in PROMPTS:
        for rep in range(args.repeats):
            m0 = metrics()
            r = run_one(prompt, args.max_tokens, args.effort)
            m1 = metrics()

            drafts = m1.get(SPEC_KEYS[0], 0) - m0.get(SPEC_KEYS[0], 0)
            dtok = m1.get(SPEC_KEYS[1], 0) - m0.get(SPEC_KEYS[1], 0)
            acc = m1.get(SPEC_KEYS[2], 0) - m0.get(SPEC_KEYS[2], 0)
            per_pos = {}
            for p in sorted(set(m0["pos"]) | set(m1["pos"])):
                per_pos[p] = m1["pos"].get(p, 0) - m0["pos"].get(p, 0)
            # Mean acceptance length: every draft step also emits one verified
            # ("bonus") token, so the length is 1 + accepted/drafts.
            al = 1 + acc / drafts if drafts else float("nan")
            k = int(round(dtok / drafts)) if drafts else 0
            pos_s = " ".join(
                f"p{p}={per_pos[p] / drafts * 100:.0f}%" for p in sorted(per_pos)
            ) if drafts else "-"

            r.update(prompt=name, rep=rep, tag=args.tag, drafts=drafts,
                     draft_tokens=dtok, accepted=acc, accept_len=al, k=k,
                     per_pos={str(p): v for p, v in per_pos.items()})
            rows.append(r)
            print(f"{name:<16} {rep:>3} {r['out_tokens']:>5} "
                  f"{r['ttft_s']:>6.2f} {r['decode_tps']:>10.2f} "
                  f"{al:>5.2f} {pos_s}", flush=True)

    code = [r for r in rows if r["prompt"].startswith("code")]
    prose = [r for r in rows if r["prompt"].startswith("prose")]

    def avg(rs, key):
        return sum(x[key] for x in rs) / len(rs) if rs else float("nan")

    tot_drafts = sum(r["drafts"] for r in rows)
    tot_acc = sum(r["accepted"] for r in rows)
    print()
    print(f"[{args.tag}] k={rows[0]['k']}  "
          f"code {avg(code, 'decode_tps'):.2f} tok/s (AL {avg(code, 'accept_len'):.2f})  "
          f"prose {avg(prose, 'decode_tps'):.2f} tok/s (AL {avg(prose, 'accept_len'):.2f})  "
          f"overall AL {1 + tot_acc / tot_drafts:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tag": args.tag, "rows": rows}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
