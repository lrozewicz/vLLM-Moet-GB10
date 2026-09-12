#!/usr/bin/env python3
"""HumanEval: generate candidates on a GLM-5.3-Flash endpoint.

Modeled on the harness in darkdatter/gb10-repo (MIT), adapted to this server:
API key, `reasoning_effort` instead of `enable_thinking` (GLM-5.3-Flash
ALWAYS thinks; only the depth is chosen), data from JSONL.

The dataset is not vendored. Get it from OpenAI's repository (MIT):
    curl -L https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz \\
      | gunzip > humaneval.jsonl

    VLLM_API_KEY=... python3 generate.py --effort low  --out gen-low.json
    VLLM_API_KEY=... python3 generate.py --effort high --out gen-high.json

Three pitfalls from their README that apply here just the same:
  1. generous max_tokens - with thinking, >4k tokens is normal; a truncation
     reads like a quality regression and is not one. finish_reason is recorded.
  2. the fence regex must tolerate a MISSING closing ``` (truncated outputs).
  3. the task prompt is prepended to the code at execution time, because
     HumanEval/38 and /50 define a helper ABOVE the target function.
"""
import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("VLLM_MODEL", "glm-5.3-flash")

INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including its signature and any needed imports) inside a single ```python "
    "code fence. No explanation.\n\n"
)


def extract(text):
    closed = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if closed:
        return max(closed, key=len)
    unclosed = re.search(r"```(?:python)?\s*\n(.*)$", text, re.S)
    if unclosed:
        return unclosed.group(1)
    return text


def chat(prompt, max_tokens, effort, timeout=3600):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    ch = d["choices"][0]
    return {
        "e2e": time.time() - t0,
        "completion_tokens": d["usage"]["completion_tokens"],
        "finish_reason": ch.get("finish_reason"),
        "content": ch["message"].get("content") or "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "humaneval.jsonl"))
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default 4096 for effort=low, 16384 otherwise")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not os.path.exists(args.data):
        sys.exit(f"{args.data} not found - see the download line at the top of this file")
    max_tokens = args.max_tokens or (4096 if args.effort == "low" else 16384)

    rows = [json.loads(line) for line in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    results = [None] * len(rows)
    work = queue.Queue()
    for i in range(len(rows)):
        work.put(i)
    lock = threading.Lock()
    done, tokens, truncated = [0], [0], [0]

    def worker():
        while True:
            try:
                i = work.get_nowait()
            except queue.Empty:
                return
            row = rows[i]
            try:
                r = chat(INSTRUCTION + row["prompt"], max_tokens, args.effort)
                results[i] = {
                    "task_id": row["task_id"], "code": extract(r["content"]),
                    "prompt": row["prompt"], "test": row["test"],
                    "entry_point": row["entry_point"],
                    "finish_reason": r["finish_reason"],
                    "completion_tokens": r["completion_tokens"], "e2e": r["e2e"],
                }
                with lock:
                    tokens[0] += r["completion_tokens"]
                    if r["finish_reason"] == "length":
                        truncated[0] += 1
            except Exception as exc:  # noqa: BLE001
                results[i] = {"task_id": row["task_id"], "code": "", "prompt": row["prompt"],
                              "test": row["test"], "entry_point": row["entry_point"],
                              "error": str(exc)[:200]}
            with lock:
                done[0] += 1
                if done[0] % 20 == 0:
                    print(f"  {done[0]}/{len(rows)}  {time.strftime('%H:%M:%S')}", flush=True)
            work.task_done()

    t0 = time.time()
    ts = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    el = time.time() - t0
    json.dump({"effort": args.effort, "max_tokens": max_tokens, "workers": args.workers,
               "elapsed_s": el, "results": results}, open(args.out, "w"))
    print(f"{len(rows)} tasks in {el:.0f}s ({tokens[0]} tok, {tokens[0] / el:.1f} tok/s total "
          f"over {args.workers} streams); truncated: {truncated[0]} -> {args.out}")


if __name__ == "__main__":
    main()
