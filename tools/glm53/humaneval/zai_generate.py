#!/usr/bin/env python3
"""HumanEval candidates from the ORIGINAL GLM-5.3-Flash on the z.ai API - the same
prompt, instruction and code extraction as generate.py, so execute.py and
report.py score the original and the local server the same way.

The z.ai API returns no logprobs, so a KLD against the original is not possible
there; task results are. Thinking cannot be disabled on z.ai; reasoning_effort
low matches the local runs. Temperature 0 is sent with do_sample=false.

The key is read from a file (never from the command line, never printed).
A GLM Coding Plan key works on the coding endpoint (the default):
    ZAI_KEY_FILE=~/.config/zai/key python3 zai_generate.py --effort low --out zai-gen-low.json
    # pay-as-you-go keys: ZAI_BASE=https://api.z.ai/api/paas/v4
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate import INSTRUCTION, extract  # noqa: E402

BASE = os.environ.get("ZAI_BASE", "https://api.z.ai/api/coding/paas/v4").rstrip("/")
KEY = open(os.path.expanduser(os.environ.get("ZAI_KEY_FILE", "~/.config/zai/key"))).read().strip()
MODEL = os.environ.get("ZAI_MODEL", "glm-5.3-flash")


def chat(prompt, max_tokens, effort, temperature):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "thinking": {"type": "enabled"}, "reasoning_effort": effort}
    if temperature == 0:
        body["do_sample"] = False
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    for attempt in range(8):
        t0 = time.time()
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
            ch = d["choices"][0]
            return {"e2e": time.time() - t0,
                    "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
                    "finish_reason": ch.get("finish_reason"),
                    "content": ch["message"].get("content") or "",
                    "reasoning_chars": len(ch["message"].get("reasoning_content") or "")}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "humaneval.jsonl"))
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.data)]
    if a.limit:
        rows = rows[:a.limit]
    results = [None] * len(rows)
    work = queue.Queue()
    for i in range(len(rows)):
        work.put(i)
    lock = threading.Lock()
    done = [0]

    def worker():
        while True:
            try:
                i = work.get_nowait()
            except queue.Empty:
                return
            row = rows[i]
            try:
                r = chat(INSTRUCTION + row["prompt"], a.max_tokens, a.effort, a.temperature)
                results[i] = {"task_id": row["task_id"], "code": extract(r["content"]),
                              "prompt": row["prompt"], "test": row["test"],
                              "entry_point": row["entry_point"], **{k: r[k] for k in r if k != "content"}}
            except Exception as exc:  # noqa: BLE001
                results[i] = {"task_id": row["task_id"], "code": "", "prompt": row["prompt"],
                              "test": row["test"], "entry_point": row["entry_point"],
                              "error": str(exc)[:200]}
            with lock:
                done[0] += 1
                if done[0] % 20 == 0:
                    print(f"  {done[0]}/{len(rows)} {time.strftime('%H:%M:%S')}", flush=True)

    t0 = time.time()
    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    errors = sum(1 for r in results if r and r.get("error"))
    json.dump({"source": f"z.ai {MODEL}", "effort": a.effort, "max_tokens": a.max_tokens,
               "temperature": a.temperature, "elapsed_s": time.time() - t0, "results": results},
              open(a.out, "w"))
    print(f"{len(rows)} tasks in {time.time() - t0:.0f}s, errors: {errors} -> {a.out}")


if __name__ == "__main__":
    main()
