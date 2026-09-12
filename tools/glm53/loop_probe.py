#!/usr/bin/env python3
"""Loop probe on LONG code: does a plane variant loop more often than another?

HumanEval cannot see this (its answers are ~150 tokens); here each task asks
for a complete module with tests, ~1200 tokens. 8 tasks x N samples, the same
sampling as quality_pl.py (temperature 1.0, top_p 0.95, reasoning_effort low)
and FIXED seeds (1000 + sample number), so two variants get the same draws.

Two measures per sample:
  rep6        share of repeated word 6-grams (quality_pl.py) - reported for
              continuity, but on code it mostly measures structure
  degenerate  loop_inspect.degenerate(): repeated 20-grams > 5% or the same
              non-trivial line/block >= 3 times in a row - the real loop test

    VLLM_API_KEY=... loop_probe.py --tag GC --out GC-loop.json
    loop_inspect.py R-loop.json GC-loop.json        # side by side, with evidence
"""
import argparse
import json
import os
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_inspect import degenerate  # noqa: E402
from quality_pl import generate, repetition  # noqa: E402

PROMPTS = [
    ("code-lru", "Write a complete Python LRU cache module with type hints, "
     "docstrings, TTL support and pytest unit tests. Code only."),
    ("py-http", "Write a complete Python asyncio HTTP client wrapper with rate "
     "limiting, retries with exponential backoff and jitter, timeouts, type hints, "
     "docstrings and pytest tests using a mock transport. Code only."),
    ("ts-events", "Write a complete TypeScript module: a strongly typed event "
     "emitter with middleware pipeline, once/off support, async listeners and error "
     "isolation, plus vitest unit tests. Code only."),
    ("go-pool", "Write a complete Go package implementing a worker pool with "
     "context cancellation, bounded queue, graceful shutdown, result collection and "
     "table-driven tests. Code only."),
    ("py-etl", "Write a complete Python CLI (argparse) that loads CSV files into "
     "SQLite with schema inference, batching, logging, idempotent upserts and "
     "pytest tests. Code only."),
    ("rs-parser", "Write a complete Rust module: a lexer and recursive-descent "
     "parser for arithmetic expressions with variables, an evaluator with error "
     "types, and unit tests. Code only."),
    ("py-api", "Write a complete FastAPI service for a todo list with Pydantic "
     "models, an in-memory repository, pagination, filtering, validation errors and "
     "pytest tests using TestClient. Code only."),
    ("sh-backup", "Write a complete Bash backup script with rotation, file locking, "
     "logging, dry-run mode, retention policy, error handling with traps and usage "
     "help. Code only."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    opts = {"temperature": 1.0, "top_p": 0.95}
    jobs = [(name, p, s) for name, p in PROMPTS for s in range(a.samples)]

    def run(job):
        name, prompt, s = job
        txt, n = generate(prompt, opts, a.max_tokens, "low", 1000 + s)
        deg, r20, run_len = degenerate(txt)
        return {"prompt": name, "sample": s, "tokens": n, "repetition": repetition(txt),
                "rep20": r20, "max_run": run_len, "degenerate": deg, "text": txt}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, jobs))
    for r in rows:
        print(f"[{a.tag}] {r['prompt']:10} #{r['sample']} {r['tokens']:5} tok  "
              f"rep6 {100 * r['repetition']:5.1f}%  rep20 {100 * r['rep20']:4.1f}%  "
              f"run {r['max_run']}{'  DEGENERATE' if r['degenerate'] else ''}", flush=True)
    reps = [r["repetition"] for r in rows]
    summary = {"tag": a.tag, "n": len(rows),
               "degenerate": sum(1 for r in rows if r["degenerate"]),
               "rep6_gt6": sum(1 for x in reps if x > 0.06),
               "mean": st.mean(reps), "median": st.median(reps), "max": max(reps)}
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"[{a.tag}] DEGENERATE {summary['degenerate']}/{len(rows)}  |  rep6 > 6%: "
          f"{summary['rep6_gt6']}/{len(rows)}, mean {100 * summary['mean']:.1f}%, "
          f"max {100 * summary['max']:.1f}%")


if __name__ == "__main__":
    main()
