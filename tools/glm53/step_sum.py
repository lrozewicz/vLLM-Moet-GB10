#!/usr/bin/env python3
"""Step time (decode_s / drafts), AL and tok/s per category from bench_mtp
JSONs: usage step_sum.py TAG-bench.json ..."""
import json
import sys

for f in sys.argv[1:]:
    rows = json.load(open(f))["rows"]
    out = []
    for cat in ("code", "prose"):
        r = [x for x in rows if x["prompt"].startswith(cat)]
        dec = sum(x["decode_s"] for x in r)
        dr = sum(x["drafts"] for x in r)
        acc = sum(x["accepted"] for x in r)
        tok = sum(x["out_tokens"] for x in r)
        out.append(f"{cat}: step {dec / dr * 1000:6.1f} ms AL {1 + acc / dr:4.2f} "
                   f"{tok / dec:5.1f} tok/s")
    print(f"{f.split('/')[-1]:18s} " + " | ".join(out))
