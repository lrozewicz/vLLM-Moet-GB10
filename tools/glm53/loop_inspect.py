#!/usr/bin/env python3
"""Tell a degenerate loop from legitimate code structure.

The prose repetition metric (share of repeated word 6-grams, quality_pl.py)
does NOT detect loops in code - it measures code's structure: a test file has
ten `self.assertEqual(`, a Go file has thirty `}`. Measured on 16 long code
samples of a healthy model: 6/16 above the 6% prose threshold, 0/16 actually
looping. A real loop copies: the same non-trivial line or block again and
again, in a row. So a sample is DEGENERATE when

  - repeated word 20-grams exceed 5% of the text, or
  - the same non-trivial line (or block of 2-4 lines) repeats >= 3 times in a row.

    loop_inspect.py LR-loop.json [LGC-loop.json] [--top 3]
"""
import argparse
import json
from collections import Counter


def rep_n(text, n):
    """Share of repeated word n-grams (quality_pl.repetition with another n)."""
    w = text.split()
    if len(w) < n * 3:
        return 0.0
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = Counter(g)
    return sum(v for v in c.values() if v > 1) / len(g)


def max_run(text, minlen=15):
    """Longest run of the SAME non-trivial line (>= minlen chars) in a row, or
    of a repeated block of 2-4 lines - a real loop copies, code structure does not."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    best = 1
    for blk in (1, 2, 3, 4):
        i = 0
        while i + blk <= len(lines):
            b = lines[i:i + blk]
            if sum(len(x) for x in b) < minlen:
                i += 1
                continue
            k = 1
            while lines[i + k * blk:i + (k + 1) * blk] == b:
                k += 1
            best = max(best, k)
            i += 1
    return best


def degenerate(text):
    r20, run = rep_n(text, 20), max_run(text)
    return (r20 > 0.05 or run >= 3), r20, run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--limit", type=int, default=1200)
    a = ap.parse_args()
    for path in a.files:
        d = json.load(open(path))
        rows = d["rows"]
        tag = d["summary"]["tag"]
        print(f"===== {tag}: {path}")
        for r in rows:
            lines = [x.strip() for x in r["text"].splitlines() if x.strip()]
            uniq = len(set(lines))
            hit = "LIMIT" if r["tokens"] >= a.limit else ""
            r["_uniq"] = uniq / max(1, len(lines))
            r["_deg"], r["_r20"], r["_run"] = degenerate(r["text"])
            print(f"{r['prompt']:10} #{r['sample']} {r['tokens']:5} tok {hit:5} "
                  f"rep6 {100 * r['repetition']:5.1f}%  rep20 {100 * r['_r20']:4.1f}%  "
                  f"run {r['_run']}  unique lines {100 * r['_uniq']:3.0f}%"
                  f"{'  <<< DEGENERATE' if r['_deg'] else ''}")
        ndeg = sum(1 for r in rows if r["_deg"])
        print(f"[{tag}] DEGENERATE (rep20 > 5% or run >= 3): {ndeg}/{len(rows)}  |  "
              f"rep6 > 6%: {sum(1 for r in rows if r['repetition'] > 0.06)}/{len(rows)}")
        for r in sorted(rows, key=lambda x: -x["repetition"])[:a.top]:
            lines = [x.strip() for x in r["text"].splitlines() if x.strip()]
            print(f"--- {r['prompt']} #{r['sample']} ({100 * r['repetition']:.1f}%): "
                  "most frequent lines")
            for line, n in Counter(lines).most_common(4):
                if n > 1:
                    print(f"    {n:2}x  {line[:100]}")
            tail = " | ".join(lines[-4:])
            print(f"    ends with: {tail[:220]}")


if __name__ == "__main__":
    main()
