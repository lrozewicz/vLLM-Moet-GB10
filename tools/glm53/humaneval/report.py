#!/usr/bin/env python3
"""HumanEval report: pass@1, truncations, tokens.  python3 report.py exec-low.json"""
import json
import sys

d = json.load(open(sys.argv[1]))
res = d["results"]
trunc = [r for r in res if r.get("finish_reason") == "length"]
fails = [r for r in res if not r["pass"]]
toks = [r.get("completion_tokens") or 0 for r in res]
print(f"pass@1: {d['pass@1']}%  ({d['passed']}/{d['total']})")
print(f"truncated (finish_reason=length): {len(trunc)}  -> of those failed: "
      f"{sum(1 for r in trunc if not r['pass'])}")
print(f"tokens/task: mean {sum(toks) / len(toks):.0f}, max {max(toks)}")
print("failed:", ", ".join(r["task_id"].split("/")[1]
                           + ("*" if r.get("finish_reason") == "length" else "") for r in fails))
