#!/usr/bin/env python3
"""Re-score existing HumanEval generations with HumanEval+ tests: replace each
candidate's `test` (and `prompt`/`entry_point`, which must match) with the
HumanEvalPlus-OriginFmt record of the same task_id. The model's code is kept
as generated, so no new generation is needed.

    plus_rewrite.py gen.json HumanEvalPlus-OriginFmt.jsonl gen-plus.json
    # then execute.py with EXEC_TIMEOUT=120 (the plus tests are large)

HumanEval+ data (EvalPlus, Apache-2.0), not vendored:
    curl -L https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-OriginFmt.jsonl.gz | gunzip > HumanEvalPlus-OriginFmt.jsonl
"""
import json
import sys

gen = json.load(open(sys.argv[1]))
plus = {r["task_id"]: r for r in map(json.loads, open(sys.argv[2]))}
res = gen["results"] if isinstance(gen, dict) else gen
missing = 0
for c in res:
    p = plus.get(c["task_id"])
    if p is None:
        missing += 1
        continue
    c["test"] = p["test"]
    c["entry_point"] = p["entry_point"]
    c["prompt"] = p["prompt"]
json.dump(gen, open(sys.argv[3], "w"))
print(f"{len(res)} candidates, {len(res) - missing} rewritten to HumanEval+ tests, {missing} without a plus record")
