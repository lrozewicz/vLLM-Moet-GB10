#!/usr/bin/env python3
"""Where did the tokens go? Repeats one loop_probe task with the given seeds and
reports the length of the `reasoning` field and of `content` separately.

Written to explain an empty answer at a full token budget: with
reasoning_effort=low the reasoning over one task reached ~750 tokens, and in
one draw it used the whole 1200-token budget before any code - a hard limit
meeting a long reasoning, not a loop. Clients that cap thinking with
`thinking_token_budget` (see the OpenCode section of the model doc) avoid it.

    VLLM_API_KEY=... reasoning_probe.py --prompt rs-parser --seeds 1000 1001 1002
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_probe import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="rs-parser")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1000, 1001, 1002])
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="send thinking_token_budget with the request")
    a = ap.parse_args()
    prompt = dict(PROMPTS)[a.prompt]
    base = os.environ.get("VLLM_BASE", "http://localhost:8000")
    key = os.environ.get("VLLM_API_KEY", "")
    for seed in a.seeds:
        body = {"model": os.environ.get("VLLM_MODEL", "glm-5.3-flash"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": a.max_tokens, "seed": seed, "temperature": 1.0,
                "top_p": 0.95, "chat_template_kwargs": {"reasoning_effort": a.effort}}
        if a.thinking_budget is not None:
            body["thinking_token_budget"] = a.thinking_budget
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json",
                                      "Authorization": "Bearer " + key})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        m = d["choices"][0]["message"]
        rs = m.get("reasoning") or m.get("reasoning_content") or ""
        ct = m.get("content") or ""
        print(f"{a.prompt} seed {seed}: reasoning {len(rs):6} chars, content {len(ct):6} chars, "
              f"tokens {d['usage']['completion_tokens']}, finish {d['choices'][0]['finish_reason']}",
              flush=True)


if __name__ == "__main__":
    main()
