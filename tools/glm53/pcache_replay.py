#!/usr/bin/env python3
"""Walk a captured agent session turn by turn to hunt prefix-cache state poisoning.

The request body captured from the failing session (messages + tools + sampling)
is replayed at every agent turn boundary (prefix of messages ending with a user
or tool message), in order, exactly as the agent sent them - so every step
restores KDA state from prefix-cache checkpoints written by the previous steps.
Each pass uses its own cache_salt (a fresh cache namespace; steps within a pass
share it). A step that looks corrupted (draft acceptance ~0, very low logprobs,
foreign scripts) is re-run at once with a new salt (no hits).

    VLLM_API_KEY=... pcache_replay.py body.json out.jsonl [minutes]
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
KEY = os.environ["VLLM_API_KEY"]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "300"))
FOREIGN = re.compile(r"[Ѐ-ӿ一-鿿぀-ヿ؀-ۿ가-힯]")
METRICS = {"vllm:spec_decode_num_draft_tokens_total": "drafted",
           "vllm:spec_decode_num_accepted_tokens_total": "accepted",
           "vllm:prefix_cache_hits_total": "hits",
           "vllm:num_requests_running": "running"}
USE_LOGPROBS = True


def metrics():
    out = {}
    for line in urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        if name in METRICS:
            out[METRICS[name]] = out.get(METRICS[name], 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def run(base_body, messages, seed, salt):
    global USE_LOGPROBS
    body = {k: v for k, v in base_body.items()
            if k not in ("messages", "stream", "stream_options", "max_completion_tokens")}
    body.update(messages=messages, max_tokens=MAX_TOKENS, seed=seed, cache_salt=salt)
    if USE_LOGPROBS:
        body["logprobs"] = True
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    m0 = metrics()
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    except urllib.error.HTTPError as e:
        if USE_LOGPROBS and e.code == 400:
            USE_LOGPROBS = False
            print("[replay] logprobs rejected - continuing without", flush=True)
            return run(base_body, messages, seed, salt)
        raise
    dt = time.time() - t0
    m1 = metrics()
    ch = d["choices"][0]
    msg = ch["message"]
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    calls = " ".join((c.get("function") or {}).get("name", "") + " " +
                     (c.get("function") or {}).get("arguments", "") for c in (msg.get("tool_calls") or []))
    lps = [t["logprob"] for t in ((ch.get("logprobs") or {}).get("content") or [])
           if t.get("logprob") is not None]
    text = reasoning + content + calls
    drafted = m1.get("drafted", 0) - m0.get("drafted", 0)
    accepted = m1.get("accepted", 0) - m0.get("accepted", 0)
    return {"seconds": round(dt, 1), "prompt_tokens": d["usage"]["prompt_tokens"],
            "completion_tokens": d["usage"]["completion_tokens"], "finish": ch.get("finish_reason"),
            "hit_tokens": int(m1.get("hits", 0) - m0.get("hits", 0)),
            "drafted": int(drafted), "accepted": int(accepted),
            "acc_rate": (accepted / drafted) if drafted else None,
            "mean_lp": (sum(lps) / len(lps)) if lps else None,
            "foreign": len(FOREIGN.findall(text)), "foreign_ratio": len(FOREIGN.findall(text)) / max(1, len(text)),
            "running_before": m0.get("running"), "running_after": m1.get("running"),
            "n_tool_calls": len(msg.get("tool_calls") or []),
            "reasoning": reasoning[:1200], "content": content[:800], "calls": calls[:800]}


def suspicious(r):
    why = []
    if r["drafted"] >= 70 and r["acc_rate"] is not None and r["acc_rate"] < 0.05:
        why.append(f"acceptance {r['acc_rate']:.3f}")
    if r["mean_lp"] is not None and r["mean_lp"] < -3.0:
        why.append(f"mean logprob {r['mean_lp']:.2f}")
    if r["foreign_ratio"] > 0.05:
        why.append(f"foreign {r['foreign']}")
    return why


def main():
    base = json.load(open(sys.argv[1]))
    out = open(sys.argv[2], "a")
    t_end = time.time() + 60 * (float(sys.argv[3]) if len(sys.argv) > 3 else 70.0)
    ms = base["messages"]
    bounds = [i + 1 for i in range(len(ms) - 1)
              if ms[i]["role"] in ("user", "tool") and ms[i + 1]["role"] == "assistant"]
    npass = nsteps = nflag = nconf = 0
    while time.time() < t_end:
        npass += 1
        salt = uuid.uuid4().hex
        for k, L in enumerate(bounds):
            if time.time() >= t_end:
                break
            seed = 7000 + 100 * npass + k
            r = run(base, ms[:L], seed, salt)
            nsteps += 1
            why = suspicious(r)
            out.write(json.dumps({"pass": npass, "prefix": L, "salted_fresh": False, "why": why, **r},
                                 ensure_ascii=False) + "\n")
            out.flush()
            acc = "-" if r["acc_rate"] is None else f"{r['acc_rate']:.2f}"
            lp = "-" if r["mean_lp"] is None else f"{r['mean_lp']:.2f}"
            print(f"[{time.strftime('%H:%M:%S')}] p{npass} L{L:3} prompt {r['prompt_tokens']:6} "
                  f"hit {r['hit_tokens']:6} acc {acc:>5} lp {lp:>6} tc {r['n_tool_calls']} "
                  f"foreign {r['foreign']:3} run {r['running_before']:.0f}/{r['running_after']:.0f} "
                  f"{r['seconds']:6.1f}s{'  <<< ' + ', '.join(why) if why else ''}", flush=True)
            if why:
                nflag += 1
                s = run(base, ms[:L], seed, uuid.uuid4().hex)
                s_why = suspicious(s)
                nconf += int(not s_why)
                out.write(json.dumps({"pass": npass, "prefix": L, "salted_fresh": True, "why": s_why, **s},
                                     ensure_ascii=False) + "\n")
                out.flush()
                acc = "-" if s["acc_rate"] is None else f"{s['acc_rate']:.2f}"
                lp = "-" if s["mean_lp"] is None else f"{s['mean_lp']:.2f}"
                print(f"           fresh salt: hit {s['hit_tokens']:6} acc {acc:>5} lp {lp:>6} "
                      f"foreign {s['foreign']:3} -> {'CONFIRMED: clean without cache' if not s_why else 'ALSO bad without cache'}",
                      flush=True)
        print(f"[replay] pass {npass} done", flush=True)
    print(f"[replay] DONE: {npass} passes, {nsteps} steps, flagged {nflag}, confirmed {nconf}", flush=True)


if __name__ == "__main__":
    main()
