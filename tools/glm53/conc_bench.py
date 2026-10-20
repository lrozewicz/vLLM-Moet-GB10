#!/usr/bin/env python3
"""GLM-5.3-Flash: pomiar 1 vs 2 strumieni na zadaniu EDYCYJNYM + pojemnosc dwoch
dlugich strumieni (noc 18/19.10.2026).

Zasady pomiaru DFlash2: wyjsciem ma byc kod przepisany z
wejscia, kazdy strumien ma unikalny prefiks (inaczej prefix cache), kazdy strumien
rozgrzany, mediana z powtorek.

  edit   - N strumieni (1 i 2) x R powtorek: plik ~3-4K tok. -> "dodaj type hinty,
           zwroc caly plik". Raport: tok/s na strumien (dekodowanie po TTFT),
           suma, akceptacja z /metrics.
  cap    - 2 rownolegle prompty po L tokenow (L z --cap-sizes): czy oba dekoduja
           naraz (max running == 2), ile preempcji, TTFT, kv_cache_usage.

    VLLM_API_KEY=... conc_bench.py --src DIR --tag K5 --out K5-conc.json
"""
import argparse
import glob
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("VLLM_MODEL", "glm-5.3-flash")

EDIT_TASK = ("\n\nRewrite the whole file above with complete PEP 484 type hints on every "
             "function and method signature. Keep all code, comments and docstrings exactly "
             "as they are otherwise. Return ONLY the complete file in one ```python block.")
CAP_TASK = ("\n\nAbove are several source files. List the five functions you consider the "
            "most complex, one line each, with a one-sentence reason.")


def metrics():
    req = urllib.request.Request(BASE + "/metrics")
    out = {}
    with urllib.request.urlopen(req, timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#") or " " not in line:
                continue
            name, val = line.rsplit(" ", 1)
            base = name.split("{")[0]
            try:
                out[base] = out.get(base, 0.0) + float(val)
            except ValueError:
                pass
    return out


def stream(prompt, max_tokens, res):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "reasoning_effort": "low", "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    first = None
    usage = None
    finish = None
    text = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "") \
                        + (delta.get("reasoning") or "")
                    if piece and first is None:
                        first = time.time()
                    text.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    t1 = time.time()
    ct = (usage or {}).get("completion_tokens", 0)
    res.update({"t0": t0, "t_first": first, "t_end": t1,
                "ttft": (first - t0) if first else None,
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": ct, "finish": finish,
                "decode_tps": (ct - 1) / (t1 - first) if first and t1 > first and ct > 1 else None,
                "tail": "".join(text)[-200:]})


def run_parallel(prompts, max_tokens):
    res = [{} for _ in prompts]
    th = [threading.Thread(target=stream, args=(p, max_tokens, r)) for p, r in zip(prompts, res)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res


def spec(m0, m1):
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in (
        "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total")}
    n = d["vllm:spec_decode_num_drafts_total"]
    return {"accepted_per_draft": round(d["vllm:spec_decode_num_accepted_tokens_total"] / n, 3) if n else None}


def edit_bench(files, streams, repeats, max_tokens):
    out = []
    for rep in range(repeats + 1):  # rep 0 = rozgrzewka kazdego strumienia (nie liczona)
        prompts = []
        for s in range(streams):
            src = open(files[s % len(files)]).read()
            tag = f"# session {random.getrandbits(48):012x} stream {s}\n"
            prompts.append(tag + "```python\n" + src + "\n```" + EDIT_TASK)
        m0 = metrics()
        t0 = time.time()
        res = run_parallel(prompts, max_tokens)
        wall = time.time() - t0
        m1 = metrics()
        row = {"rep": rep, "warmup": rep == 0, "wall_s": round(wall, 1),
               "per_stream_tps": [round(r["decode_tps"], 2) if r.get("decode_tps") else None for r in res],
               "tokens": [r.get("completion_tokens") for r in res],
               "ttft": [round(r["ttft"], 2) if r.get("ttft") else None for r in res],
               "finish": [r.get("finish") for r in res],
               "errors": [r.get("error") for r in res if r.get("error")],
               "agg_tps": round(sum(r.get("completion_tokens") or 0 for r in res) / wall, 2), **spec(m0, m1)}
        print(f"  edit x{streams} rep{rep}: {row}", flush=True)
        out.append(row)
    meas = [r for r in out if not r["warmup"]]
    per = [x for r in meas for x in r["per_stream_tps"] if x]
    return {"streams": streams, "runs": out,
            "median_per_stream_tps": round(statistics.median(per), 2) if per else None,
            "median_agg_tps": round(statistics.median([r["agg_tps"] for r in meas]), 2) if meas else None}


def filler(files, n_tokens, salt):
    parts, size = [f"Session {salt}.\n"], 0
    i = 0
    while size < n_tokens * 3.3:  # ~3,3 znaku/token dla kodu Pythona (sprawdzane po usage)
        f = files[i % len(files)]
        txt = open(f).read()
        parts.append(f"\n### FILE {i}: {os.path.basename(f)}\n{txt}")
        size += len(txt)
        i += 1
    return "".join(parts)[: int(n_tokens * 3.3)]


def cap_bench(files, size):
    prompts = [filler(files[s::2] or files, size, f"{random.getrandbits(48):012x}-{s}") + CAP_TASK
               for s in range(2)]
    m0 = metrics()
    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                m = metrics()
                samples.append((time.time(), m.get("vllm:num_requests_running", 0),
                                m.get("vllm:num_requests_waiting", 0), m.get("vllm:kv_cache_usage_perc", 0)))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)

    pt = threading.Thread(target=poll)
    pt.start()
    res = run_parallel(prompts, 300)
    stop.set()
    pt.join()
    m1 = metrics()
    # Czy dekodowanie obu sie nakladalo: przedzial [t_first, t_end] kazdego.
    ov = None
    if all(r.get("t_first") for r in res):
        ov = max(0.0, min(r["t_end"] for r in res) - max(r["t_first"] for r in res))
    row = {"size": size, "prompt_tokens": [r.get("prompt_tokens") for r in res],
           "ttft": [round(r["ttft"], 1) if r.get("ttft") else None for r in res],
           "decode_tps": [round(r["decode_tps"], 2) if r.get("decode_tps") else None for r in res],
           "decode_overlap_s": round(ov, 1) if ov is not None else None,
           "max_running": max((s[1] for s in samples), default=None),
           "max_waiting": max((s[2] for s in samples), default=None),
           "max_kv_usage": round(max((s[3] for s in samples), default=0), 3),
           "preemptions": m1.get("vllm:num_preemptions_total", 0) - m0.get("vllm:num_preemptions_total", 0),
           "errors": [r.get("error") for r in res if r.get("error")],
           "finish": [r.get("finish") for r in res]}
    print(f"  cap {size}: {row}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="katalog z plikami .py (zrodlo promptow)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--cap-sizes", default="32000,64000,96000")
    ap.add_argument("--skip-edit", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "*.py")))
    edit_files = [f for f in files if 9000 <= os.path.getsize(f) <= 14000][:4]
    res = {"tag": a.tag, "time": time.strftime("%F %T")}
    try:
        info = [l for l in urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode().splitlines()
                if l.startswith("vllm:cache_config_info")]
        res["num_gpu_blocks"] = re.search(r'num_gpu_blocks="(\d+)"', info[0]).group(1) if info else None
    except Exception:  # noqa: BLE001
        res["num_gpu_blocks"] = None
    print(f"[{a.tag}] num_gpu_blocks={res['num_gpu_blocks']} edit files={[os.path.basename(f) for f in edit_files]}",
          flush=True)
    if not a.skip_edit:
        res["edit1"] = edit_bench(edit_files, 1, a.repeats, a.max_tokens)
        res["edit2"] = edit_bench(edit_files, 2, a.repeats, a.max_tokens)
    res["cap"] = []
    for s in [int(x) for x in a.cap_sizes.split(",") if x]:
        row = cap_bench(files, s)
        res["cap"].append(row)
        json.dump(res, open(a.out, "w"), indent=1)
        if row["errors"]:
            break
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[{a.tag}] edit1 {res.get('edit1', {}).get('median_per_stream_tps')} tok/s, "
          f"edit2 per-stream {res.get('edit2', {}).get('median_per_stream_tps')} / agg "
          f"{res.get('edit2', {}).get('median_agg_tps')}; cap "
          f"{[(c['size'], c['max_running'], c['preemptions']) for c in res['cap']]}", flush=True)


if __name__ == "__main__":
    main()
