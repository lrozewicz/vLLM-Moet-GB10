"""Jaka czesc IDENTYCZNEGO promptu trafia w prefix cache przy kolejnych
powtorzeniach - i czy zalezy to od dlugosci.

Jesli udzial trafien nie rosnie z powtorzeniami, ograniczenie jest
strukturalne (granulacja checkpointow stanu KDA / bloku), a nie kwestia
rozgrzania. To tlumaczy caly "podatek na ture agenta".
"""
import json, os, re, time, urllib.request

KEY = os.environ["VLLM_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
BASE = "http://localhost:8000"


def m():
    raw = urllib.request.urlopen(urllib.request.Request(
        BASE + "/metrics", headers=H), timeout=60).read().decode()
    g = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        x = re.match(r"([a-z_:]+)\{[^}]*\}\s+([0-9.eE+-]+)", line)
        if x and x.group(1) in ("vllm:prefix_cache_hits_total",
                                "vllm:prefix_cache_queries_total",
                                "vllm:request_prefill_time_seconds_sum"):
            g[x.group(1)] = g.get(x.group(1), 0.0) + float(x.group(2))
    return g


def run(prompt):
    b0 = m()
    t0 = time.perf_counter()
    first = None
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/completions", headers=H,
        data=json.dumps(dict(model="glm-5.3-flash", prompt=prompt, stream=True,
                             max_tokens=4, temperature=0)).encode()), timeout=900)
    for raw in r:
        if raw.startswith(b"data: ") and raw[6:].strip() != b"[DONE]":
            o = json.loads(raw[6:].strip())
            if o.get("choices") and first is None and o["choices"][0].get("text"):
                first = time.perf_counter()
    dt = (first or time.perf_counter()) - t0
    time.sleep(2)
    b1 = m()
    q = b1.get("vllm:prefix_cache_queries_total", 0) - b0.get("vllm:prefix_cache_queries_total", 0)
    h = b1.get("vllm:prefix_cache_hits_total", 0) - b0.get("vllm:prefix_cache_hits_total", 0)
    pf = b1.get("vllm:request_prefill_time_seconds_sum", 0) - b0.get("vllm:request_prefill_time_seconds_sum", 0)
    return dt, q, h, pf


word = "kontener obraz wdrozenie pamiec kwantyzacja ekspert warstwa token "
for n in (300, 600, 1500, 5000):
    text = "Przeanalizuj:\n\n" + word * n
    print(f"\n--- prompt ~{n} slow ---", flush=True)
    for i in range(2):
        dt, q, h, pf = run(text)
        print(f"  powtorzenie {i}: TTFT {dt:6.2f} s | prefill {pf:6.2f} s | "
              f"trafienia {h:.0f}/{q:.0f} = {100 * h / max(q, 1):5.1f}% | "
              f"do przeliczenia {q - h:.0f} tok -> {(q - h) / 656:.1f} s", flush=True)
