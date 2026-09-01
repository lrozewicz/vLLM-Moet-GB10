"""Gdzie ucieka czas przy pelnym trafieniu w prefix cache: w silniku czy we
frontendzie API?

Dla kazdego wariantu promptu (tekst / gotowe token ids) czyta histogramy
z /metrics PRZED i PO requescie i pokazuje przyrost sum. Jesli sumy silnika
(kolejka + prefill + inference) sa dla obu wariantow takie same, a TTFT
mierzony przez klienta sie rozni, to roznica siedzi we frontendzie
(FastAPI + tokenizacja + budowa requestu), a nie w obliczeniach.
"""
import json, os, re, time, urllib.request

KEY = os.environ["VLLM_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
BASE = "http://localhost:8000"
WANT = ("vllm:request_queue_time_seconds_sum",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_inference_time_seconds_sum",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:request_decode_time_seconds_sum",
        "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_hits_total", "vllm:gpu_prefix_cache_queries_total")


def metrics():
    raw = urllib.request.urlopen(urllib.request.Request(
        BASE + "/metrics", headers=H), timeout=60).read().decode()
    out = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"([a-z_:]+)\{[^}]*\}\s+([0-9.eE+-]+)", line)
        if m and m.group(1) in WANT:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def run(body, tag):
    before = metrics()
    t0 = time.perf_counter()
    first = None
    ptok = 0
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/completions", data=json.dumps(body).encode(), headers=H),
        timeout=900)
    for raw in r:
        if not raw.startswith(b"data: "):
            continue
        c = raw[6:].strip()
        if c == b"[DONE]":
            break
        o = json.loads(c)
        if o.get("usage"):
            ptok = o["usage"]["prompt_tokens"]
        if o.get("choices") and first is None and o["choices"][0].get("text"):
            first = time.perf_counter()
    dt = (first or time.perf_counter()) - t0
    time.sleep(2)
    after = metrics()
    d = {k: after.get(k, 0) - before.get(k, 0) for k in WANT}
    print(f"\n{tag}: prompt={ptok} tok, TTFT(klient)={dt:.2f} s")
    for k in WANT:
        if abs(d[k]) > 1e-9:
            print(f"    {k.replace('vllm:',''):42s} {d[k]:8.2f}")
    return dt


word = "kontener obraz wdrozenie pamiec kwantyzacja ekspert warstwa token "
text = "Przeanalizuj:\n\n" + word * 3000
ids = json.loads(urllib.request.urlopen(urllib.request.Request(
    BASE + "/tokenize", data=json.dumps({"model": "glm-5.3-flash",
                                         "prompt": text}).encode(),
    headers=H), timeout=300).read())["tokens"]
base = dict(model="glm-5.3-flash", stream=True, max_tokens=4, temperature=0,
            stream_options={"include_usage": True})

run(dict(base, prompt=text), "rozgrzewka (zapelnia cache)")
a = run(dict(base, prompt=text), "TEKST, w cache'u")
b = run(dict(base, prompt=ids), "TOKEN IDS, w cache'u")
print(f"\nroznica TTFT klienta: {a - b:+.2f} s")
