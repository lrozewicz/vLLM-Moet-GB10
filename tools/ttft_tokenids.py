"""Czy narzut 0.186 ms/token przy pelnym trafieniu w prefix cache to
TOKENIZACJA po stronie serwera?

Ten sam prompt wysylany dwiema drogami:
  /v1/chat/completions  -> serwer dostaje TEKST i musi go stokenizowac
  /v1/completions       -> serwer dostaje GOTOWE token ids
Oba trafiaja w ten sam prefix cache. Jesli druga droga jest znaczaco
szybsza, narzut jest tokenizacja, a nie obliczeniami.
"""
import json, os, time, urllib.request

KEY = os.environ["VLLM_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
URL = "http://localhost:8000/v1"


def post(path, body):
    t0 = time.perf_counter()
    first = None
    ptok = 0
    r = urllib.request.urlopen(urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), headers=H), timeout=900)
    for raw in r:
        if not raw.startswith(b"data: "):
            continue
        c = raw[6:].strip()
        if c == b"[DONE]":
            break
        o = json.loads(c)
        if o.get("usage"):
            ptok = o["usage"]["prompt_tokens"]
        if o.get("choices") and first is None and (
                o["choices"][0].get("text")
                or (o["choices"][0].get("delta") or {}).get("content")):
            first = time.perf_counter()
    return ptok, (first or time.perf_counter()) - t0


word = "kontener obraz wdrozenie pamiec kwantyzacja ekspert warstwa token "
text = "Przeanalizuj:\n\n" + word * 3000          # ~57K tokenow

# 1. tokenizacja po stronie serwera (endpoint tokenize) - zeby dostac ids
t0 = time.perf_counter()
r = urllib.request.urlopen(urllib.request.Request(
    "http://localhost:8000/tokenize", data=json.dumps({"model": "glm-5.3-flash",
                                           "prompt": text}).encode(),
    headers=H), timeout=900)
tok = json.loads(r.read())
t_tok = time.perf_counter() - t0
ids = tok["tokens"]
print(f"tokenizacja przez /tokenize: {len(ids)} tok w {t_tok:.2f} s "
      f"({1000 * t_tok / len(ids):.3f} ms/token)", flush=True)

base = dict(model="glm-5.3-flash", stream=True, max_tokens=4, temperature=0,
            stream_options={"include_usage": True})

for i in range(2):
    p, dt = post("/completions", dict(base, prompt=text))
    print(f"  /completions TEKST      #{i}: {p} tok, TTFT {dt:6.2f} s", flush=True)
for i in range(2):
    p, dt = post("/completions", dict(base, prompt=ids))
    print(f"  /completions TOKEN IDS  #{i}: {p} tok, TTFT {dt:6.2f} s", flush=True)
