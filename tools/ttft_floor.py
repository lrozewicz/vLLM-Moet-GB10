"""Z czego sklada sie TTFT, gdy prefiks jest JUZ w cache'u.

Wysyla ten sam dlugi prompt dwa razy pod rzad (drugi raz to pelne trafienie
w prefix cache, wiec compute ~0) i porownuje z promptem krotkim. Roznica
miedzy "krotki" a "dlugi, w pelni cache'owany" to koszt, ktory skaluje sie
z DLUGOSCIA PROMPTU, a nie z iloscia nowych tokenow - czyli tokenizacja,
transport i scheduling. To jest ten staly narzut ~8.5 s z pomiaru agenta.
"""
import json, os, sys, time, urllib.request

KEY = os.environ["VLLM_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}


def ttft(text, tag):
    body = {"model": "glm-5.3-flash", "stream": True, "max_tokens": 4,
            "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"},
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": text}]}
    t0 = time.perf_counter()
    first = None
    ptok = 0
    r = urllib.request.urlopen(urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=json.dumps(body).encode(), headers=H), timeout=900)
    for raw in r:
        if not raw.startswith(b"data: "):
            continue
        chunk = raw[6:].strip()
        if chunk == b"[DONE]":
            break
        o = json.loads(chunk)
        if o.get("usage"):
            ptok = o["usage"]["prompt_tokens"]
        if o.get("choices") and first is None:
            first = time.perf_counter()
    dt = (first or time.perf_counter()) - t0
    print(f"{tag:28s} prompt={ptok:7d} tok   TTFT={dt:6.2f} s", flush=True)
    return ptok, dt


word = "kontener obraz wdrozenie pamiec kwantyzacja ekspert warstwa token "
short = "Powiedz 'ok'."
for n_words, label in ((0, "krotki"), (6000, "~10K"), (20000, "~34K"),
                       (40000, "~68K")):
    text = short if n_words == 0 else ("Przeanalizuj:\n\n" + word * n_words)
    p1, t1 = ttft(text, f"{label}, ZIMNY")
    p2, t2 = ttft(text, f"{label}, w cache'u")
    if p1:
        print(f"    -> narzut przy pelnym trafieniu: {t2:.2f} s "
              f"({1000 * t2 / max(p1, 1):.3f} ms/token promptu)", flush=True)
