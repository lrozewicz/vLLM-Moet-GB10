#!/usr/bin/env python3
"""Generation quality probe for a 2-bit GLM-5.3-Flash. Measures the two failure
modes observed on free-form Polish prose; English and code are the controls
and must stay clean.

  LOOP     - cycles ("po polsku, po polsku, ..."), via repeated word n-grams.
             Dictionary-free.
  GIBBERISH - invented words ("warknawy wsuhab"). Detected with the model's
             own TOKENIZER: invented words split into more pieces than real
             ones (0.42 vs 0.35 tokens/char). Weak per word, but over 300+
             words it separates reliably.

Why Polish: it is where bare 2-bit broke first and hardest (4 of 6 samples
unusable on the reference box), so it is the most sensitive canary we have.

Must run next to the server: /tokenize is served on localhost:8000 and is
usually not exposed through a proxy. Every text is saved to --out so a human
can check the verdicts.

    VLLM_API_KEY=... quality_pl.py --tag R --samples 2 --out R-qpl.json
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter

BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("VLLM_MODEL", "glm-5.3-flash")
WORD_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{3,}")
FOREIGN_RE = re.compile(r"[Ѐ-ӿ一-鿿぀-ヿ؀-ۿ]")


def _post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if KEY:
        req.add_header("Authorization", "Bearer " + KEY)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tok_per_char(text):
    """Mean tokens per word character - higher = more invented words."""
    words = WORD_RE.findall(text)
    if len(words) < 20:
        return float("nan"), len(words)
    # one request for the whole word sample, separated by newlines
    sample = words[:400]
    joined = "\n".join(" " + w for w in sample)
    n = _post("/tokenize", {"model": MODEL, "prompt": joined}, timeout=120)["count"]
    n -= len(sample)                      # subtract the newline tokens
    chars = sum(len(w) for w in sample)
    return n / chars, len(words)


def repetition(text, n=6):
    words = text.split()
    if len(words) < n * 3:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    c = Counter(grams)
    return sum(v for v in c.values() if v > 1) / len(grams)


def verdict(tpc, rep, foreign):
    """Thresholds calibrated on known samples from 2026-08-29: correct Polish
    ~0.38 tokens/char, gibberish ~0.44; a loop is unambiguous. Prose only -
    for code use loop_inspect.py (code structure trips the 6-gram metric)."""
    if rep > 0.15 or foreign > 3 or (tpc == tpc and tpc > 0.42):
        return "BAD"
    if rep > 0.06 or (tpc == tpc and tpc > 0.405):
        return "WEAK"
    return "ok"


# The Polish prompts are written without diacritics on purpose: that is how
# they were calibrated, and changing them would move the thresholds.
PROMPTS = [
    ("pl-targ", "pl", "Opisz barwnie i szczegolowo jesienny targ w malym polskim "
     "miasteczku - zapachy, dzwieki, ludzi, towary. Okolo 400 slow."),
    ("pl-opowiadanie", "pl", "Napisz krotkie opowiadanie po polsku o listonoszu, ktory "
     "w deszczowy dzien znajduje list bez adresu. Okolo 350 slow."),
    ("pl-wyjasnienie", "pl", "Wyjasnij po polsku, wlasnymi slowami, dlaczego niebo jest "
     "niebieskie, a zachody slonca czerwone. Okolo 300 slow, bez rownan."),
    ("en-market", "en", "Describe an autumn farmers market in a small town in vivid "
     "detail - smells, sounds, people, goods. About 400 words."),
    ("code-lru", "code", "Write a complete Python LRU cache module with type hints, "
     "docstrings, TTL support and pytest unit tests. Code only."),
]


def generate(prompt, opts, max_tokens, effort, seed):
    body = dict(opts, model=MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, seed=seed,
                chat_template_kwargs={"reasoning_effort": effort})
    d = _post("/v1/chat/completions", body)
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip(), d["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--only", default=None, help="run only prompts with this prefix")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    opts = {"temperature": a.temperature}
    if a.temperature > 0:
        opts["top_p"] = a.top_p
    if a.top_k:
        opts["top_k"] = a.top_k
    if a.presence_penalty is not None:
        opts["presence_penalty"] = a.presence_penalty

    rows = []
    print(f"[{a.tag}] t={a.temperature} p={a.top_p} k={a.top_k} "
          f"pp={a.presence_penalty} effort={a.effort}")
    print(f"{'prompt':<16}{'s':>2}{'tok':>6}{'tok/char':>10}{'loop':>8}{'foreign':>8}  verdict")
    for name, kind, prompt in PROMPTS:
        if a.only and not name.startswith(a.only):
            continue
        for s in range(a.samples):
            try:
                txt, n = generate(prompt, opts, a.max_tokens, a.effort, 1000 + s)
            except Exception as e:                                   # noqa: BLE001
                print(f"{name:<16}{s:>2}  ERROR: {str(e)[:80]}")
                continue
            tpc, nw = tok_per_char(txt)
            rep = repetition(txt)
            fg = len(FOREIGN_RE.findall(txt))
            v = verdict(tpc, rep, fg)
            rows.append(dict(prompt=name, kind=kind, sample=s, tokens=n, words=nw,
                             tok_per_char=tpc, repetition=rep, foreign=fg,
                             verdict=v, tag=a.tag, text=txt))
            print(f"{name:<16}{s:>2}{n:>6}{tpc:>10.3f}{rep:>8.1%}{fg:>8}  {v}", flush=True)

    pl = [r for r in rows if r["kind"] == "pl"]
    ctl = [r for r in rows if r["kind"] != "pl"]
    bad = sum(1 for r in pl if r["verdict"] == "BAD")
    weak = sum(1 for r in pl if r["verdict"] == "WEAK")
    print(f"\n[{a.tag}] PL: {bad} BAD / {weak} WEAK / {len(pl) - bad - weak} ok"
          f"   | controls: {sum(1 for r in ctl if r['verdict'] != 'ok')} flagged of {len(ctl)}")
    if a.out:
        json.dump({"tag": a.tag, "params": vars(a), "rows": rows},
                  open(a.out, "w"), indent=2, ensure_ascii=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
