#!/usr/bin/env python3
"""Igly w stogu dla GLM-5.3-Flash: 8 rekordow "access code for <imie>" rozsianych
po prawdziwym kodzie zrodlowym (rowne glebokosci), model ma zwrocic JSON z
wszystkimi kodami. Kazdy prompt ma unikalna sol (zero trafien w prefix cache).
Wynik: trafione igly / wszystkie, per rozmiar i per glebokosc.

    VLLM_API_KEY=... needle.py --src DIR --sizes 96000,180000 --prompts 3 --out X.json
"""
import argparse
import glob
import json
import os
import random
import re
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
KEY = os.environ["VLLM_API_KEY"]
NAMES = ["Adelajda", "Bonifacy", "Celestyna", "Dobromir", "Eufemia", "Fortunat",
         "Gertruda", "Hilary", "Ildefons", "Jadwiga", "Kunegunda", "Leopold"]


def haystack(files, n_chars, rng):
    parts, size = [], 0
    order = files[:]
    rng.shuffle(order)
    i = 0
    while size < n_chars:
        txt = open(order[i % len(order)]).read()
        parts.append(f"\n# ===== file {i}: {os.path.basename(order[i % len(order)])} =====\n{txt}")
        size += len(txt)
        i += 1
    return "".join(parts)[:n_chars]


SURNAMES = ["Nowak", "Kowalska", "Wisniewski", "Wojcik", "Kaminska", "Lewandowski"]


def build_hard(files, tokens, seed):
    """24 rekordy: 12 imion x 2 nazwiska (pary mylace sie), pytanie o 8 z nich."""
    rng = random.Random(seed)
    people = []
    for n in rng.sample(NAMES, 12):
        for sn in rng.sample(SURNAMES, 2):
            people.append(f"{n} {sn}")
    rng.shuffle(people)
    codes = {p: f"{rng.randint(100000, 999999)}" for p in people}
    text = haystack(files, int(tokens * 3.6), rng)
    pieces, last = [], 0
    for j, person in enumerate(people):
        pos = int(len(text) * (0.03 + 0.94 * j / (len(people) - 1)))
        pos = text.find("\n", pos) + 1 or pos
        pieces.append(text[last:pos])
        pieces.append(f"# NOTE: the access code for {person} is {codes[person]}.\n")
        last = pos
    pieces.append(text[last:])
    asked = rng.sample(people, 8)
    q = ("\n\nThe text above contains 24 notes 'the access code for FIRSTNAME SURNAME is CODE'. Several people "
         "share a first name, so match the FULL name. Return ONLY a JSON object mapping each of these full names "
         "to its code: " + ", ".join(asked) + ".")
    return f"[session {seed}]\n" + "".join(pieces) + q, asked, codes


def build(files, tokens, seed):
    rng = random.Random(seed)
    names = rng.sample(NAMES, 8)
    codes = {n: f"{rng.randint(100000, 999999)}" for n in names}
    text = haystack(files, int(tokens * 3.6), rng)  # ~3,6-4 znaku/token dla kodu
    # rowne glebokosci 5%..95%, wstawiane na granicy linii
    pieces, last = [], 0
    for j, n in enumerate(names):
        pos = int(len(text) * (0.05 + 0.9 * j / 7))
        pos = text.find("\n", pos) + 1 or pos
        pieces.append(text[last:pos])
        pieces.append(f"# NOTE: the access code for {n} is {codes[n]}.\n")
        last = pos
    pieces.append(text[last:])
    q = ("\n\nThe text above contains exactly 8 notes of the form 'the access code for NAME is CODE'. "
         "Return ONLY a JSON object mapping each NAME to its CODE, for all 8 notes.")
    return f"[session {seed}]\n" + "".join(pieces) + q, names, codes


def ask(prompt):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500, "temperature": 0.0, "reasoning_effort": "low"}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    ch = d["choices"][0]
    d["usage"]["finish_reason"] = ch.get("finish_reason")
    return ch["message"].get("content") or "", d["usage"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--sizes", default="96000,180000")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hard", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "*.py")))
    res = {"runs": []}
    for size in [int(x) for x in a.sizes.split(",")]:
        for p in range(a.prompts):
            seed = a.seed0 + size + p
            prompt, names, codes = (build_hard if a.hard else build)(files, size, seed)
            content, usage, dt = ask(prompt)
            hits = [n for n in names if re.search(rf"{n}\W+{codes[n]}", content)]
            run = {"size": size, "seed": seed, "prompt_tokens": usage["prompt_tokens"], "time_s": round(dt, 1),
                   "hits": len(hits), "finish": usage.get("finish_reason"), "empty": not content.strip(), "missed": [(i, n) for i, n in enumerate(names) if n not in hits],
                   "content": content[-600:]}
            res["runs"].append(run)
            print(f"size {size} seed {seed}: {len(hits)}/8 (prompt {usage['prompt_tokens']} tok, {dt:.0f} s) "
                  f"missed depth-index {[i for i, _ in run['missed']]} finish={run['finish']} empty={run['empty']}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    tot = sum(r["hits"] for r in res["runs"])
    res["total"] = f"{tot}/{8 * len(res['runs'])}"
    json.dump(res, open(a.out, "w"), indent=1)
    print("TOTAL", res["total"], flush=True)


if __name__ == "__main__":
    main()
