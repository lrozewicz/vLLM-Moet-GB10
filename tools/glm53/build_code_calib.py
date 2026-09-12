#!/usr/bin/env python3
"""Build a "code" calibration corpus for GPTQ planes (tools/glm53/gptq_build.py).

The lesson that shaped this script: GPTQ learns the distribution of its
calibration text. Our first run calibrated on a mix that was mostly Markdown
documentation plus Polish Wikipedia - Polish perplexity improved 3.7%, code
got WORSE and HumanEval dropped 3.6 points. Calibrating on real source code
from many repositories and languages, with a small share of English docs and
(optionally) prose in the language you care about, improved all three
domains at once (code/EN/PL perplexity -0.7/-1.3/-3.2%, HumanEval 95.7 ->
97.0%).

Point it at LOCAL clones of open-source repositories (nothing is downloaded).
Files that contain anything that looks like a secret are skipped, but you are
still responsible for what you feed it: only use code you are allowed to
process, never private repositories you would not publish.

    build_code_calib.py --repo ~/src/llama.cpp --repo ~/src/keycloak \\
        --repo ~/src/ComfyUI --prose ~/corpus/pl-wiki.txt --out calib_code.txt

The reference corpus (5.6 MB) came from llama.cpp, ik_llama.cpp, RAGFlow,
Keycloak, ComfyUI, vLLM-Moet and a few other OSS projects: Python 28%,
C/C++/CUDA 21%, TS/JS 17%, Java 6%, shell/SQL/YAML 4%, Go/Rust 2%, EN docs
11%, Polish prose 11%.
"""
import argparse
import os
import random
import re

# share of the budget, in characters
MIX = {
    "py": (0.26, (".py",)),
    "tsjs": (0.16, (".ts", ".tsx", ".js", ".mjs")),
    "java": (0.10, (".java", ".kt")),
    "cfam": (0.20, (".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh")),
    "gors": (0.04, (".go", ".rs")),
    "sh": (0.04, (".sh", ".bash", ".sql", ".yaml", ".yml", ".toml")),
    "md_en": (0.10, (".md",)),
}
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "vendor", "third_party",
             "__pycache__", ".venv", "venv", "site-packages", "target", ".next"}
SECRET = re.compile(
    r"(?i)(api[_-]?key|secret|passw(or)?d|token|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{16,}"
    r"|BEGIN [A-Z ]*PRIVATE KEY|VLLM_API_KEY=|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}"
    r"|[A-Fa-f0-9]{40,}")
NON_ASCII_LETTER = re.compile(r"[^\x00-\x7f]")


def candidates(repos, exclude):
    by = {k: [] for k in MIX}
    for base in repos:
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")
                      and not any(x in d.lower() for x in exclude)]
            for fn in fns:
                if fn.endswith((".min.js", ".lock", ".map")) or "generated" in fn:
                    continue
                p = os.path.join(dp, fn)
                for k, (_, exts) in MIX.items():
                    if fn.endswith(exts):
                        try:
                            sz = os.path.getsize(p)
                        except OSError:
                            break
                        if 800 <= sz <= 120_000:
                            by[k].append((base, p))
                        break
    return by


def read_ok(p, lang):
    try:
        t = open(p, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        return None
    if SECRET.search(t):
        return None
    lines = t.splitlines()
    if not lines or max(len(x) for x in lines) > 400:      # minified / data
        return None
    # English docs only: a Markdown file full of non-ASCII letters is not
    if lang == "md_en" and len(NON_ASCII_LETTER.findall(t)) > 0.01 * len(t):
        return None
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", action="append", required=True,
                    help="local clone of an open-source repository (repeatable)")
    ap.add_argument("--exclude-dir", action="append", default=[],
                    help="skip directories whose name contains this (repeatable)")
    ap.add_argument("--prose", default=None,
                    help="optional prose file in your language, mixed in at --prose-share")
    ap.add_argument("--prose-share", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    repos = [os.path.abspath(os.path.expanduser(r)) for r in a.repo]
    exclude = [x.lower() for x in a.exclude_dir]
    rnd = random.Random(a.seed)
    budget = int(a.mb * 1e6)
    # source balance: one repository may fill at most this share of a language
    cap = max(0.4, 1.0 / len(repos))
    by = candidates(repos, exclude)
    docs, stats = [], {}
    for lang, (share, _) in MIX.items():
        files = by[lang]
        rnd.shuffle(files)
        got, n, skipped = 0, 0, 0
        per_repo = {}
        for base, p in files:
            if got >= share * budget:
                break
            if per_repo.get(base, 0) > cap * share * budget:
                continue
            t = read_ok(p, lang)
            if t is None:
                skipped += 1
                continue
            rel = os.path.join(os.path.basename(base), os.path.relpath(p, base))
            docs.append(f"# file: {rel}\n{t}\n")
            got += len(t)
            n += 1
            per_repo[base] = per_repo.get(base, 0) + len(t)
        stats[lang] = (n, got, skipped, len(files),
                       {os.path.basename(r): round(v / 1e3) for r, v in per_repo.items()})
    if a.prose:
        # prose in ~8 kB pieces, shuffled in between the code files
        t = open(a.prose, encoding="utf-8").read()[:int(a.prose_share * budget)]
        for i in range(0, len(t), 8000):
            docs.append(t[i:i + 8000] + "\n")
        stats["prose"] = (len(t) // 8000 + 1, len(t), 0, 0,
                          {os.path.basename(a.prose): round(len(t) / 1e3)})
    rnd.shuffle(docs)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(docs))
    total = sum(v[1] for v in stats.values())
    for lang, (n, got, sk, cand, rep) in stats.items():
        print(f"{lang:6} {n:5} files {got / 1e6:5.2f} MB ({100 * got / max(1, total):4.1f}%)  "
              f"skipped {sk}, candidates {cand}  {rep}")
    print(f"-> {a.out}: {total / 1e6:.2f} MB, {len(docs)} documents")


if __name__ == "__main__":
    main()
