#!/usr/bin/env python3
"""Run HumanEval candidates against the real unit tests.

RUN IT IN A SANDBOX - this is model-generated code. The serving image with
the network cut off is a convenient one (no extra image to pull):

    docker run --rm --network none --entrypoint python3 \\
      -v "$PWD":/w:ro -w /tmp vllm-moet-glm53:dflash2 \\
      /w/execute.py /w/gen-low.json > exec-low.json
    python3 report.py exec-low.json

Each candidate runs in its own process with a 15 s limit. The task prompt is
prepended to the model's code (helpers of HumanEval/38 and /50); the model's
own version of the function comes second, so it wins.
"""
import json
import os
import subprocess
import sys

TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "15"))
WORK = "/tmp/humaneval_run"


def main():
    gen = json.load(open(sys.argv[1]))
    cands = gen["results"] if isinstance(gen, dict) else gen
    os.makedirs(WORK, exist_ok=True)
    passed, out = 0, []
    for c in cands:
        program = (c.get("prompt", "") + "\n\n" + c["code"] + "\n\n"
                   + c["test"] + f"\n\ncheck({c['entry_point']})\n")
        path = os.path.join(WORK, c["task_id"].replace("/", "_") + ".py")
        open(path, "w").write(program)
        try:
            p = subprocess.run([sys.executable, path], capture_output=True, timeout=TIMEOUT)
            ok, err = p.returncode == 0, ("" if p.returncode == 0 else p.stderr.decode()[-200:])
        except subprocess.TimeoutExpired:
            ok, err = False, "TIMEOUT"
        except Exception as exc:  # noqa: BLE001
            ok, err = False, str(exc)[:200]
        passed += ok
        out.append({"task_id": c["task_id"], "pass": ok, "err": err,
                    "finish_reason": c.get("finish_reason"),
                    "completion_tokens": c.get("completion_tokens")})
    print(json.dumps({"total": len(cands), "passed": passed,
                      "pass@1": round(100.0 * passed / len(cands), 1), "results": out}))


if __name__ == "__main__":
    main()
