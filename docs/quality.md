# Quality — methodology, ablation, and delta recovery

The headline tables are in the [README](../README.md). This is the detailed companion.

## What we read and compare to
Baseline = the **untouched official DeepSeek‑V4‑Flash checkpoint** (~149 GiB), verified from its
safetensors headers: MoE experts are **FP4** (`I8`, e2m1 packed 2/byte) with **`F8_E8M0`**
block‑32 scales (`layers.0..42.ffn.experts.E.{w1,w2,w3}.weight`, 33,024 tensors); dense and
attention are **FP8** (`F8_E4M3`, block‑128, per `config.json`). This project changes **only the
expert codes** (FP4 → 2‑bit symmetric) — scales, dense FP8, and headers stay byte‑identical — and
runs on the *same* serving stack (only `MODEL_DIR` differs).

## Probe protocol (`tools/probe_quant_quality.py`)
1. **MTP acceptance** — mean accepted draft length + avg draft‑acceptance rate over 3×512‑token
   greedy generations (`ignore_eos`). Primary fidelity proxy (drafter+target share the model, so a
   degraded target lowers acceptance). Noise band ≈ ±0.05.
2. **Coherence** — 12 fixed prompts, exact‑match vs the official‑checkpoint outputs + manual verdict.
3. **Arithmetic** — 5 multi‑step problems (the official model itself tops out at 3/5).
4. **Long context** — passphrase needle at 100K/250K/440K depths (`tools/needle_probe.py`).

## Bits‑vs‑quality ablation (same stack, only the expert codes change)
| codebook | bits | MTP acc. length | draft accept % | arithmetic | coherence |
|---|---|---|---|---|---|
| FP4 (16‑level) — official | 4.0 | 2.68 | 84.1 | 3/5 | 12/12 |
| K=8 (3‑bit) | 3.0 | 2.60 | 79.8 | 3/5 | 12/12 |
| K=6 (~2.58‑bit) | 2.58 | 2.68 | 84.0 | 2/5 | 11/12 |
| **K=4 naive 2‑bit (asymmetric)** | 2.0 | **1.00** | **0.0** | 0/5 | **0/12 — broken** |
| **K=4 tensor‑sym `{−4,−1,1,4}` (ours)** | 2.0 | **2.73** | **86.3** | 2/5 | **12/12** |

## The finding: it was sign bias, not magnitude
Naive 2‑bit collapses to a degenerate loop (`}<?}<?…`). The optimal‑L2 2‑bit codebook goes
**sign‑asymmetric** (drops one sign's tail, e.g. `{−3,−1,0.5,4}`); accumulated per‑expert sign
bias over 43 layers destroys the model. Forcing a **sign‑symmetric** codebook at the *same* L2
error restores MTP acceptance to **at/above** the official FP4 experts (2.73/86.3 vs 2.68/84.1),
12/12 coherent. Symmetry — not granularity — is the lever (33,023/33,024 tensors pick the same
`{−4,−1,1,4}`).

## Delta recovery (FP4 hot‑expert tier)
Symmetric 2‑bit already matches the official experts on acceptance + coherence. The residual is
**numerical precision** (multi‑step arithmetic) — the 2‑bit base is "coherent but numerically
sloppy." Caching the hot experts at FP4 closes it:

| | 2‑bit base only | **2‑bit + FP4 delta** |
|---|---|---|
| arithmetic, e.g. `17×23−100` | ✗ | **✓ (=291)** |
| single‑stream decode | 151 tok/s | **143 tok/s** (≈ −0.6 ms/step) |
| extra VRAM | 0 | **~2 GiB** (170 hot‑expert slots) |

Production 1‑GPU end‑to‑end A/B (warmed): MTP acceptance with the delta tier **2.725 / 2.744**
(mean/median, σ 0.136) — at/above the same‑run two‑card FP4 control **2.546 / 2.496** (σ 0.172).

## Studying the precision tiering

The delta tier is a *workload‑adaptive* mixed‑precision scheme — the set of FP4 experts is
chosen at runtime from routing. The manager is instrumented so this can be measured directly
(default off; logging only, no effect on the serving path):

```bash
VLLM_MOE_W2_DELTA_TRACE=1                 # coverage + churn summary + per-layer FP4 histogram
VLLM_MOE_W2_DELTA_TRACE=2                 # + one line per promote/evict event
VLLM_MOE_W2_DELTA_TRACE_EVERY=64          # ticks between summaries
VLLM_MOE_W2_DELTA_DUMP=/tmp/delta.json    # atomic precision-map snapshots (tail-able)
```

Each summary reports filled slots, coverage (`cached / layers×experts`), per‑window and
cumulative promote/evict counts, and FP4 experts per layer; the JSON dump is the full
`{layer: [fp4 expert ids]}` map plus counters. `DeltaTier.precision_of(layer, expert)` and
`precision_map()` expose the same state programmatically. `tools/delta_trace_demo.py` drives
the real manager with synthetic Zipfian routing so you can see this output (and validate the
instrumentation) in ~1 s without loading the model. Questions this makes answerable:

- **Coverage vs. pool size** — sweep `VLLM_MOE_W2_DELTA_GB` and watch how coverage % and the
  arithmetic/recall probes move; find the knee where extra VRAM stops buying quality.
- **Working‑set stability** — cumulative evictions / promotions over a long run is a churn
  rate; low churn means routing is concentrated and a small pool suffices.
- **Per‑layer concentration** — the histogram shows whether some layers route to far more
  distinct experts than others (where precision matters most).

These are open questions the design invites; the README tables are the headline, this is the
instrumentation to push further.

## Gate operating point: why we stay at tau=0.60 / promote=64

The confidence gate has two live knobs (see the compose comments for the file-based
override). `tau` decides **when** the safety net fires: the gate takes the `max_prob` of the
least-confident row the sampler can actually emit and fires when it drops to `tau` or below —
the step is then re-decided with its routed experts upgraded from 2-bit to FP4.
`max_promote` decides **how much** each fire upgrades: a cap on the (layer, expert) pairs
pulled up per fired step, most-needed first.

Lowering both makes the gate cheaper: fewer steps trigger it, and each trigger upgrades less.
The question is what that buys, and what it costs — and both answers turned out to depend on
the workload.

**Throughput — and it depends entirely on the workload** (GB10, image `v031`, spec-mask on,
500-token runs, 2026-08-09). `tau` thresholds *confidence*, so how often the gate fires — and
therefore what it costs — is a property of the traffic, not of the setting alone:

| workload | fires at tau=0.60 | tau=0.60 / mp=64 | tau=0.30 / mp=16 | cost of the safe point |
|---|---|---|---|---|
| code, temperature 0 | 2.3% of steps | 21.7 tok/s (n=3, 20.7–22.9) | 22.6 (n=3, 21.3–23.7) | −0.8 tok/s (**−4%**) |
| prose (PL), temperature 1.0 | 11.8% of steps | 12.8 tok/s (n=11, 10.7–14.5) | 15.6 (n=8, 13.5–16.4) | −2.8 tok/s (**−18%**) |
| prose (EN), temperature 1.0 | 13.9% of steps | 12.9 tok/s (n=3, 11.9–13.9) | 17.5 (n=3, 16.7–18.0) | −4.5 tok/s (**−26%**) |

**On the workload this box actually serves — agentic coding — the protective setting costs
about 4%,** and the two points are barely separable given the run-to-run spread. Greedy
decoding on code is confident, so the safety net almost never triggers no matter where `tau`
sits (2.3% of steps at 0.60, 0.2% at 0.30).

On sampled prose the penalty is real and larger than we first measured: 18% in Polish, 26% in
English. Note what the numbers do at the two points — at 0.30/16 English runs clearly faster
than Polish (17.5 vs 15.6 tok/s), but at 0.60/64 both collapse to ~12.9. Once the gate fires on
one step in seven it *is* the bottleneck, and whatever the language was worth disappears into
it. Chat-style prose is the Open WebUI path here, not the primary one, but anyone serving that
traffic should weigh 0.30/16 on its own merits.

Two measurement notes. The sweep's original "~38% of steps fire" for prose predates the
spec-mask (v028); with the mask the same point measures 11.8% — that change is exactly why
throughput moved 7.9 → 14.3. And a pool left cold for the gate (14 h served at tau=0.30)
starts near 12.4 tok/s and climbs to ~13.5 as fires warm it: measure warm.

Recorded at sweep time for reference: 14.3 tok/s at 0.60/64, 15.8 at 0.60/16, 15.9 at 0.30/16,
16.5 with the gate quiet (tau=0), all on the prose prompt.

**Quality**, full suite at both points (`bench-compare`, `results/moet` = 0.60/64,
`results/moet-tau030` = 0.30/16, temperature 0):

| | 0.60 / 64 | 0.30 / 16 | flips (down/up) | McNemar |
|---|---|---|---|---|
| HumanEval (164) | 93.9% | 92.1% | 4 / 1 | p=0.375 |
| HumanEval+ (164) | 90.2% | 87.8% | 6 / 2 | p=0.289 |
| MBPP (378) | 89.7% | 89.7% | 4 / 4 | p=1.000 |
| MBPP+ (378) | 73.5% | 73.0% | 10 / 8 | p=0.815 |
| BFCL lite (300) | 89.3% | 88.3% | — | — |
| needle (40) | 100% | 100% | 0 / 0 | — |
| GPQA Diamond | **incomplete** | **incomplete** | — | — |

**No individual result is significant.** MBPP flips are symmetric (4↔4, 10↔8) — single-task
instability, not regression. HumanEval leans down (4:1, 6:2) but at p≈0.3 that is not a signal.

**We ship 0.60/64.** On the traffic this box serves the safe point costs ~4%, and even the
worst-case 26% would not buy a quality risk this suite cannot resolve, for three reasons:

1. **The suites that came back clean are the short ones.** HumanEval, MBPP and BFCL are
   single-turn tasks of a few hundred tokens. The gate protects *uncertain* tokens, and
   uncertainty compounds over long sessions and hard multi-step programming work — exactly the
   regime none of these measure.
2. **The one suite that is sensitive here never finished.** GPQA Diamond is where promotion
   policy already showed a real effect (uncapped vs capped: 60.6% vs 70.2%, McNemar p=0.002 —
   see above). Its top-up run for this pair was interrupted, so the pair is unresolved precisely
   where it matters most.
3. **The knobs trade recall, and that is measurable in principle.** At tau=0.60 the gate catches
   roughly two thirds of the cases where FP4 would have decided a token differently than 2-bit.
   Dropping to 0.30 hands the 0.30–0.60 confidence band back to 2-bit. "No visible loss on short
   coding tasks" is not the same as "no loss".

If someone wants the faster point, the knobs are live-tunable with no restart (write `0.30` and
`16` to the knob files); this is a defaults decision, not a capability limit. Reopen it when the
GPQA pair is complete.

## Reproduce
```bash
# Native reference collection (internal baseline recipe, pinned model/context)
python3 bench/runner/bench.py run \
  --recipe deepseek-v4-flash/pro6000x2-tp2-native \
  --box bench/boxes/<box>.yaml --release <native-release> \
  --suite quality-native

# Candidate paired against strict baselines in bench/baselines/registry.yaml
python3 bench/runner/bench.py run \
  --recipe deepseek-v4-flash/pro6000x2-tp2-maxq \
  --box bench/boxes/<box>.yaml --release <quality-release> \
  --suite quality
```
Schema‑2 quality runs bind the exact recipe, effective suite, model revision,
serving context, canonical serve-argument hash, evaluator hash, raw artifact
and native baseline hash. Candidate and native serving geometry must match
exactly—not only context, but TP/speculation, batching, sequence count and
compilation arguments. DS4 greedy GPQA moved about five points between native
MTP‑2 and no-spec at the same 24k window; that geometry delta is larger than
most candidate effects. GSM8K‑200 remains the fast screen; paired GPQA/McNemar,
completion-token behavior, THINK mode and needle retrieval form the release
gate.

For non‑think GPQA, token parity is gated on the paired per-item median and
geometric-mean token ratios, an exact sign test, and additional truncations.
Raw mean, aggregate p50/p90 and the runaway census remain reported for audit.
This is deliberate: native C2 runs with identical accuracy differed by 33% in
mean tokens from one chronic tail, while identical-accuracy candidate reruns
moved aggregate p90 by over 6%. The paired robust gate still catches systematic
inflation such as the measured 2-bit prefill-KV regression without making a
few unstable trajectories the release oracle. Lightweight
arithmetic/coherence tools remain useful for debugging, but cannot certify
native-grade quality.
