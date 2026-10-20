# GLM-5.3-Flash on one DGX Spark (GB10)

**[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** is a
321B-parameter Mixture-of-Experts model (18B active per token, 288 routed
experts, top-8) with a hybrid attention stack - 34 KDA linear-attention layers
and 11 NoPE sparse-MLA layers - a vision tower, and 1M positions. Its FP8
checkpoint is 306 GiB. A GB10 box (NVIDIA DGX Spark, ASUS Ascent GX10, ...)
has 121.6 GiB of unified memory, shared between the CPU and the GPU.

This guide covers what we changed to make the model fit and run fast, what we
measured, why the quality holds, and how to reproduce it on your own box
step by step.

## At a glance

Measured on the reference box (ASUS Ascent GX10, GB10, 121.6 GiB, aarch64),
release v0.3.0 plus patches 13 (FP8 W8A16, 2026-10-20), 14 (MoE launch without
empty pairs) and 15 (adaptive draft depth, both 2026-10-21):

| | |
|---|---|
| decode, code | **~42 tok/s** (41.5-42.2 over three boots; patch 13 alone 38.9-39.2, v0.3.0 29-31) - DFlash2 k=7, acceptance length 5.5-5.6, temperature 0, `reasoning_effort` low |
| decode, prose | **21.3 tok/s** (bench prompt; fixed k=7: 17.6-18.6, v0.3.0 12.4-13.5); 18.2 on six long Polish texts (k=7: 15.0-15.8) |
| two streams | 25.4 tok/s each, 46.6 together (code edits); two 96K prompts at once without preemption |
| prefill | ~640 tok/s (643 on a 70K prompt, 632 on a 198K prompt) |
| context | **200K** (204,800); a 198K-token prompt: 313 s to the first token, zero preemptions |
| HumanEval pass@1 | **97.0%** (159/164) at `reasoning_effort` low; HumanEval+ **93.9%**. The original model on the z.ai API, same harness: 95.1% / 89.6% |
| loop probe, long code | 0 of 16 samples degenerate |
| memory | model 92.7 GiB, KV cache 12.8 GiB fixed (182 blocks, ~510K tokens) |
| boot | first-time setup ~2 h (one-time quantization and builds), then 11-19 min per boot depending on the disk |

## What it took

Four problems stood between this checkpoint and one GB10. The first is solved
by the upstream engine, the second by community work, and the last two are
ours.

### 1. Size: 2-bit experts (vLLM-Moet W2)

The routed experts are 97% of the parameters: 43 layers x 288 experts x
(gate + up + down) x 2048 x 4096 = 311.6e9 weights. The
[vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet) engine stores them as
2-bit **planes**: every weight is one of {-4, -1, 1, 4} times a per-block
scale (one UE8M0 scale per 32 weights), and hand-written SASS GEMMs compute
directly on that format. That is 1.69 GiB per layer instead of ~7 GiB.
**Everything else keeps the checkpoint's own precision**: attention and KDA,
the shared expert, the three dense layers, embeddings, the router and the
vision tower stay FP8/BF16. The BF16 part is packed losslessly into 12 bits
(`BF12`, bit-exact, verified at every boot, -2.4 GiB).

Getting the engine onto this model took two things:

- **A different vLLM.** GLM-5.3-Flash's implementation (`glm5next`: KDA, NoPE
  sparse MLA, mHC hyper-connections, the MTP head) exists only in the vendor
  image `vllm/vllm-openai:glm53-flash-*` (vLLM 0.29-dev), not in the v0.24.0
  base the rest of this fork uses.
  [`tools/apply_w2_overlay.py`](../../tools/apply_w2_overlay.py) extracts the
  model-agnostic engine (planes, kernels, FP8-to-2-bit loader hook) from the
  generated patch and installs it into that image by anchor.
- **Four gaps in the FP8 path** that only matter on unified memory: no
  streaming plane build (a staging buffer per layer), no allocator release
  in the f64 requant, and neither half of the persistent planes cache (write
  and read). Each one alone was enough to stop the boot.

### 2. Attention: the NoPE overlay (community)

This checkpoint is NoPE (`qk_rope_head_dim: 0`), and on consumer Blackwell
(sm_12x) the only sparse-MLA backend requires a 64-wide rope half. The fix is
[chriswritescode-dev/glm-5.3-flash-sm120](https://github.com/chriswritescode-dev/glm-5.3-flash-sm120),
from [vllm#53963](https://github.com/vllm-project/vllm/issues/53963): zero-pad
the 512-wide latent into the existing 576-wide GLM_NSA geometry. A zero rope
block contributes nothing to the QK product, so the result is exact, and no
CUDA is involved. Their repository has no license, so
[`build_combined.sh`](../../deploy/gb10-glm53/build_combined.sh) downloads
their Dockerfile at build time instead of vendoring it.

### 3. Speed: DFlash2 instead of the MTP head

The model's own MTP draft head is a **full MoE layer** with its own 288
experts. At k=4 its four sequential passes took ~46% of every decode step.
The [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) block
diffusion drafter (1B parameters, 5 layers, reading hidden states from target
layers 5/14/24/33/42) proposes a block of 8 tokens in **one** pass. It is
integrated through
[tonyd2wild's overlay](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark),
a port of vLLM PR #52816 made for the same vendor image; its GLM glue also
has no license and is downloaded at build time
([`build_dflash2.sh`](../../deploy/gb10-glm53/build_dflash2.sh)).

On one box it needed one more fix of ours (patch 8). With TP=1 the MLA page
is 656 bytes/token = 16 x 41, so the overlay's "exact fit" drafter KV layout
can never apply, and its fallback would charge a 7168-token drafter page to
every KV block (-58% capacity). Patch 8 gives the drafter a 1024-token block
instead (-16%).

Result: **+49% on code** against MTP k=4 (29.3 vs 19.6 tok/s), +7% on prose,
where the drafter guesses less often.

### 4. Quality: a runtime FP4 tier on the V2 runner, and GPTQ planes

Bare 2-bit is not good enough on its own (next section). Upstream's answer is
the FP4 delta tier: a pool of experts served from 4-bit instead of 2-bit. Its
per-step hooks lived in vLLM's V1 model runner, while speculative decoding on
a hybrid model works only on the V2 runner. So we ported the hooks to V2
(`patches/patch_v2_runner.py`), then made the confidence gate's replay safe
for recurrent state. The gate re-runs a step, which advances the 34 KDA
states twice; it now snapshots and rewinds them. Finally we added promotion
hysteresis, so the pool stops ping-ponging (+15% decode). The last piece is
GPTQ planes, below.

The complete list of patches, with what each one fixes, is in
[`deploy/gb10-glm53/patches/README.md`](../../deploy/gb10-glm53/patches/README.md).

### 5. Prefix cache and the recurrent state (patch 12)

The prefix cache stores the KDA state only at block boundaries (every 7168
tokens), and a cached block must hold the state after exactly that many
tokens. vLLM guarantees it by ending prefill chunks on those boundaries,
using `cache_config.block_size` as the block. With the DFlash2 drafter that
value is lowered to 1024, the drafter's block and the smallest block of any
group in the prefix cache. A chunk could then cross a 7168 boundary. The
block kept the state of an earlier chunk end but was cached under the
boundary's hash, so a later turn restoring it lost up to 1024 tokens of
recurrent state. Usually that went unnoticed; occasionally the turn was
gibberish from its first token, with draft acceptance at 0%.

It showed up in a long agent session: one bad turn in ~36. We reproduced it
deterministically by replaying the captured session turn by turn
([`tools/glm53/pcache_replay.py`](../../tools/glm53/pcache_replay.py)). The
same turn failed on every run and was clean without prefix-cache hits.
Patch 12 aligns the chunks to the KDA block, and with it the replay is
clean, including that turn. We also tried removing the KDA groups from
vLLM's EAGLE prefix-cache handling (the idea of the unmerged vllm#47861); it
did not help.

### 6. Speed again: the BF16 linears in FP8 (patch 13)

An nsys profile of one decode step (k=7, one stream) showed it bandwidth
bound everywhere, and 46% of it in BF16 GEMMs: the FP8 release keeps the
q/k/v/o of the 34 KDA layers (9.1 GB) and the lm_head in BF16, the NoPE
overlay dequantizes the MLA q_b/o projections to BF16 on load, and the
DFlash2 drafter is BF16 throughout. Patch 13 requantizes all of them after
load to block-128 FP8 (the checkpoint's own format) and keeps one plain copy.
Decode runs a Triton W8A16 kernel on it - BF16 activations, FP32
accumulation, as fast as FP8 Marlin, which would need a second, repacked
copy - and prefill dequantizes it to BF16 for the stock GEMM.

| | v0.3.0 | + patch 13 |
|---|---:|---:|
| decode, code / prose | 29-31 / 12.8-13.2 tok/s | **38.9 / 17.1** |
| model in memory | 99.6 GiB | **92.7 GiB** |
| KV cache | 7.8 GiB (111 blocks) | **12.8 GiB (182 blocks)** |
| HumanEval / HumanEval+, of 164 (mean of 5 runs / 3 runs) | 155.8 / 147.0 | 154.7 / 146.0 |
| KLD vs all-BF16 linears (code / en / pl) | - | 0.021 / 0.028 / 0.021 |
| needles in ~105K / ~190K of code (8 per prompt, 6 prompts) | 48/48 | 48/48 |
| hard needles, ~190K (24 near-duplicate names, 8 asked, 3 prompts) | 16/24 | 17/24 |
| prefill, ~190K-token prompt | 282-284 s | 289-298 s |

The KLD is 2-4x the harness's noise floor (0.005-0.009 for this comparison,
without the FP4 pool), perplexity moves by under 1% and in Polish goes down;
HumanEval differs by less than its boot-to-boot spread (~3 tasks). The KDA
gates (b, f_a, g_a), `kv_b_proj`, the DSA indexer and the router stay BF16:
community runs of this model found BF16 gates, indexer and activations help
long-context recall (local-inference-lab vllm#1027), and they are small.
`W8A16_RE=` switches the patch off.

### 7. Speed again: the empty pairs of the MoE launch (patch 14)

The expert GEMMs (the hand-written `moe_w2_mm` / `moe_w4_mm`) already run at
GB10's bandwidth ceiling (210-224 GB/s measured, ~230 for a plain read), so
the kernel body has nothing left to give. The launch around it did: a decode
forward launches each tier with `grid.y` = the moe_align table length (64 at
T = 8), and only that tier's live pairs do work; the others are CTAs that
load their descriptor and exit - ~0.5 us (w2) / ~0.8 us (w4) per empty pair
per launch, 4-5 ms per step. Patch 14 compacts each tier's live pairs to the
front of its table and launches a small fixed grid (16 / 12) of looped
kernels that walk the table (`kernels/cubins-sm120-loop`; the kernel body is
the shipped one instruction for instruction, see its README). It also fuses
the MoE glue around them: the token quant, row gather, scale copy and the FP4
tier's seen flags in one Triton kernel, the SwiGLU and the second quant in
another. Everything is bit-exact with the old path (GPU checks in
`tools/glm53/`); measured the same night against a fresh patch-13 boot:

| | patch 13 | + loop | + loop + glue fusion |
|---|---:|---:|---:|
| decode step, code / prose (ms) | 140.6 / 137.6 | 132.6-133.2 / 131.2-132.3 | **131.5 / 130.5** |
| decode, code (bench, tok/s) | 39.2 | 41.5-42.0 | **42.2** |
| prose, 6 long Polish texts (tok/s) | 14.2 | 15.0-15.2 | **15.8** |
| HumanEval / HumanEval+, of 164 | 154 / 146 | 154 / 144, 149 / 141 | 156 / 147 |

HumanEval moves with the boot, not with the patch (the kernels compute the
same bits; the FP4 pool's fill order changes the greedy text from boot to
boot, ~+-3 tasks). `LOOP_GRID= FUSE_GLUE=0` switches it off.

### 8. The draft depth per step (patch 15)

DFlash2 drafts 7 tokens and the target verifies all 8 positions, but on this
box a decode step is not flat in the number of verified tokens: more tokens
mean more distinct experts to stream. Measured with fixed k (same night):

| k | step, ms | code tok/s (acceptance) | prose tok/s (acceptance) |
|---:|---:|---:|---:|
| 3 | 103.1 | 34.5 (3.56) | **21.4** (2.19) |
| 5 | 117.3 | 40.4 (4.74) | 18.7 (2.18) |
| 7 | 132.8 | **42.7** (5.66) | 18.6 (2.44) |

so T(k) ~ 81 + 7.4 k ms: code wants k = 7 (acceptance keeps growing to the last
position), prose wants k = 3. Patch 15 picks per step: the scheduler keeps a
per-request EMA of the conditional acceptance p and takes the k in {3, 7} that
maximizes E[tokens](k) / T(k) (a cost-driven controller, ported from our MiMo
deployment). Verification still decides every token, so the output is the
model's own; only how many drafted tokens get checked changes.

| | fixed k = 7 | adaptive {3, 7} |
|---|---:|---:|
| code, bench (tok/s) | 41.5-42.7 | 41.6 |
| prose, bench (tok/s) | 18.5-18.6 | **21.3** |
| prose, 6 long Polish texts (tok/s) | 15.0-15.8 | **18.2** |
| HumanEval / HumanEval+, of 164 | 154-156 / 144-147 | 156 / 146 |

{3, 5, 7} with a faster EMA (decay 0.8) cost 3% on code: the estimate is
noisy, and around code's acceptance k = 5 and 7 are within 5% of each other,
so it kept stepping down. `ADAPTIVE=0` = fixed k.

## Why the quality holds

Short version: per token, only a small part of the computation runs through
2-bit weights. The most used part of that is repaired at runtime, the 2-bit
codes are chosen for the model's real inputs, and all of it was measured
against the configuration it replaced.

**1. Bare 2-bit alone is NOT enough - measured.** The relative RMS weight
error of the 2-bit experts is ~36%. On bare 2-bit, free-form Polish prose
broke in 4 of 6 samples: loops covering 33-94% of the text, invented words,
Cyrillic mixed in. Lowering the temperature only turned gibberish into loops
("krawaty, krawaty, ..." at temperature 0). The failure is in the model, not
in the sampling settings.

**2. Only the routed experts are 2-bit.** Attention, KDA, the shared expert,
the three dense layers, embeddings and the router keep the checkpoint's own
precision, so every token's path through the network is exact except for
the 8 routed experts it picks in each layer.

**3. The FP4 pool repairs the experts that are actually used.** 2 GiB of
FP4 slots (~170 experts, ~1.4% of all of them) hold the experts the router
picks most often, re-chosen continuously from the live routing. The
controlled experiment settled it. With the same settings and the pool
switched off, the degeneration came back (32% and 54% loops). With it, all
samples were clean (10 of 10 at 3 GiB). And on code: HumanEval 92.1% without
the pool, 95.7% with it. The pool costs speed - without it decode is 26%
faster - and we pay that on purpose.

**4. GPTQ picks the 2-bit codes for the real inputs.** Round-to-nearest
quantizes every weight on its own. GPTQ quantizes column by column and
pushes each rounding error onto the weights not yet quantized, weighted by
how the layer's real inputs correlate. The errors then cancel in the layer's
output instead of adding up. Same format, same scales, same kernels - only
the choice of code changes. On the reference box it cut the output error of
the experts' gate/up projection by 22% on held-out tokens, and with a
calibration on source code:

- perplexity fell in all three evaluation domains: code -0.7%, English
  -1.3%, Polish -3.2%, on a corpus disjoint from the calibration;
- HumanEval rose from 95.7% to **97.0%**;
- decode did not slow down (29.4 tok/s), and the long-code loop probe stayed
  at 0 of 16.

The calibration is the lever, not a detail. The same method calibrated on
Markdown documentation and Polish Wikipedia improved Polish by 3.7% and
**cost HumanEval 3.6 points**. GPTQ does exactly what it is calibrated for.

**5. The drafter cannot change the answer.** Every drafted token is verified
by the full model, and a draft that disagrees is rejected. Speculative
decoding changes how fast the model's own output arrives, not what it is.

**6. It is measured, not assumed.** The tools are in
[`tools/glm53/`](../../tools/glm53/):

- HumanEval, with execution in a network-less container;
- a KLD harness: teacher forcing on 147K tokens of code, English and Polish
  written after the model's release, compared against a measured noise
  floor;
- a Polish prose probe that catches loops and invented words;
- a long-code loop probe with a detector that tells real loops from code
  structure.

**7. Against the original.** The same HumanEval harness was run against
the original model on the z.ai API
([`tools/glm53/humaneval/zai_generate.py`](../../tools/glm53/humaneval/zai_generate.py)).
Everything was the same: prompts, code extraction, `reasoning_effort` low,
temperature 0. Both sets of answers were scored with the same sandboxed
tests, and again with the much stricter HumanEval+ (EvalPlus) tests:

| | HumanEval | HumanEval+ |
|---|---:|---:|
| original, z.ai API | 95.1% (156/164) | 89.6% (147/164) |
| this setup (GPTQ-code planes) | **97.0%** (159/164) | **93.9%** (154/164) |

On HumanEval+ the original failed 10 tasks we passed, and we failed 3 that
it passed. An exact McNemar test gives p = 0.09 (HumanEval: p = 0.45). The
honest reading is **no measurable loss against the original on coding
tasks**, not that a 2-bit model beats it. How z.ai serves the model
(precision, templates) is not published.

For scale: sayyidfareed
[reports](https://huggingface.co/sayyidfareed/GLM-5.3-Flash-Spark-Q2XL-MTP)
97.0% HumanEval and 93.3% on HumanEval+ Mini (a smaller subset of the plus
tests) for a ~2.8-bit GGUF build on one Spark, with a 32K context. Here it
is 2.0 bits per expert weight, a 200K context and vLLM's serving stack
(prefix cache, batching, tool and reasoning parsers).

**What we cannot claim.** The z.ai API returns no logprobs, so there is no
KLD against the original - only task results. Every KLD and perplexity
figure above is against our own previous configuration. Also, with a
dynamic pool the served model is not a fixed function: the same text
evaluated twice in one boot differs by KLD 0.010-0.019 (0.006-0.010 without
the pool).

## Measurements

All on the reference box. Speed is `tools/glm53/bench_mtp.py` (mean of three
code prompts, 600 tokens, temperature 0, `reasoning_effort` low, warm server,
2 repeats); boot-to-boot variance is ~8%.

### How we got here

| configuration | code tok/s | quality |
|---|---:|---|
| bare 2-bit, no speculation | 11.4 | Polish prose degenerates (4 of 6 samples) |
| bare 2-bit + MTP k=2 | 22.4 | same |
| + FP4 pool 2 GiB (V1 runner: no MTP, no CUDA graphs possible) | 7.4 | Polish clean |
| V2 runner port: FP4 3 GiB + MTP k=4 + CUDA graphs | 18.2 | 10 of 10 samples clean |
| + promotion hysteresis 1.25, 200K context | 19.6 | |
| DFlash2 k=7 instead of the MTP head | 29.3 | HumanEval 95.7% |
| GPTQ planes (code calibration), pool 2 GiB | 29.4 | HumanEval 97.0% |
| FP8 W8A16 for the BF16 linears (patch 13) | 38.9 | HumanEval / HE+ in noise, KLD 0.02-0.03 |
| MoE launch without empty pairs + glue fusion (patch 14) | 42.2 | bit-exact |
| **adaptive draft depth (patch 15)** | **41.6 (prose 21.3)** | **lossless (verification)** |

### Speed and context (current configuration)

| | |
|---|---|
| decode, code | ~42 tok/s, acceptance length 5.5 (first draft position accepted 92-94%, the seventh ~40%) |
| decode, code, `reasoning_effort` high | 24.6 tok/s at v0.3.0, not re-measured since (thinking is harder to draft) |
| decode, prose | 17.6-18.5 tok/s (bench prompt); 15.0-15.8 on six long Polish texts |
| prefill | 643 tok/s at 70K, 632 tok/s at 198K (flat) |
| one 198K-token prompt | TTFT 313 s, 86% of the KV pool, 0 preemptions |
| two concurrent streams | two short ones (up to ~28K) fine; two long ones serialise |
| KV cache | 6.2-7.5 GiB depending on the boot (200K needs 5.12 GiB) |

### Quality

HumanEval pass@1 (164 tasks, temperature 0, ~+-2 tasks of noise):

| configuration | `effort` low | notes |
|---|---:|---|
| RTN planes, FP4 pool | 95.7% | 97.0% at `effort` high |
| RTN planes, no pool | 92.1% | +26% decode |
| GPTQ planes, calibrated on docs + Polish Wikipedia | 92.1% | Polish perplexity -3.7% |
| **GPTQ planes, calibrated on code (production)** | **97.0%** | |

Against the original model (z.ai API, same harness). HumanEval+ is the
EvalPlus v0.1.10 test set, 120 s per task, with time-outs re-run at 600 s:

| | HumanEval | HumanEval+ |
|---|---:|---:|
| original, z.ai API | 95.1% | 89.6% |
| **GPTQ-code planes (production)** | **97.0%** | **93.9%** |

KLD harness, production (GPTQ-code) against the RTN planes. The noise floor
is the same RTN configuration evaluated twice.

| domain | noise floor KLD | GPTQ-code vs RTN, KLD | perplexity RTN -> GPTQ-code |
|---|---:|---:|---|
| code | 0.019 | 0.110 | 2.792 -> **2.773** (-0.7%) |
| English | 0.018 | 0.196 | 12.79 -> **12.63** (-1.3%) |
| Polish | 0.010 | 0.175 | 4.714 -> **4.565** (-3.2%) |

GPTQ moves the model 6-18 times further than the noise, and every domain
moves in the right direction; per chunk, code improved in 14 of 16. The
corpus: 16 chunks of code from 14 GitHub repositories created after
2026-08-16, 10 of arXiv abstracts (2026-08-20 ... 09-08) and 10 of Polish
Wikipedia, 4096 tokens each, none of it in the calibration.

Generation probes, production planes:
- **Loop probe:** 0 of 16 long-code samples degenerate (8 tasks x 2; the
  RTN planes also 0 of 16).
- **Polish prose probe:** 0 bad / 1 weak / 5 ok. Bare 2-bit scored 4 bad /
  0 weak / 2 ok.

## Hardware and disk

- A GB10 box with 121.6 GiB of unified memory, NVIDIA driver (tested
  580.159.03, CUDA 13.0), docker with the NVIDIA container toolkit, and
  `python3` on the host.
- The server takes the whole machine: stop other GPU workloads first.

| what | size | where |
|---|---:|---|
| FP8 checkpoint | 306 GiB | any disk. It is read on **every** boot (the non-expert weights), so its speed sets the boot time |
| planes cache (2-bit + FP4 sections) | 227 GB | any disk (sequential reads) |
| GPTQ planes (optional, recommended) | 48 GB | any disk |
| FP4 store | 146 GB | **internal NVMe** (random reads while serving) |
| GPTQ calibration dumps (temporary) | 85 GB | NVMe, deleted after the build |
| drafter | 2.3 GB | any disk |
| torch.compile / FlashInfer cache | ~4 GB | any disk |

A checkpoint on a USB SSD works, but raise the read-ahead first
(`read_ahead_kb` 16384): on our enclosure that took boots from ~45 to ~17 min.

## Setup, step by step

Every step below was run on the reference box. The whole sequence from an
empty disk has not been re-run in one go, so if a step behaves differently
for you, the per-step notes say what to look at.

### 0. Host preparation

```bash
git clone https://github.com/lrozewicz/vLLM-Moet-GB10.git
cd vLLM-Moet-GB10

export DATA_DIR=/path/to/a/big/disk        # checkpoint + planes caches (~590 GB)
sudo mkdir -p /var/lib/glm53-packs-nvme    # FP4 store, internal NVMe (150 GB + 85 GB during GPTQ)
sudo chown "$USER": /var/lib/glm53-packs-nvme
mkdir -p ~/glm53-run

# the API key clients will send as "Authorization: Bearer ..."
openssl rand -hex 24 > ~/glm53-run/api-key && chmod 600 ~/glm53-run/api-key
export VLLM_API_KEY=$(cat ~/glm53-run/api-key)
```

`boot.sh` reads every path from the environment (`DATA_DIR`, or `MODEL_DIR`
/ `CACHE_DIR` / `PACKS_DIR` / `NVME_DIR` separately), so export `DATA_DIR`
and `VLLM_API_KEY` in every shell you boot from.

### 1. Images (~1 h)

```bash
# the vendor base (vLLM 0.1.dev20051+g487ecf187), pinned by digest - see below
docker pull vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
./deploy/gb10-glm53/build_combined.sh    # -> vllm-moet-glm53:combined  (NoPE overlay + W2 engine)
./deploy/gb10-glm53/build_dflash2.sh     # -> vllm-moet-glm53:dflash2   (DFlash2 overlay + patches 8, 9)
```

Everything this depends on is pinned to exactly what the measurements were
made with:
- **The base image, by digest.** It is the arm64 variant of the multi-arch
  image the NoPE overlay itself pins. The tag
  `vllm/vllm-openai:glm53-flash-arm64-cu130` has since been re-pushed with a
  different image (2026-09-09).
- **The two overlays, by commit.** They are downloaded into gitignored
  directories.

`BASE=...`, `UPSTREAM_URL=...` and `UP=...` override the pins if you want to
try newer versions. Every anchor is checked while building, so an
incompatible version fails the build, not the boot 20 minutes later.

### 2. Checkpoint (306 GiB)

```bash
DEST=$DATA_DIR/glm-5.3-flash RATE=20M \
  setsid nohup ./deploy/gb10-glm53/download_glm53.sh </dev/null >/dev/null 2>&1 &
tail -f ~/glm53-run/dl-ckpt.log
```

One stream, rate-limited (`RATE`, default 6M - a consumer router in our LAN
collapsed under parallel HTTPS streams), and every shard is verified against
its sha256 from Hugging Face. Re-run to resume. The script also creates the
`-cache` and `-packs` directories next to the checkpoint that the compose
file expects. `hf download zai-org/GLM-5.3-Flash --local-dir
$DATA_DIR/glm-5.3-flash` works too; then create
`$DATA_DIR/glm-5.3-flash-cache` and `$DATA_DIR/glm-5.3-flash-packs` yourself.

### 3. Drafter (2.3 GB)

```bash
hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir ~/glm53-run/dflash2-draft
```

License: CC-BY-NC-ND-4.0 - research and personal use; for commercial use,
contact the authors.

### 4. Patched vLLM files

```bash
./deploy/gb10-glm53/patches/build_mounts.sh      # -> ~/glm53-run/patch
```

It copies six files out of the image, applies the patches, and prints md5
sums; compare them with the table in
[`patches/README.md`](../../deploy/gb10-glm53/patches/README.md). `boot.sh`
mounts the results read-only over the originals.

### 5. First boot: quantization (~35 min)

```bash
DELTA_GB=0 ./deploy/gb10-glm53/boot.sh 120
docker logs -f glm53-moet | grep -E "moe_w2|startup complete"
```

`DELTA_GB=0` (no FP4 pool) because the FP4 sections do not exist yet. You
should see `STREAM-BUILD armed`, then `layer N planes built` for every
expert layer, then `Application startup complete`. The 2-bit planes (~82 GB)
are now in `$DATA_DIR/glm-5.3-flash-packs/planes-cache-fp4`. The server
works at this point (bare 2-bit, lower quality); stop it for the next step:

```bash
docker rm -f glm53-moet
```

### 6. FP4 sections, offline (~40 min)

Builds the FP4 copy of every expert layer from the checkpoint, one layer at
a time, with the model NOT loaded. Building these during a boot, with the
other layers resident, runs out of memory.

```bash
CACHE=$(cd $DATA_DIR/glm-5.3-flash-packs && ls -d planes-cache-fp4/v2/*/tp1-rank0)
FP4="docker run --rm --gpus all --entrypoint python3 \
  -v $DATA_DIR/glm-5.3-flash:/model:ro -v $DATA_DIR/glm-5.3-flash-packs:/packs \
  -v $PWD/tools:/w:ro vllm-moet-glm53:dflash2 /w/build_fp4_sections.py --cache /packs/$CACHE"

$FP4 --layers 10 --dry-run          # control: must print ZGODNE (= identical) for the 4 2-bit sections
$FP4 --layers $(seq -s, 3 44)       # layers 3-44 (layer 45 is the MTP head, unused with DFlash2)
```

The script does not re-implement the quantizer: it imports the one in the
image and recomputes the 2-bit sections as well, byte-comparing them with the
cache before writing anything. Any mismatch aborts without writing.

### 7. Production boot

```bash
./deploy/gb10-glm53/boot.sh 60
```

The first boot with the FP4 tier also writes the 146 GB FP4 store to the
NVMe, so it takes longer; later boots take 11-19 min. `boot.sh` ends by
printing the KV numbers. 200K needs `Available KV cache memory` >= 5.12 GiB;
if the boot fails with `max seq len ... larger than the maximum number of
tokens that can be stored in KV cache`, use `MAX_LEN=180000` or a smaller
`DELTA_GB`.

Smoke test:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flash",
       "messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],
       "max_tokens":2000, "temperature":1.0, "top_p":0.95,
       "chat_template_kwargs":{"reasoning_effort":"low"}}'
```

At this point you have the RTN configuration (HumanEval 95.7%).

### 8. GPTQ planes calibrated on code (recommended, ~1.5 h)

**a. Calibration corpus** from local clones of open-source repositories, as
many languages as you use:

```bash
python3 tools/glm53/build_code_calib.py --out ~/glm53-run/calib_code.txt \
  --repo ~/src/llama.cpp --repo ~/src/keycloak --repo ~/src/ComfyUI   # ... more is better
```

Optionally add prose in your language with `--prose FILE`: 10% Polish kept
the Polish gains in ours.

**b. Record the MoE inputs on a live server.** Boot with the capture hook
(patch 10). It stays idle until the `ARM` file exists:

```bash
mkdir -p /var/lib/glm53-packs-nvme/xdump
XDUMP=1 ./deploy/gb10-glm53/boot.sh 60

KLD="docker run --rm --network host -e VLLM_API_KEY -v $PWD/tools/glm53:/t:ro \
  -v $HOME/glm53-run:/r --entrypoint python3 vllm-moet-glm53:dflash2 /t/kld_glm.py"
$KLD prep-server --corpus code /r/calib_code.txt 64 --spread --out /r/calib_code.json
touch /var/lib/glm53-packs-nvme/xdump/ARM
$KLD dump --chunks /r/calib_code.json --k 1 --out /r/calib.npz --tag calib   # ~12 min, ~85 GB of dumps
rm /var/lib/glm53-packs-nvme/xdump/ARM
```

**c. Build the planes** with the server stopped (~50 min):

```bash
docker rm -f glm53-moet
docker run --rm --gpus all --entrypoint python3 \
  -v $DATA_DIR/glm-5.3-flash:/model:ro -v $DATA_DIR/glm-5.3-flash-packs:/packs \
  -v /var/lib/glm53-packs-nvme/xdump:/x:ro -v $PWD/tools:/w:ro vllm-moet-glm53:dflash2 \
  /w/glm53/gptq_build.py --src /packs/planes-cache-fp4 --dst /packs/planes-cache-gptq-code --xdump /x
rm -rf /var/lib/glm53-packs-nvme/xdump
```

Each layer is self-checked: the RTN codes of one expert, packed by the
script, must reproduce the existing cache byte for byte, or it aborts. Only
the new 2-bit gate/up planes are written (48 GB); everything else is
symlinked to the RTN cache. The build prints the per-layer error reduction,
which was -14% to -42% on ours, -22% on average.

**d. Boot.** `boot.sh` picks `planes-cache-gptq-code` automatically when it
exists (`PLANES_DIR=planes-cache-fp4` goes back to RTN):

```bash
./deploy/gb10-glm53/boot.sh 60
```

### 9. Check what you got

```bash
python3 tools/glm53/bench_mtp.py --repeats 2 --warmup      # ~39 tok/s on the code prompts (~29 with W8A16_RE=)
```

HumanEval (~40 min): see [`tools/glm53/humaneval/`](../../tools/glm53/humaneval/)
- expect 96-97% at `effort` low. For comparing configurations properly, read
the rules in [`tools/glm53/README.md`](../../tools/glm53/README.md).

## Using the server

- OpenAI-compatible API on port 8000, model name `glm-5.3-flash`, Bearer
  key from `VLLM_API_KEY`.
- **The model always thinks**; `chat_template_kwargs.reasoning_effort`
  (`low` / `high` / `max`) only picks how deeply, and **the default is `max`**.
  Agents should pass `low` (or `high`) explicitly. The reasoning comes back
  in the message's `reasoning` field (not `reasoning_content`), separate
  from `content`.
- Tool calling is on (`--tool-call-parser glm47`, auto tool choice).
- Sampling: `temperature` 1.0, `top_p` 0.95. `min_p` is rejected with
  speculative decoding (HTTP 400).
- **Concurrency.** DFlash2 k=7 pins 32 KV pool blocks per request (one
  recurrent-state copy per draft position in 4 KDA groups), so one long
  stream fits and a second long one waits. For several long sessions at once,
  boot the MTP variant (`DFLASH=0`, MTP k=4, about 2/3 of the speed). It
  needs layer 45 in the planes cache - run step 5 with `DFLASH=0`, and step 6
  with layers 3-45.
- **Agent turns pay a fixed prefill tax.** The KDA state forces a 7168-token
  KV block, and the prefix cache recovers every full block except the last.
  So each turn re-computes 1-2 blocks, ~16 s, however little it adds.
  Batching tool calls into fewer turns helps; trimming the context by a few
  thousand tokens does not.
- **Images are off** in the default configuration (`MM_IMG=0`): the vision
  tower's profiling reserves a large activation budget. `MM_IMG=1` turns
  images on at the cost of KV cache; we have not measured vision quality.

## Settings reference

All are environment variables of `boot.sh`; the defaults are the production
values. `DRY_RUN=1` prints the resolved compose file without starting
anything.

| variable | default | meaning |
|---|---|---|
| `DATA_DIR` | `/mnt/adata/ai/models` | parent of the checkpoint, `-cache` and `-packs` directories |
| `MODEL_DIR` / `CACHE_DIR` / `PACKS_DIR` | under `DATA_DIR` | override individually |
| `NVME_DIR` | `/var/lib/glm53-packs-nvme` | FP4 store (and GPTQ dumps); internal NVMe |
| `RUN_DIR` / `PATCH_DIR` / `DRAFT_DIR` | `~/glm53-run`, `.../patch`, `.../dflash2-draft` | logs, mounted files, drafter |
| `PLANES_DIR` | `planes-cache-gptq-code` if present, else `planes-cache-fp4` | which planes cache to load |
| `DFLASH` | 1 | 1 = DFlash2 k=7 (image `:dflash2`); 0 = MTP k=4 (image `:combined`) |
| `DELTA_GB` | 2 | FP4 pool in GiB; 0 = bare 2-bit |
| `PROMO_HYST` | 1.25 | promotion hysteresis (0 = upstream default, off) |
| `MAX_LEN` | 204800 | context window |
| `UTIL` | 0.94 | `--gpu-memory-utilization` |
| `MNBT` | 2048 | prefill chunk (`--max-num-batched-tokens`) |
| `MM_IMG` / `MM_VID` | 0 / 0 | images / videos per prompt |
| `MAX_LOGPROBS` | 20 | 64 for the KLD harness |
| `W8A16_RE` | KDA q/k/v/o, lm_head, MLA, drafter | module-name regex of the BF16 linears patch 13 runs in FP8; empty = all BF16 |
| `KV_MEM` | 13743895347 (12.8 GiB) | fixed KV pool (`--kv-cache-memory`); empty = vLLM's profiling (75-126 blocks at random on GB10). With `W8A16_RE=` use 8375186227 |
| `XDUMP` | 0 | 1 = mount the GPTQ capture hook |
| `MEM_MIN` | 105.0 | GiB of `MemFree` required before starting |
| `GUARD_GIB` | 1.5 | hostguard kills the container below this much available memory |
| `MOUNT_PATCHES` | 1 | 0 = run the image's original files (A/B) |
| `PATCH_SCHED` | 1 | 0 = without patch 12 only (reproduces the prefix-cache bug; A/B) |
| `DROP_DOCKER_LAYERS` | 1 | also drop the page cache of the docker image layers |

`USE_V2`, `GATE`, `BF12` and `CG_EST` are there for experiments; their
defaults are the only values we recommend. Their reasons are in the compose
file's comments.

## Measured, and not recommended

- **No FP4 pool** (`DELTA_GB=0`): +26% decode and +34% KV, but HumanEval
  92.1% instead of 95.7%.
- **KDA state in bf16 or fp16** (`KDA_SSM`, patch 7): bf16 breaks code; fp16
  looks free in prefill but its error grows with every rounding in decode
  (KLD x2, perplexity +0.4-0.5%).
- **GPTQ calibrated on the wrong text:** documentation and Wikipedia made
  code worse. Calibrate on what you will run.
- **The confidence gate** (`GATE=1`): needs a ~12.8 GiB pool, which does not
  fit. **The MTP head in FP8**: does not fit either.
- **`UTIL=0.95`**: leaves the host ~0.6 GB, below the memory guard.
- **Drafter KV in fp8** (patch 9): 83 instead of 81 KV blocks - not worth it.
- **CPU pinning** to the big cores: no effect; decode is GPU-bound.
- **Restarting a boot that is only slow.** A slow disk makes loading slow,
  and a restart throws away everything read so far. `boot.sh` never restarts
  on its own.

## Limitations and open work

- **Newer vendor images are not validated.** The build pins the base image
  (config `sha256:d42649063a2b...`, created 2026-08-26) and the overlay
  commits. The vendor tag has moved since (re-pushed 2026-09-09). A newer
  base may well work - the patches anchor on code and fail loudly at build
  time - but we have not measured it.
- No FP8 reference measurement (see above), and the dynamic pool makes the
  served model slightly non-deterministic.
- vLLM's start-up profiling gives a different KV pool on every boot on
  GB10 (it counts page cache as used memory); `KV_MEM` fixes it.
- The ~16 s prefill tax per agent turn.
- Open: a static FP4 pool (deterministic serving), a hybrid KDA state (fp32
  working state, fp16 checkpoints: smaller block, cheaper agent turns, two
  long streams with DFlash2), GPTQ for the down projection, damping and
  calibration sweeps.

## Files

| path | what |
|---|---|
| `deploy/gb10-glm53/build_combined.sh`, `w2-tail.dockerfile` | image with the NoPE overlay (downloaded) and the W2 engine |
| `deploy/gb10-glm53/build_dflash2.sh`, `dflash2.dockerfile` | image with DFlash2 (overlay downloaded) and patches 8/9/9b |
| `deploy/gb10-glm53/docker-compose.quality-fp4.yml` | the serving compose file; every knob is commented with its experiment |
| `deploy/gb10-glm53/boot.sh` | the way to start it |
| `deploy/gb10-glm53/patches/` | mounted patches, `build_mounts.sh`, the patch list |
| `deploy/gb10-glm53/hostguard.py`, `drop_page_cache.py` | memory guard, page-cache release |
| `deploy/gb10-glm53/download_glm53.sh`, `run_download.sh` | verified, resumable checkpoint download |
| `tools/apply_w2_overlay.py`, `tools/build_fp4_sections.py` | engine installer, offline FP4 sections |
| `tools/glm53/` | measurement tools and the GPTQ builder |
| `docs/GLM-5.3-FLASH.md` | the bring-up log (historical) |
| `deploy/gb10-glm53/Dockerfile`, `docker-compose.glm-5.3-flash.yml`, `first_boot.sh` | bring-up artifacts (before the NoPE overlay), kept for the record |

## Credits

- [zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash) for the model.
- [kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)
  for the 2-bit engine, the kernels and the FP4 tier.
- [chriswritescode-dev/glm-5.3-flash-sm120](https://github.com/chriswritescode-dev/glm-5.3-flash-sm120)
  and [vllm#53963](https://github.com/vllm-project/vllm/issues/53963) for
  the NoPE sparse-MLA fix.
- [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
  for the DFlash2 overlay and GLM glue,
  [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) for the drafter.
- [darkdatter/gb10-repo](https://github.com/darkdatter/gb10-repo) for the
  HumanEval harness pattern and the drafter lessons.

## OpenCode configuration for coding

The optimal configuration of this model for programming, on the example of
[OpenCode](https://opencode.ai). Add the model to an OpenAI-compatible
provider in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "spark": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DGX Spark",
      "options": {
        "baseURL": "http://<your-spark>:8000/v1",
        "apiKey": "{env:VLLM_API_KEY}"
      },
      "models": {
        "glm-5.3-flash": { "...": "the block below" }
      }
    }
  }
}
```

The model block:

```json
"glm-5.3-flash": {
  "name": "GLM-5.3-Flash (MoE 2-bit, thinking low)",
  "attachment": true,
  "tool_call": true,
  "temperature": true,
  "reasoning": true,

  "interleaved": {
    "field": "reasoning"
  },

  "modalities": {
    "input": ["text", "image"],
    "output": ["text"]
  },

  "limit": {
    "context": 200000,
    "output": 65536
  },

  "options": {
    "chat_template_kwargs": {
      "reasoning_effort": "low",
      "clear_thinking": true
    },

    "thinking_token_budget": 1536,

    "temperature": 1.0,
    "top_p": 0.95
  }
}
```

**The most important parameter is `"thinking_token_budget": 1536`.** Without
it the model often ignores `"reasoning_effort": "low"` and falls into long
thinking. Our reasoning probe shows the same thing: at `low`, the thinking
over one mid-sized task reached ~750 tokens, and in one draw it used a whole
1200-token budget before writing any code. The budget caps the thinking
phase, so the answer always gets its turn.

The rest, briefly:
- `reasoning_effort: low` is the fast setting for agent work (29 vs 25 tok/s
  at `high`).
- `clear_thinking: true` keeps the reasoning of earlier turns out of the
  prompt, so a long session does not carry every past thought.
- `interleaved.field: "reasoning"` matches the field this server returns
  reasoning in.
- Sampling follows the model card (`temperature` 1.0, `top_p` 0.95).
- `limit.context` 200000 stays just under the server's 204,800-token window.
- `"image"` input needs `MM_IMG=1` on the server (off in the default
  configuration, see [Using the server](#using-the-server)).
