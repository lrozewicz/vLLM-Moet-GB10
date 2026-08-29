# GLM-5.3-Flash on a single GB10 — feasibility, budget, and the port

**Status (2026-08-29): SERVING.** GLM-5.3-Flash answers on a single GX10.

```
INFO: Application startup complete.
```

Measured decode on code (temperature 0, n=3):

| | no MTP | **MTP k=2** |
|---|---:|---:|
| decode | 11.4 tok/s | **22.4 tok/s** (21.8 / 22.8 / 22.7) |
| mean acceptance length | — | **2.81-2.95** |
| context served | 128K | 64K |
| free memory at serve | ~7 GiB | ~3 GiB |

Acceptance of 2.9 against a theoretical ceiling of 3.0 (two drafts plus the
base token) means the drafter is right nearly every step. That answers the
open question from bring-up: **a 2-bit drafter does not cost acceptance here.**
Putting layer 45 under W2 (`VLLM_MOE_W2_NUM_LAYERS=46`) was necessary rather
than merely tidy - that layer is a full 288-expert MoE layer, 6.75 GiB at FP8,
against ~7 GiB of headroom.

Context drops to 64K with MTP on: the drafter takes the slack, and 128K then
wants 1.7 GiB of KV against 1.18 GiB available. KV is cheap on this model (12
of 45 layers keep a growing cache), so this is a straight trade against the
drafter, not a wall.

For scale, the same box serves DeepSeek-V4-Flash-0731 at ~21 tok/s on code.
GLM-5.3-Flash is newer and larger (321B vs 304B) and matches it - before the
FP4 tier and confidence gate, which are still off here.

It took two independent fixes, and only one of them is ours:

| blocker | fix | whose |
|---|---|---|
| 306 GiB will not fit in 121.6 GiB | 2-bit planes -> ~96 GiB | this fork (four FP8-path gaps) |
| no attention backend for NoPE on sm_12x | zero-pad the latent 512 -> 576 | [chriswritescode-dev/glm-5.3-flash-sm120](https://github.com/chriswritescode-dev/glm-5.3-flash-sm120), out of [vllm#53963](https://github.com/vllm-project/vllm/issues/53963) |

The second one deserves the credit for unblocking this. Rather than write a
kernel for `d=512`, it reuses the existing `d=576` GLM_NSA one and zero-pads:
a zero RoPE block contributes nothing to the QK dot, softmax is unchanged, and
the value comes from the 512-wide NoPE region — exact, and not one line of
CUDA. It costs 656 B/token instead of ~528 (~24% more DSA KV over 11 layers).

Their overlay was verified on 4x RTX PRO 6000 (384 GB) against the 306 GiB FP8
checkpoint. What this fork adds is the half that makes it fit on **one** box.
Build both with `deploy/gb10-glm53/build_combined.sh`, which fetches their
Dockerfile at build time — their repo carries no LICENSE, so we must not
redistribute it.

The sections below were written during bring-up and are kept because the
negative results still cost ~10 minutes of loading each to reproduce.

## The short answer

It fits, and by a wider margin than DeepSeek-V4-Flash-0731 did.

| | DS4-Flash-0731 (shipping) | GLM-5.3-Flash (this port) |
|---|---:|---:|
| weights at serve | ~91.4 GiB | **~86.3 GiB** (computed) |
| layers with a growing KV cache | 43 of 43 | **12 of 45** |
| new SASS kernels needed | — | **none** |

The two things that make it work are the same two that made DS4 work — 2-bit
tensor-symmetric expert planes, and an FP4 delta tier with a confidence gate to
buy the quality back — plus one property of this model that DS4 did not have:
**two thirds of its layers are KDA linear attention with a constant-size
recurrent state**, so long context costs almost nothing in memory.

## Established: what the model actually is

From `config.json` and the safetensors headers (shapes read directly from the
shard headers over range requests, not inferred):

- 45 language layers + **1 MTP draft layer**, hidden size 4096.
- **Hybrid attention**: 34 KDA linear-attention layers (constant state) and 11
  NoPE sparse-MLA layers (`kv_lora_rank` 512, `qk_rope_head_dim` 0, DSA indexer
  at `index_topk` 2048), plus 1 more MLA layer inside the MTP head.
- **MoE**: 288 routed experts, top-8, `moe_intermediate_size` 2048, 1 shared
  expert; layers 0-2 dense (`first_k_dense_replace` 3).
- Manifold-Constrained Hyper-Connections (`mhc`, `hc_mult` 4) — small tensors
  (`hc_*`, ~70 MiB total), no effect on the budget.
- A 24-block vision tower (~1.1 GB BF16) and a 154880-token vocabulary.
- Weights are **native FP8** with 128x128 block scales, except the KDA
  projections, `kv_b_proj`, the indexer, embeddings and the vision tower, which
  are BF16. Total 328.3 GB on disk.

## Established: the memory budget

The routed experts are 97% of the checkpoint:

```
43 layers x 288 experts x 3 matrices x 2048 x 4096 = 311.6e9 parameters
```

At FP8 that is ~306 GiB — unloadable on a 121.6 GiB box at any batch size. At
the 2-bit planes it is **1.688 GiB per layer**, which is (coincidentally) the
same per-layer figure the DS4 loader prints on this machine.

| component | size |
|---|---:|
| routed experts, 42 main-stack layers, 2-bit planes | 70.9 GiB |
| KDA projections, 34 layers, BF16 | 8.7 GiB |
| embeddings + lm_head, BF16 | 2.4 GiB |
| sparse-MLA + indexer, 12 layers | 1.7 GiB |
| shared experts, 43 layers, FP8 | 1.0 GiB |
| vision tower, BF16 | 1.1 GiB |
| dense MLP, 3 layers, FP8 | 0.4 GiB |
| routers, norms, hyper-connections | 0.2 GiB |
| **total, no speculation** | **~86.3 GiB** |

That leaves roughly **35 GiB** for the KV pool, activations, CUDA graphs and
the FP4 delta pool on a 121.6 GiB machine.

KV is cheap here in a way it is not on DS4: only the 12 MLA layers keep a
cache that grows with context (512 elements/token/layer, NoPE so no rope half),
and the 34 KDA layers keep a fixed recurrent state of about 71 MiB per
sequence. Long context on this model is bounded by **prefill time**, not by
memory.

### Where to cut if it turns out tight

In the order they should be considered, with what each buys:

1. **The vision tower — 1.1 GiB.** Not needed for coding, and the user asked
   about exactly this. Cheapest cut available and fully reversible.
2. **KDA projections to FP8 — ~4.4 GiB.** The single largest non-expert item
   is BF16 today. This is real quantization work with a quality question
   attached (linear attention states accumulate), so it is a later resort, not
   a first one.
3. **Embeddings/lm_head to FP8 — 1.2 GiB.**

None of these are needed if the computed budget holds; they exist so that the
answer to "we are 3 GiB short" is a decision rather than a dead end.

## Established: no new kernels are needed

`moe_w2_mm` cubins are per-K. This model needs K=4096 (gate/up: `[2048, 4096]`)
and K=2048 (down: `[4096, 2048]`). Both are in the shipped `_a32` live set,
for every variant the serving path selects (`mc4`, `mc4afrag`) and for the FP4
delta GEMM (`moe_w4_mm`). The Dockerfile asserts their presence at build time
rather than discovering it at weight load.

This is the difference between a port and a research project: GLM-5.2 needed
K=6144 and Kimi-K2.7 needed K=7168, and each of those was new SASS work.

## Established: how the engine attaches

The W2 engine hooks the **quantization layer**, not the model. `Fp8MoEMethod`
gets four hooks — host-stage the experts at `create_weights`, build 2-bit
planes in `process_weights_after_loading`, route the forward through
`moe_w2_mm` in `apply`, and refuse EPLB — and the model is never touched.
GLM-5.3-Flash builds its experts through the stock `FusedMoEFactory`, so it
lands on exactly that path.

Two details that would otherwise have needed model-specific work already
handle this checkpoint:

- `is_w2_layer` matches `.layers.<N>.` generically, so
  `model.language_model.layers.N...` works with no change.
- `_layer_cutoff` reads `num_hidden_layers` through `get_text_config()`, which
  unwraps composite VLM configs — necessary here, since this is a
  `...ForConditionalGeneration` with the language model under `text_config`.
  Its failure mode is a *silent* fallback to 43 (the DS4 layout), which on a
  45-layer model would send layers 43-44 down the stock FP8 path: ~13.5 GiB
  more memory and a quietly mixed-precision stack. The compose file therefore
  pins `VLLM_MOE_W2_NUM_LAYERS=45` rather than trusting the lookup.

### The plane builder needs no changes

`build_layer_planes_fp8` is shape-generic: it reads `E, N13, K13` and
`_, N2, K2` straight off the staged tensors and calls `_require_kernels(K13,
K2)`. For this checkpoint that is E=288, N13=4096 (gate|up fused), K13=4096,
N2=4096, K2=2048 — both K values in the shipped cubin set, and every plane
size divisible by the 4- and 32-element packing the planes use. The block
scales are the `(128, 128)` the builder already assumes, and vLLM fuses the
checkpoint's separate `[16, 32]` gate/up scales into the `[32, 32]` it wants.

### The monolithic-dispatch trap (found and closed statically)

Skipping `_setup_kernel` leaves `self.moe_kernel` as `None`, and in 0.29-dev
`is_monolithic` falls back to `experts_cls.is_monolithic()` in exactly that
case. Several FP8 backends advertise a monolithic class *first* —
`FLASHINFER_TRTLLM` returns `[TrtLlmFp8ExpertsMonolithic,
TrtLlmFp8ExpertsModular]` — and a monolithic dispatch calls `apply_monolithic`,
which routes internally and reads `w13_weight`/`w2_weight`: tensors the plane
builder has already replaced. The W2 hook in `apply` would simply never run.

Whether GB10 actually lands on that backend depends on
`is_supported_config` at run time, so the overlay does not rely on the answer:
a W2 layer sets `_moe_w2_force_modular` at `create_weights`, the
`is_monolithic` override returns `False` for it, and `apply_monolithic` carries
a guard that raises rather than computing on replaced tensors. The W2 GEMM
needs the modular form regardless — it consumes `topk_weights`/`topk_ids` from
the router.

This class of bug does not exist in the v0.24.0 patch's world, which is a fair
warning about the rest of the port: the *hooks* transfer literally, but the
dispatch around them did move.

## The port: why not the v0.24.0 patch

`patch/vllm-moet-v0.24.0.patch` cannot be used. GLM-5.3-Flash needs vLLM's
`glm5next` implementation (KDA, NoPE sparse MLA, mHC, the MTP head, the vision
tower), which does not exist in 0.24.0 and — as of 2026-08-28 — is not in the
public vLLM repository at all: it ships only inside the
`vllm/vllm-openai:glm53-flash*` images (vLLM 0.29-dev). The vendor recipe says
as much ("use docker before the integration is included in the public repo").

So the layering is inverted: **take the vendor image that already knows the
model, and add only the 2-bit expert engine on top.**
`tools/apply_w2_overlay.py` does that, and it extracts the W2 modules *from the
patch* rather than carrying a second copy — the patch stays the single source
of truth for W2 code (AGENTS.md), and a regeneration propagates for free.

### What transferred, and what was verified

| piece | how it transferred |
|---|---|
| 9 `moe_w2_*` modules (~7.7k lines) | copied verbatim; **every vLLM symbol they import was checked against 0.29-dev and still resolves** |
| `Fp8MoEMethod` hooks | literal — the four signatures are unchanged between 0.24.0 and 0.29-dev |
| `envs.py` extension-variable handling | re-anchored (the validator collapsed to one line upstream) |
| runner tier lifecycle + confidence gate | **rewritten for the single-GB10 profile**, see below |
| SM120 cubins | copied; K=4096/2048 asserted present at build |

Deliberately **not** ported: the DSpark/DFlash speculators (DeepSeek heads),
the v0.24.0 SM120 base fixes (upstream by 0.29), the sparse-MLA backends and
the NVFP4 KV cache (DS4-family layouts; this model gets its attention stack
from the base image).

### The runner hooks are a rewrite, not a copy

Upstream's runner hooks carry three concerns this box does not have: a
host-resident base cache with miss-replay, pipeline parallelism, and
tensor-parallel rank consensus. GB10 is **one device** — TP=PP=1 — so every
`all_reduce` in those paths collapses to the local decision, and the PP
full-pipeline `gate_reforward` is dead code.

Carrying a mis-ported collective is worse than not carrying it: it desyncs or
hangs, and a partial replay silently corrupts output. So the overlay installs
the single-rank form and **refuses multi-rank serving with an explicit error**
rather than approximating it. What is installed:

- drain pending residency commits before each forward (`wait_manager_idle`),
- the confidence gate's promote → re-forward **fixed point** (one replay is
  only first-order: experts upgraded early re-route later layers onto still
  cold ones),
- `step_end` on **every** step, not only when the gate fired — upstream
  measured that skipping it accumulates pins until the whole pool is
  unpromotable and the first decode tokens corrupt,
- the boot-time pool-floor guard, which fails a pool too small to hold a
  step's working set instead of serving zeroed expert contributions.

## The hybrid trap: why the gate needed new work

This is the one place where GLM-5.3-Flash is *harder* than DS4, and it is
architectural rather than incidental.

The confidence gate works by **replaying a step's forward** with the routed
experts promoted to FP4. On DS4 that is safe because every layer is attention:
the replay rewrites the same KV slots with the same values, so it is
idempotent — upstream states this explicitly and unit-tests it.

GLM-5.3-Flash has **34 KDA linear-attention layers out of 45**. In the decode
path they receive the state cache itself as `initial_state` and the kernel
advances it **in place**:

```python
core_attn_out_non_spec, _ = fused_recurrent_kda(
    ...,
    initial_state=recurrent_state,      # the cache, not a copy
    ssm_state_indices=non_spec_state_indices_tensor,
)
```

Replaying that step advances the recurrent state a second time. Nothing
crashes; the sequence simply drifts, and the damage is cumulative and silent —
the worst possible failure mode, and one the DS4 stack has no defence against
because it never needed one.

The overlay handles it by making the replay state-neutral: it collects the
Mamba-spec cache tensors at KV-cache init (attention KV is tens of GiB and is
never cloned), snapshots them when the gate fires, and rewinds them before each
re-forward, so every pass starts from the step's entry state. If the handles
were never discovered on a model that has recurrent layers, the gate refuses to
fire rather than corrupt the sequence.

Cost: ~214 MiB of copies per fire at `max-num-seqs 2` (34 layers x a few MiB),
against fire rates DS4 measured at 2-4% of steps on code and 12-14% on sampled
prose — roughly 0.1% of decode time. Correctness here is cheap; the reason it
is worth writing up is that it is invisible until you look for it.

## Staging

1. **2-bit base, no delta/gate/MTP.** Needs no runner hooks at all — the
   overlay's `--no-runner` mode is exactly this. Verifies loading, the plane
   builder, the K=4096/2048 kernels and basic generation.
2. **+ FP4 delta tier and confidence gate.** Quality. Upstream measured bare
   2-bit degenerating on this class of model, so this is not optional for
   serving, only for first light.
3. **+ MTP.** The largest decode win available (~2x on DS4) and a real memory
   decision — see below.

<a name="mtp"></a>
### MTP: the drafter is not free

Layer 45 is a **full MoE layer with its own 288 experts** — 6.75 GiB at FP8.
`is_w2_layer` excludes it deliberately: upstream's acceptance numbers were
measured with the drafter unquantized. On a 121.6 GiB box that choice costs
5 GiB of headroom, and the alternative (`VLLM_MOE_W2_NUM_LAYERS=46`, drafting
at 2 bits) trades acceptance for memory in a way nobody has measured on this
architecture. Both arms are cheap to test once the model boots; neither should
be guessed at now.

Note also the DS4 lesson, which is about bandwidth and not about GLM: the
vendor recipe suggests `num_speculative_tokens: 5`, but on GB10 the optimum for
DS4 was **k=2** — deeper drafts accepted ~0% at the tail while costing a full
drafter pass each, against 273 GB/s of memory bandwidth. Sweep k here rather
than adopting the recipe's number.

## What the first boot actually taught us (2026-08-29)

The checkpoint finished downloading (306 GiB, 62/62 shards sha256-verified)
and the boot ran automatically. It did not serve. Two findings, in order of
how much they cost:

### 1. The FP8 path had no streaming build (fixed)

Upstream arms the streaming per-layer build for **modelopt-NVFP4** and
**mxfp4** checkpoints. `Fp8MoEMethod` has no such arming: it stages the entire
expert checkpoint in host RAM and requantizes afterwards. On a discrete GPU
that is wasteful; here it is impossible - ~306 GiB against 121.6 GiB.

DS4-Flash never exposed this: its checkpoint is FP4 and takes the mxfp4 path,
which IS armed. **The FP8 path had simply never run on a unified-memory box.**

Fixed in `tools/apply_w2_overlay.py`: `arm_stream_build` learns the FP8
variant (checked *before* the NVFP4 one - an FP8 layer has no `scale_2` and
would otherwise fall through to the wrong builder), and `plan_pack_skip` stops
assuming the `*_weight_scale` suffix. Confirmed on hardware: `STREAM-BUILD
armed`, **42 of 42 layers armed, none skipped**.

### 2. The loader does not walk the shards in order (open)

With streaming armed, memory still reached 121 GiB with **one** layer built.
Instrumenting the loader showed why:

```
DBG armed: 42    NOT armed: 0
DBG BUILD start key=29 layer=...layers.32...      <- no matching "BUILD done"
DBG pend layers.32 {w13_weight: 86, w13_weight_scale_inv: 0, ...}
```

Layer **32** completed while the progress counter read **14%**. That is the
whole problem: this checkpoint is layer-major, and the index proves it - at
most **two** layers span any shard boundary, so peak staging should be about
**14 GiB**:

```
maximum layers with an open shard range: 2 (at shard 4): [10, 11]
```

The Run:ai streamer reads them in some other order, so every layer it touches
early materializes a ~7.2 GiB staging buffer and then waits for the rest of
its tensors. Forty-two of those do not fit anywhere.

Next step is the plain safetensors loader: the usual argument against it (mmap
never releases its mappings) does not survive the streaming build, because
each layer's staging is freed the moment it is requantized, and what mmap
leaves behind is *page cache* - reclaimable - rather than anonymous
allocations, which are not. If shard-ordered reading is not enough, the
fallback is to back the staging buffers with a file on NVMe
(`torch.from_file`), which makes the peak independent of load order entirely.

### The operational lesson: cap the container

Two runaway loads wedged the whole box - the host ran out of memory, `sshd`
could no longer fork, and a power cycle was the only way back. `memguard`
cannot save this: it does not get scheduled either. The compose file now sets
`mem_limit: 100g`, so the kernel kills the container and the machine stays
reachable. It bounds host-side allocations (the ones that actually run away);
`cudaMalloc` is shmem-backed and not fully accounted, so it is a blast shield,
not a budget.

## Where it stands: the model loads, vLLM cannot attend to it on sm_121

**The W2 port works.** With the four FP8-path fixes and the planes cache, a
boot now loads all 43 layers, serves 39+ of them from cache, allocates KV and
reaches CUDA-graph capture. Peak memory behaves as computed (~96 GiB of
planes, ~10 GiB of allocator reserve, freed per layer).

**What blocks serving is upstream, not this fork.** GLM-5.3-Flash is NoPE
sparse MLA (`qk_rope_head_dim: 0` with an indexer), and on consumer Blackwell
vLLM 0.29-dev has no attention backend that can serve that shape. All three
candidates were tried on hardware:

| backend | result |
|---|---|
| `FLASHINFER_MLA_SPARSE_SM120` (auto-selected) | *requires* the packed `fp8_ds_mla` layout, whose writer asserts `pe_dim == 64` — this model has 0 |
| `TRITON_MLA` (the other candidate for capability 12) | `Reason: ['sparse not supported']` |
| `FLASHINFER_MLA_SPARSE` | `Reason: ['compute capability not supported']` |

The KV dtype cannot route around it either: `auto` and `fp8` are both
rewritten to `fp8_ds_mla` for that backend, and `bfloat16` leaves no valid
backend at all for `head_size=512`. FlashInfer 0.6.18 — the vendor recipe's
advice for sparse-MLA init trouble — changes nothing, because the assert
lives in vLLM's own `cache_kernels.cu`, not in FlashInfer.

vLLM's own source says as much. `platforms/cuda.py` returns
`[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]` for capability 12, while the
branch immediately below it describes **exactly this model** — "NoPE sparse
MLA (GLM-5-Next shape: qk_rope_head_dim=0 with an indexer)" — and routes it to
FlashInfer's SM90 FA3 path. Consumer Blackwell is simply not wired for it yet.

### What would unblock it

1. **Upstream support** — the cheapest path by far: a NoPE-capable sparse-MLA
   backend for capability 12, or letting the SM120 backend drive a
   non-`fp8_ds_mla` layout. Worth filing against vLLM with the three errors
   above; the fix is theirs, and the recipe already claims Blackwell support.
2. **Teach `fp8_ds_mla` a zero-width rope half** — patch
   `concat_and_cache_mla` and the matching read path. Real CUDA work, and it
   would have to be redone against every vLLM bump.
3. **SGLang** — the model card lists it first, and this box already runs
   SGLang nightlies for Qwen3.8. Whether its sparse-MLA path is NoPE-capable
   on sm_121 is unknown and worth one experiment before writing CUDA.

## OPEN questions (only a boot can answer)

1. **Load order**: does a shard-ordered loader keep the in-flight set at the
   two layers the file layout implies? See above.
2. **Does the recurrent-state rewind actually hold?** The mechanism is
   implemented and described above, but only a long generation on hardware can
   confirm it. The failure signature is specific and worth knowing in advance:
   output that starts coherent and drifts or degenerates as the reply grows,
   appearing only with `VLLM_MOE_W2_GATE=1`.
3. Prefill throughput, and therefore the usable context window.
4. Whether the FP8 backend oracle picks a monolithic class here at all - the
   override makes it moot, but the log line says which backend was chosen and
   is worth reading once.

## Reproducing

```bash
# 1. the vendor base image (one stream, rate-limited - this LAN's router
#    collapses on parallel pulls; `docker pull` has no rate limit at all)
REPO=vllm/vllm-openai TAG=glm53-flash-arm64-cu130 ~/glm53-run/pull-image.sh
tar -cC ~/glm53-run/oci . | docker load

# 2. the serving image
docker build -f deploy/gb10-glm53/Dockerfile -t vllm-moet-glm53:latest .

# 3. the checkpoint (~306 GiB, ~15 h at the safe rate)
setsid nohup deploy/gb10-glm53/download_glm53.sh </dev/null >/dev/null 2>&1 &

# 4. serve
docker compose -f deploy/gb10-glm53/docker-compose.glm-5.3-flash.yml up -d
./deploy/gb10/memguard.sh glm53-moet 8 60 20     # arm for the first boot
```

## Roads not taken

- **SSD streaming of experts** (`sglang-ssd-stream`). It streams Qwen3.8's
  47.7 GiB *lookup table*, where each token needs 16 rows and their addresses
  are known one block ahead — a sparse, prefetchable access pattern. MoE
  experts are neither: 8 of 288 experts across 43 layers is ~620 MiB of
  *distinct* weights per token even at 2 bits, i.e. ~12 GB/s at 20 tok/s
  against an NVMe that delivers 3-6. The technique is excellent and does not
  apply here.
- **vLLM's `async_llm_streaming` example.** That is streaming generated tokens
  out of `AsyncLLM`, not streaming weights in; it has no bearing on capacity.
- **Serving an NVFP4 checkpoint directly.** The experts alone are ~146 GiB at
  NVFP4 — still over budget — and using one as the *source* for the 2-bit
  planes is worse than the FP8 original on both counts: double quantization,
  and a larger non-expert footprint (those repacks leave attention, dense MLP
  and shared experts in BF16, ~+4 GiB).
