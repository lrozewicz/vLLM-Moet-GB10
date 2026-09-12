# Changelog

Releases are git tags. Each names the serving-image tags
(`vllm-moet-gb10:vNNN`) built along the way, since that is what the compose
file pulls. GLM-5.3-Flash images are built locally by the scripts in
`deploy/gb10-glm53/` (`vllm-moet-glm53:combined`, `vllm-moet-glm53:dflash2`).

Numbers are measured on the reference box: ASUS Ascent GX10 (GB10, 48 SM,
121.6 GiB unified, aarch64). DeepSeek-V4-Flash-0731 entries: DSpark k=2,
decode over 500-token runs. GLM-5.3-Flash entries: DFlash2 k=7,
`reasoning_effort` low, `tools/glm53/bench_mtp.py` (600-token runs,
temperature 0, mean of three code prompts).

The what-and-why of each change against upstream is in
[`docs/CHANGES-VS-UPSTREAM.md`](docs/CHANGES-VS-UPSTREAM.md).

## v0.3.0 - 2026-09-12

**GLM-5.3-Flash on a single GB10.** A 321B-parameter model whose FP8
checkpoint is 2.5x the machine's memory serves at ~29 tok/s on code with a
200K context and HumanEval pass@1 97.0%. Images `vllm-moet-glm53:combined`
and `vllm-moet-glm53:dflash2` (built locally). Guide, measurements and
setup: [`docs/models/glm-5.3-flash.md`](docs/models/glm-5.3-flash.md).

### Added

- **GLM-5.3-Flash serving configuration** (`deploy/gb10-glm53/`): 2-bit
  routed experts (W2 planes) + a 2 GiB runtime FP4 pool + DFlash2 k=7 +
  CUDA graphs on the V2 model runner, 200K context. Measured: code
  **29.4 tok/s** (acceptance length 5.5), prose 12.4-13.5, prefill ~640
  tok/s (a 198K-token prompt in 313 s, zero preemptions), HumanEval pass@1
  **97.0%** at `reasoning_effort` low, 0/16 degenerate samples on a
  long-code loop probe. Model load 99.6 GiB, KV 6.2-7.5 GiB.

- **GPTQ planes calibrated on code** (`tools/glm53/gptq_build.py`, capture
  hook `deploy/gb10-glm53/patches/patch_xdump.py`). Same 2-bit format, same
  scales, same kernels - only the choice of code per weight changes, made
  with one Hessian per layer from the real MoE inputs of a live server
  (64 x 4096 tokens of source code in many languages). w13 output error on
  held-out tokens -22.0% on average over 42 layers. Against the RTN planes,
  on an evaluation corpus disjoint from the calibration: perplexity code
  -0.7%, EN -1.3%, PL -3.2%; HumanEval 95.7 -> 97.0%; decode unchanged
  (29.4 tok/s); loop probe 0/16 (RTN 0/16). The calibration is the lever:
  the same method calibrated on Markdown docs and Polish Wikipedia improved
  Polish (-3.7%) and cost HumanEval 3.6 points.

- **KLD harness and quality tools** (`tools/glm53/`). `kld_glm.py` measures
  KL divergence, top-1 agreement and perplexity between two configurations
  by teacher forcing through prompt logprobs (compose knob `MAX_LOGPROBS`,
  default 20 = unchanged). First finding: with the dynamic FP4 pool the
  served model is not a fixed function - two dumps of the same text in one
  boot differ by KLD 0.010-0.019 (0.006-0.010 without the pool), the noise
  floor every quality comparison has to clear. Also `quality_pl.py`,
  `loop_probe.py` + `loop_inspect.py` (the prose repetition metric flags code
  structure, not loops: 6/16 false alarms on healthy code), `reasoning_probe.py`,
  `bench_mtp.py` and a HumanEval harness.

- **Reproducible GLM deployment.** `deploy/gb10-glm53/boot.sh` (the
  production configuration as defaults, API key only from the environment,
  `DRY_RUN=1`), `patches/build_mounts.sh` (builds the mounted files from the
  image) and `drop_page_cache.py`; the compose file's volume paths are now
  variables. Verified on the reference box: `build_mounts.sh` output is
  byte-identical to the production mounts, and the dry run reproduces the
  running production container's environment, command and mounts.
  The builds are pinned to what was measured: the vendor base by digest
  (the tag `glm53-flash-arm64-cu130` was re-pushed with a different image on
  2026-09-09) and the two downloaded overlays by commit.

- **DFlash2 speculative decoding for GLM-5.3-Flash on one GB10**
  (`deploy/gb10-glm53/build_dflash2.sh` -> image `vllm-moet-glm53:dflash2`).
  The model's MTP head is a full MoE layer, so at k=4 its four sequential
  draft passes were ~46% of the decode step; the 1B block-diffusion drafter
  (`incoai/GLM-5.3-Flash-DFlash2`, CC-BY-NC-ND) proposes a block of 8 in
  one pass. Measured on the reference box, temperature 0, `reasoning_effort
  low`: code **29.3 tok/s** vs 19.6 with MTP k=4 (+49%, acceptance length
  5.3-5.9), prose 12.4 vs 11.5, prefill unchanged (~640 tok/s), HumanEval
  pass@1 95.7% (97.0% with `reasoning_effort` high), a single 198K-token prompt served with zero preemptions.
  - The overlay is tonyd2wild's port of vLLM PR #52816 for the same day-0
    image; its GLM glue files carry no license, so the build script
    downloads them into a gitignored directory (same rule as the NoPE overlay).
  - `patches/patch_glm5_draft_small_block.py` (patch 8): at TP=1 the MLA
    page is 656 B/token = 16 x 41, so the overlay's exact-fit drafter KV
    mode can never apply; its standalone fallback would charge a 7168-token
    drafter page per layer to every pool block (-58% KV capacity). The patch
    gives the standalone drafter a 1024-token block (-16%) and strips the
    `page_size_padded` that `--kv-cache-dtype-skip-layers=sliding_window`
    stamps on the drafter spec (that flag is required: no SWA backend on
    sm_121 accepts `fp8_ds_mla`).
  - `tools/sim_glm5_draft_geometry.py`: runs the patched KV grouping on CPU
    inside the image with this box's geometry, `--padded` reproducing the
    skip-quant spec - one run instead of a 12-minute boot.
  - `patches/patch_skip_layers_kv_dtype.py` + `patch_flashinfer_builder_dtype.py`
    (patches 9/9b, knob `VLLM_KV_SKIP_LAYERS_DTYPE`): the drafter's KV can be
    fp8 e4m3 instead of bf16. Measured: it computes correctly (acceptance
    unchanged) but FlashInfer's fp8 path takes ~0.67 GiB of workspace, which
    is what the smaller page saves - 83 pool blocks instead of 81. Default
    stays `auto`.
  - Known cost: with `mamba_cache_mode=align` each of the 4 KDA groups holds
    `1 + k` pool blocks per request (one state per draft position), so k=7
    pins ~32 of 81 blocks per request. One 200K stream fits; two long streams
    serialise, two 8K streams already preempt. For concurrent serving keep
    the MTP image (`:combined`).

- **GLM-5.3-Flash bring-up** (321B / 18B active, multimodal, hybrid KDA +
  NoPE sparse MLA), first boot 2026-08-29. The pieces that made it fit and
  serve (the bring-up log is `docs/GLM-5.3-FLASH.md`):
  - `tools/apply_w2_overlay.py`: installs the 2-bit expert engine onto a vLLM
    tree that is not v0.24.0, by extracting the `moe_w2_*` modules **from the
    generated patch** (so the patch stays the single source of truth) and
    re-applying the hooks by anchor. Needed because vLLM's `glm5next`
    implementation exists only in the vendor `glm53-flash` images (0.29-dev).
    Verified: every vLLM symbol the W2 modules import still resolves in
    0.29-dev, and the four `Fp8MoEMethod` signatures are unchanged.
  - `deploy/gb10-glm53/`: image, compose and a rate-limited checkpoint
    downloader.
  - Two hazards found and closed statically, before any boot:
    **monolithic dispatch** - skipping `_setup_kernel` leaves `moe_kernel`
    None, so `is_monolithic` falls through to `experts_cls`, and some FP8
    backends advertise a monolithic class first; the W2 hook in `apply` would
    never run. W2 layers now force the modular path, and `apply_monolithic`
    raises instead of computing on replaced tensors.
    **Recurrent-state corruption** - the confidence gate replays a step's
    forward, which is idempotent on DS4's all-attention stack but not here:
    34 of 45 layers advance a recurrent state in place. The gate now snapshots
    the Mamba-spec caches and rewinds them before each replay, and refuses to
    fire if it cannot find them.
  - Established without hardware: ~86.3 GiB of weights (vs ~91.4 for
    DS4-0731), and no new SASS kernels - K=4096/2048 are both in the shipped
    `_a32` set. Details: `docs/GLM-5.3-FLASH.md`.

### Fixed

- **Tool calls emitted inside a `<think>` span were dropped** ([#1](https://github.com/lrozewicz/vLLM-Moet-GB10/issues/1)).
  0731 opens a thinking block spontaneously even with `enable_thinking: false`
  - v0.2.0 stopped the raw `</think>` tags from leaking, but not this: the
  router sent the whole span to `reasoning`, and the tool parser only ever
  sees `content`, so the call never executed and its raw DSML markup showed
  up in the client. Reported with three runs on two large coding projects,
  onset around 50K tokens of context. A tool-call opener now acts as an
  implicit `</think>`. Regression test: `tests/reasoning/test_think_block_router.py`
  (8 text-level cases, no tokenizer or network; 5 fail before the fix).

- **DFlash aux hidden states were a no-op flatten.** The DFlash branch fed
  drafts `hidden_states.flatten(1)`, but between decoder layers
  `hidden_states` is the single-stream `x` returned by
  `mhc_fused_post_pre_tilelang` - 2-D `[T, hidden]`, so the flatten did
  nothing and the drafter received `n_aux x hidden` (5 x 4096 = 20480)
  features where a DFlash fc wants `n_aux x hc_mult x hidden` (81920). The
  streams only exist as `aux_recon` `[T, hc_mult, hidden]`; flatten that
  instead. Both assembly paths concatenate the aux list as-is
  (`gpu_model_runner.py`, `spec_decode/dflash/speculator.py`) and
  `llm_base_proposer` asserts the combined width, so on images `v026`-`v031`
  a DFlash boot against this target could not run at all.

### Changed

- **GLM-5.3-Flash: FP4 pool 2 GiB instead of 3.** At 200K the KV requirement
  is 5.12 GiB, and with a 3 GiB pool the available KV varied 4.49-5.83 GiB
  between boots of the same configuration - production failed to boot one
  morning. 2 GiB gives 6.2-7.5 GiB. Pool sizes between 2 and 5 GiB showed no
  measurable quality difference; 0 GiB costs 3.6 points of HumanEval.
- README restructured for several models; the DeepSeek-V4-Flash-0731 guide
  moved to `docs/models/deepseek-v4-flash-0731.md`.
- **Gate operating point settled: tau=0.60 / promote=64 stays the default.** Both
  endpoints of the 2026-08-05 sweep ran the full quality suite. No significant
  difference on HumanEval (93.9 vs 92.1, flips 4/1, McNemar p=0.375), HumanEval+
  (90.2 vs 87.8, 6/2, p=0.289), MBPP (89.7 vs 89.7, 4/4, p=1.000), MBPP+ (73.5 vs
  73.0, 10/8, p=0.815), BFCL lite (89.3 vs 88.3) or needle (100 vs 100); GPQA
  Diamond is incomplete for the pair. Throughput re-measured 2026-08-09 across
  three workloads, because the gate's cost is a property of the traffic: sampled
  prose pays 18% in Polish (12.8 vs 15.6 tok/s, fires 11.8% of steps) and 26% in
  English (12.9 vs 17.5, fires 13.9%), but **on code at temperature 0 the safe
  point costs 4%** (21.7 vs 22.6, fires 2.3% of steps vs 0.2%) - greedy code
  decoding is confident, so the safety net rarely triggers wherever tau sits.
  (At 0.60/64 both prose languages collapse to ~12.9 tok/s: once the gate fires
  on one step in seven, it is the bottleneck.) Agentic coding is what this box serves, so
  we keep the protective point: a few percent does not buy a risk that short
  single-turn suites cannot resolve, and long sessions and hard multi-step
  programming - where uncertainty compounds - are unmeasured. The knobs stay
  live-tunable. Rationale: `docs/quality.md`.
- The DFlash acceptance figure is now reported per workload. Measured on
  `-0731` at k=7 with the fix above: **11.8%** on code at greedy (position 1
  **45.7%**, 1.82 accepted tokens/step) and **2.7%** on prose at temperature
  1.0. The previously published "~3%" was the prose number alone and
  understates the published checkpoint by roughly 4x; the conclusion is
  unchanged (it does not transfer to `-0731`; base target: 41.6% / 78.8% /
  3.9), but the starting point for a fine-tune is better than recorded.

### Measured and rejected (GLM-5.3-Flash)

- No FP4 pool (`DELTA_GB=0`): +26% decode (32.8 tok/s) and +34% KV, but
  HumanEval 92.1% (-3.6 points, 6 new failures). The pool protects code.
- KDA recurrent state in fp16 (`VLLM_KDA_SSM_DTYPE`, patch 7): smaller KV
  block and +35% KV, but the error grows with the number of roundings -
  KLD x2 and perplexity +0.4-0.5% in decode-like chunking. bf16 breaks code.
- Drafter KV in fp8 (patches 9/9b): correct, but FlashInfer's fp8 workspace
  eats the saving (83 vs 81 KV blocks).
- Pinning the engine to the Cortex-X925 cores: within noise (the decode step
  is GPU-bound).

### Known limitations (GLM-5.3-Flash)

- DFlash2 k=7 with `mamba_cache_mode=align` pins 32 KV pool blocks per
  request: one long stream (200K) fits, two long streams serialise. For
  concurrent long sessions use the MTP image (`boot.sh DFLASH=0`).
- The KDA state forces a 7168-token attention block, so an agent pays a
  roughly constant ~16 s prefill per turn however little it adds.
- The drafter `incoai/GLM-5.3-Flash-DFlash2` is CC-BY-NC-ND-4.0
  (non-commercial).

## v0.2.1 - 2026-08-08

Image `v031`.

### Added

- The NVFP4 KV write extension now JIT-compiles for the **local** device
  capability instead of hardcoded sm_120a (the `a` suffix is device-specific,
  not family-compatible, so the stock build could not even load on sm_121).
  `VLLM_NVFP4_DS_MLA_ARCHS` (e.g. `"120a;121a"`) prebuilds a fatbin on a
  machine without a GPU - the Dockerfile uses it.
- Dockerfile: new step 3b applies the FlashInfer read-side patch
  (`tools/nvfp4_flashinfer_sm120/patch_flashinfer.py`) - earlier images never
  ran it, so the packed read path was silently absent; step 3c prebuilds the
  write extension for sm_120a+sm_121a into `VLLM_NVFP4_DS_MLA_EXT_DIR`.

### Fixed

- The nvfp4 KV config validator rejected every real DeepseekV4 checkpoint on
  every GPU: it read `kv_lora_rank`, which V4 configs do not define (the
  cached latent width is `head_dim`). It now falls back to `head_dim` for
  `model_type=deepseek_v4`. Nobody had hit this before because the sm_120a
  capability gate above it fired first.

### Known limitation

- `--kv-cache-dtype=nvfp4` does **not** cover DeepseekV4 models. The packed
  nvfp4 path (write kernel + FlashInfer read patch) serves the generic
  `FLASHINFER_MLA_SPARSE_SM120` backend and its 512/64 cache line (V3.2
  family, 656 -> 352 B/token). V4 model classes use their own 448/64
  `fp8_ds_mla` layout (584 B/token) and have no nvfp4 variant; adding one
  would be new kernel work for a capacity-only win. The reference compose
  stays on `--kv-cache-dtype=fp8`; caveats are documented in the Dockerfile
  header and the 0731 compose.

## v0.2.0 - 2026-08-06

Images `v028` (decode), `v029` and `v030` (reasoning output).

### Performance

- Decode **9.8 -> 14.3 tok/s** at tau=0.60/promote=64, quality semantics
  unchanged. The FP4 confidence gate had been firing on ~38% of prose steps
  at temperature 1.0, because the V2 gate hook evaluated ALL verify rows -
  including rows the sampler discards after the first rejected draft. Each
  fire costs up to 768 MiB of synchronous NVMe promotion reads and up to 3
  full re-forwards. Porting the S4 spec-mask to the hook brings the fire rate
  to ~1-2%.
- Prefill **~500 -> ~753 tok/s** on a 23.7K prompt (chunk 2048 -> 4096 plus
  `VLLM_MOE_W2_FUSED_UNPERMUTE=1`).
- Further operating points, measured: promote=16 gives ~15.8 tok/s and
  tau=0.30/promote=16 ~15.9 tok/s. Both change what the gate lets through, so
  they want a quality re-validation before use; ~16.5 tok/s is the ceiling
  with the gate quiet.

### Added

- `VLLM_MOE_W2_GATE_SPEC_MASK=1` now works under speculative decoding on the
  V2 runner (the mask itself is upstream's; the V2 hook never passed it the
  spec metadata).
- `ThinkBlockRouterParser` for `--reasoning-parser=deepseek_v3`: routes the
  inside of **every** `<think>...</think>` span to `reasoning` and strips the
  tags, in both thinking and non-thinking mode. State-driven over decoded
  text, so multi-token speculative deltas cannot split a tag.
- `route_late_content`, an opt-in hook parsers may implement, applied by
  `DelegatingParser` to content produced after the reasoning phase closes.
- `VLLM_DS_V3_LEGACY_REASONING=1` restores the stock reasoning behavior.
- Gate telemetry and live-tunable knobs in the reference compose:
  `VLLM_MOE_W2_GATE_TRACE`, `VLLM_MOE_W2_KPI_EVERY`, `VLLM_PREFILL_TIMERS`,
  and file-backed `GATE_TAU` / `GATE_MAX_PROMOTE` for sweeping a warm server
  without restarts.

### Fixed

- `<think>` blocks reopened mid-answer leaked raw tags into `content`, so
  agent clients showed a stray `</think>` in the text. Two stacked causes:
  the stock parser handles a single block at the start of the output, and
  `DelegatingParser` latches `reasoning_ended` one-way - with thinking
  disabled that latch closes before the first generated token, so the
  reasoning parser never ran at all. Verified on the parser pipeline without
  model sampling: stock delegation leaks in 4 of 6 scenarios, this release in
  0 of 6, with tag-free prose and literal `<` in text untouched.
- `draft_sample_method=probabilistic` crashed at load with the native DSpark
  head (`AttributeError: draft_id_to_target_id`, which full-vocab draft
  classes do not define at all).

### Measured and rejected

- `draft_sample_method=probabilistic`: acceptance 44% vs 52% for greedy on
  this target, decode flat. Greedy stays the default; the fix above only
  makes the knob fail honestly instead of killing the boot.

### Documentation

- `CHANGELOG.md` and `docs/CHANGES-VS-UPSTREAM.md` split out of the README,
  which now covers running the thing rather than its history.

## v0.1.0 - 2026-08-01

First working release: 304B MoE served on one GB10 box, ~9.8 tok/s decode
with DSpark k=2, 256K context. Images `v024` through `v027`.

### Added

- sm_121 family unlock for the sm_120 SASS cubins (80/80 load, compute
  bit-identical) and stream-build for the mxfp4 path, so the ~150 GB expert
  checkpoint is no longer staged in host RAM - which on unified memory cannot
  fit at all. Image `v024`; folded from build-time fixups into the patch
  itself in `v027`.
- Model Runner V2 hooks for the W2 path (gate promote -> re-forward fixed
  point, per-step tier lifecycle), which enables the native DSpark drafter:
  **+27% decode** (~7.7-8 -> ~9.8 tok/s). Image `v025`.
- DFlash-on-MLA-target support: draft KV-cache dtype override, raw flattened
  aux hidden states, layer-scoped attention guard, un-widened draft hidden
  buffers, non-MLA speculator layers routed around the MLA layer-tuple
  builder. Boots, drafts and serves. Image `v026`.
- A reference deployment (Dockerfile, annotated compose, memory guard,
  preflight checks) that turns upstream's own knobs into a config that boots
  here: the FP4 tier store on NVMe instead of pinned RAM, the persistent
  planes cache (~35 min first boot, ~10 min after), and the boot-guard
  consent valves - each with the measurement that justifies it.

### Known limitation

- The published RedHatAI DFlash speculator is trained against **base**
  DeepSeek-V4-Flash; against `-0731` acceptance collapses from 41.6% to ~3%.
  The runtime is ready for a 0731-trained speculator.

### Measured and rejected

- DSpark k=7 (the model card's suggestion): 5.1 tok/s. Positions 4-7 accept
  ~0% and seven sequential drafter passes eat the 273 GB/s of bandwidth.
- `deepseek_mtp`: incompatible with the 0731 head architecture
  (`main_norm`/`main_proj`, 3-block `mtp.{0,1,2}` stack); fails at weight
  load.
