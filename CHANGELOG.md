# Changelog

Releases are git tags. Each names the serving-image tags
(`vllm-moet-gb10:vNNN`) built along the way, since that is what the compose
file pulls.

Numbers are measured on the reference box: ASUS Ascent GX10 (GB10, 48 SM,
121.6 GiB unified, aarch64), DeepSeek-V4-Flash-0731, DSpark k=2, decode over
500-token runs.

The what-and-why of each change against upstream is in
[`docs/CHANGES-VS-UPSTREAM.md`](docs/CHANGES-VS-UPSTREAM.md).

## Unreleased

### Added

- **GLM-5.3-Flash bring-up** (321B / 18B active, multimodal, hybrid KDA +
  NoPE sparse MLA). Not yet booted - no serving numbers here until it is.
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
