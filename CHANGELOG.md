# Changelog

Releases are git tags. Each names the serving-image tags
(`vllm-moet-gb10:vNNN`) built along the way, since that is what the compose
file pulls.

Numbers are measured on the reference box: ASUS Ascent GX10 (GB10, 48 SM,
121.6 GiB unified, aarch64), DeepSeek-V4-Flash-0731, DSpark k=2, decode over
500-token runs.

The what-and-why of each change against upstream is in
[`docs/CHANGES-VS-UPSTREAM.md`](docs/CHANGES-VS-UPSTREAM.md).

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
