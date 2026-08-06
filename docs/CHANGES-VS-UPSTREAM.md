# What this fork changes

Everything here is a delta against
[kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet).
Upstream targets discrete-GPU boxes (RTX PRO 6000, 5090, multi-GPU rigs);
this fork makes the same engine work on the **GB10 unified-memory** platform,
where several of upstream's entirely reasonable assumptions break. Each item
below is the assumption, what it costs on GB10, and how it was verified.

Per-release ordering of these changes is in [`../CHANGELOG.md`](../CHANGELOG.md).
Every knob with the measurement behind it lives in
[`../deploy/gb10/docker-compose.deepseek-v4-flash-0731.yml`](../deploy/gb10/docker-compose.deepseek-v4-flash-0731.yml).

## Platform enablement

**1. sm_121 family unlock for the SASS cubins.** The expert GEMM kernels are
hand-written sm_120 SASS and upstream gates them on an exact match. GB10 is
sm_121, same family. Verified rather than assumed: 80/80 cubins load and
compute is **bit-identical** (`test_split_equiv` max_abs_diff = 0).

**2. Stream-build for the mxfp4 path** (DeepSeek-V4-Flash checkpoints).
Upstream stream-builds only modelopt-NVFP4 checkpoints; the mxfp4 path staged
the whole ~150 GB expert checkpoint in host RAM first. On unified memory host
RAM *is* VRAM, so the staging cannot fit at all - the engine died mid-load
with no `moe_w2` log line, looking like an OOM of unknown origin.

**3. FP4 tier host store on NVMe** (configuration plus the diagnosis behind
it - the store itself is upstream's). With the delta tier on, the tier stages
~3.0 GiB/layer of **pinned** host RAM during load (~129 GiB total). Invisible
next to a discrete GPU, fatal here: boot wedged around layer 15-18, and
because pinned pages neither swap nor drop, unkillably. Setting
`VLLM_MOE_W2_STORE_DIR` bounds it to reclaimable page cache and load growth
drops from ~4.7 to 1.69 GiB/layer. Finding that this - and not a leak - was
the cause took most of the bring-up.

**4. Boot-guard consent valves for unified memory** (documentation, not
code). `torch.cuda.mem_get_info` counts page cache as "used" on GB10, so the
resident-fit guard refuses configs that serve fine (measured: guard saw
"free 49.4 GiB" with 117 GiB MemAvailable). `VLLM_MOE_W2_FORCE_RESIDENT=1` is
the code's own consent valve and is measured-safe here; a real shortfall
still fails as a plain torch OOM.

## Speculative decoding

**5. Model Runner V2 hooks for the W2 path** (gate promote -> re-forward
fixed point, per-step tier lifecycle). The native DSpark drafter of
DS4-Flash-0731 is V2-runner-only, and upstream's W2 path rejected V2
wholesale. This port is what makes speculative decoding available on 0731
at all (+27% decode when it landed).

**6. S4 spec-mask ported to the V2 gate hook**
(`VLLM_MOE_W2_GATE_SPEC_MASK` now works under spec decode on V2). The V2 port
evaluated ALL verify rows, so the FP4 confidence gate fired on ~38% of prose
steps at temperature 1.0 - on uncertainty in rows the sampler *discards*
after the first rejected draft. Each fire costs up to 768 MiB of synchronous
NVMe promotion reads plus up to 3 full re-forwards. With the mask the quality
semantics are unchanged (masked rows never reach the sampler) and decode goes
**7.9 -> 14.3 tok/s** at tau=0.60/promote=64.

**7. DFlash-on-MLA-target support** (draft KV dtype, aux hidden-state layout,
layer-scoped attention guard, hidden buffer widths, KV-cache grouping). Lets
heterogeneous GQA drafts run against MLA targets. Mechanically complete - it
boots, drafts and serves - but blocked in practice on a 0731-trained
speculator; see the DFlash section in the main README.

**8. `draft_sample_method=probabilistic` load-crash fix.** The native DSpark
head does not define `draft_id_to_target_id` at all, and the reduced-vocab
probe read it unconditionally, killing the boot. One-line `getattr`. Measured
while there: probabilistic acceptance 44% vs 52% greedy on this target, so
greedy stays the default - the fix matters so the knob fails honestly instead
of crashing.

## Serving correctness

**9. `<think>` blocks stop leaking into content.** Two stacked limitations in
stock vLLM, both visible as a raw `</think>` in agent clients:

- the `deepseek_v3` reasoning parser delegates to a parser that handles at
  most one reasoning block at the start of the output (R1) or none at all
  (Identity, used whenever thinking is disabled);
- `DelegatingParser` latches `reasoning_ended` **one-way**. With thinking
  disabled the latch closes at prompt inspection, before the first generated
  token, so the reasoning parser never runs; in thinking mode it closes at
  the first `</think>`. After that the parser is never consulted again.

0731 is agent-tuned and at temperature 1.0 reopens `<think>` mid-answer, so
raw tags reached clients in **both** modes. This fork replaces the delegation
with `ThinkBlockRouterParser` (routes every block, wherever it appears, to
`reasoning`) and adds an opt-in `route_late_content` hook that
`DelegatingParser` applies to post-latch content. Parsers that do not
implement the hook are untouched, so other models see no behavior change.
`VLLM_DS_V3_LEGACY_REASONING=1` restores stock behavior.

Verified deterministically on the parser pipeline (no model sampling): stock
delegation leaks in 4 of 6 scenarios, this fork in 0 of 6, with tag-free
prose and literal `<` in text unaffected.

## Offered back

All of it is offered upstream as patches. Reports written against the
upstream projects, with diagnosis and proposed fixes, are collected in the
project notes (`notes/02-zgloszenia-upstream/` in the working directory):
mxfp4 stream-build, unified-memory load growth, the FP4 pinned-store root
cause, the DSpark probabilistic crash - plus one for
[vllm-project/vllm](https://github.com/vllm-project/vllm) on the reasoning
latch, which affects any agent-tuned thinking model, not just this one.
