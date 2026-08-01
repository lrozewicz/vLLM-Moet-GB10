# vLLM-Moet-GB10

**Run DeepSeek-V4-Flash-0731 (304B MoE: 284B main model + 20B built-in DSpark speculator head) on a single NVIDIA GB10 box** - NVIDIA
DGX Spark, ASUS Ascent GX10, and other GB10 systems with 121.6 GiB of unified
memory. Measured: **~9.8 tok/s decode** with speculative decoding, **256K
context**, quality-recovering FP4 tier, tool calling and split reasoning for
agent use. First boot ~35 min (one-time quantization); every boot after that
~10 min.

## Credit where credit is due

This project is a hardware adaptation of
**[kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)**,
and the heavy lifting is all upstream: the 2-bit tensor-symmetric expert
planes with hand-written SASS GEMM kernels, the FP4 delta tier with a
confidence gate that recovers quality where 2-bit loses it, the persistent
pack/planes caches, and an unusually rigorous, measurement-driven engineering
culture (every knob in that codebase carries the experiment that justified
it). None of what is in this repo would exist without that work - thank you.
If the upstream project accepts contributions again, everything here is
offered back as patches.

What upstream targets is discrete-GPU boxes (RTX PRO 6000, 5090, multi-GPU
rigs). This fork makes the same engine work on the **GB10 unified-memory**
platform, where several of upstream's (entirely reasonable) assumptions break
in interesting ways - see below.

## What this fork changes

| # | Change | Why GB10 needs it |
|---|---|---|
| 1 | **sm_121 family unlock** for the SASS cubins | The kernels are hand-written sm_120 SASS. Family compatibility verified on sm_121: 80/80 cubins load, compute is **bit-identical** (`test_split_equiv` max_abs_diff = 0). |
| 2 | **Stream-build for the mxfp4 path** (DeepSeek-V4-Flash checkpoints) | Upstream stream-builds only modelopt-NVFP4 checkpoints; the mxfp4 path staged the whole ~150 GB expert checkpoint in host RAM first. On unified memory host RAM *is* VRAM - the staging cannot fit at all. |
| 3 | **FP4 tier host store on NVMe** (config, `VLLM_MOE_W2_STORE_DIR`) | With the delta tier on, the tier stages ~3.0 GiB/layer of **pinned** host RAM during load (~129 GiB total) - invisible next to a discrete GPU, fatal on unified memory. The pack-file store bounds it to reclaimable page cache. |
| 4 | **Model Runner V2 hooks for the W2 path** (gate promote→re-forward fixed point, per-step tier lifecycle) | The native DSpark drafter of DS4-Flash-0731 is V2-runner-only, and upstream's W2 path rejected V2 wholesale. This port enables **speculative decoding** on 0731 (+27% decode). |
| 5 | **DFlash-on-MLA-target support** (6 fixes: draft KV dtype, aux hidden-state layout, KV-cache grouping, …) | Lets heterogeneous GQA drafts (e.g. RedHatAI's DFlash speculators) run against MLA targets. Mechanically complete; blocked in practice on a 0731-trained speculator (see below). |
| 6 | **Boot-guard consent valves documented for unified memory** | `torch.cuda.mem_get_info` counts page cache as "used" on GB10, so the resident-fit guard refuses configs that serve fine; `VLLM_MOE_W2_FORCE_RESIDENT=1` is measured-safe here. |

Full deployment lore - every knob with the measurement behind it - lives in
[`deploy/gb10/docker-compose.deepseek-v4-flash-0731.yml`](deploy/gb10/docker-compose.deepseek-v4-flash-0731.yml).

## Measured results (ASUS Ascent GX10, GB10, 121.6 GiB unified, aarch64)

| metric | value |
|---|---|
| decode, DSpark k=2 | **~9.8 tok/s** (500-token runs; acceptance 52%, mean accept len 2.04) |
| decode, no speculation | ~7.7–8 tok/s |
| prefill | ~500 tok/s at chunk 2048 (28K-token prompt → TTFT ~53 s) |
| context | 256K served (KV pool ~660K tokens; model is natively 1M - the ceiling is prefill time, not memory) |
| first boot | ~35 min (43-layer streaming requant + pack writes) |
| later boots | **~10 min** (LOADER-SKIP from the planes cache: zero staging, zero requant) |
| memory at serve | ~117/121.6 GiB used, PSI ≈ 0 |

Anti-recommendations (measured, resist the urge):
- **DSpark k=7** (the model card's suggestion): 5.1 tok/s - positions 4–7
  accept ~0% and seven sequential drafter passes eat the 273 GB/s bandwidth.
  k=2 is the optimum on this hardware.
- **`deepseek_mtp` method**: incompatible with 0731's head architecture
  (`main_norm`/`main_proj`, 3-block `mtp.{0,1,2}` stack) - fails at weight load.

## Quick start

Prerequisites: a GB10 box with the NVIDIA driver (tested: 580.159.03, CUDA
13.0), docker + NVIDIA container toolkit, ~160 GB for the checkpoint, ~210 GB
for the planes cache (any disk), ~135 GB on the **internal NVMe** for the FP4
pack, and network access for the one-time downloads.

```bash
git clone https://github.com/lrozewicz/vLLM-Moet-GB10.git
cd vLLM-Moet-GB10

# 1. Build the serving image (~40 min first time; DeepGEMM compiles from source)
docker build -f deploy/gb10/Dockerfile -t vllm-moet-gb10:latest .

# 2. Download the stock checkpoint (~156 GB). Keep max workers low - high
#    parallelism can DoS consumer routers (measured; see script comments).
bash deploy/gb10/download_ds4_0731.sh /path/to/models/deepseek-v4-flash-0731

# 3. Edit the volume paths at the top of the compose file, then:
docker compose -f deploy/gb10/docker-compose.deepseek-v4-flash-0731.yml up -d

# 4. (recommended for the first boot) arm the memory guard in a second shell -
#    it stops the container on sustained memory pressure instead of letting a
#    livelock force a hard reset:
./deploy/gb10/memguard.sh deepseek-moet 8 60 20

# 5. Watch progress:
docker logs -f deepseek-moet | grep -E "moe_w2|Completed|startup complete"
```

First boot: you should see `moe_w2 STREAM-BUILD armed`, then
`moe_w2: layer N planes built (1.69 GiB)` climbing to 43/43, then CUDA graph
capture, then `Application startup complete`. Later boots print
`moe_w2 LOADER-SKIP armed` and `layer N planes from cache` instead - that is
the persistent cache doing its job.

Smoke test:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer change-me" -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash-0731",
       "messages":[{"role":"user","content":"Hello!"}],
       "max_tokens":100, "temperature":1.0, "top_p":0.95,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

## Serving notes (agents, thinking, tools)

- Two model aliases are served: `deepseek-v4-flash-0731` (chat) and
  `deepseek-v4-flash-0731-think`. Thinking is a **per-request** toggle via
  `chat_template_kwargs: {"enable_thinking": true|false}`; with
  `--reasoning-parser=deepseek_v3` the chain of thought comes back in the
  message's `reasoning` field (note: `reasoning`, not `reasoning_content`),
  cleanly separated from `content`.
- Tool calling works out of the box (`--enable-auto-tool-choice
  --tool-call-parser=deepseek_v4` - the DSML grammar of this checkpoint).
- Sampling per the model card: `temperature=1.0`, `top_p=0.95` for agentic
  use, `top_p=1.0` otherwise.
- `reasoning_effort`: only `"max"` changes anything (injects the
  maximum-deliberation preamble, thinking mode only). The official encoding's
  default is `low` = empty prefix - you are already there.

## Unified-memory survival guide

Things that are true on GB10 and surprising coming from discrete GPUs:

1. **`MemAvailable` is not a danger signal.** It can read 0 while the machine
   is healthy and healthy while the machine is livelocked. Watch
   `/proc/pressure/memory` (`full` line) and log progress instead.
2. **`Shmem` ≈ GPU allocation.** `cudaMalloc` is backed by shmem; after
   stopping the container `Shmem` drops to ~0. `drop_caches` frees nothing of
   it, swap does not touch it.
3. **Pinned host memory is a silent killer.** Anything a discrete-GPU design
   parks in "cheap host RAM" (staging buffers, host-resident tiers) competes
   with your VRAM 1:1 here.
4. **`torch.cuda.mem_get_info` lies during I/O.** Page cache from concurrent
   checkpoint reads counts as "used", so boot-time budget guards misfire -
   hence the documented consent valves in the compose file.
5. **A wedged box may still answer SSH's banner but never log you in.** Keep
   `memguard.sh` armed during first boots; it kills the container at sustained
   PSI before the point of no return.

## DFlash status (help wanted)

[RedHatAI/DeepSeek-V4-Flash-speculator.dflash](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-speculator.dflash)
is a 2B parallel-drafting speculator with excellent numbers on its target.
This fork contains the complete runtime support for running it against an MLA
target (six fixes, from draft KV-cache dtype to KV-cache grouping) - it boots,
drafts and serves. However, the published speculator was trained against
**base** `deepseek-ai/DeepSeek-V4-Flash`, whose weights differ from `-0731`
(different shard hashes), and draft acceptance collapses from 41.6% to ~3%.
The moment a 0731-trained DFlash speculator exists, this stack should hit
~2× the current decode. If you have the training capacity - the runtime is
ready and we will gladly test.

## Roadmap

- More big-MoE models on single-GB10.
- A 0731-trained DFlash speculator (see above).

## Repository layout

- `patch/vllm-moet-v0.24.0.patch` - the whole engine as a patch onto the
  official `vllm/vllm-openai:v0.24.0` arm64 image (upstream's distribution
  model, preserved; regenerated with `tools/check_patch_files.py`).
- `deploy/gb10/` - Dockerfile, the annotated compose file, memory guard,
  preflight checks, checkpoint downloader.
- `kernels/cubins-sm120/` - upstream's hand-written SASS kernels (run
  bit-identically on sm_121).
- `docs/UPSTREAM-README.md` - the original upstream README (worth reading:
  the architecture and the quality methodology are described there).

## License

Apache-2.0, same as upstream. See `LICENSE` and `NOTICE`.
