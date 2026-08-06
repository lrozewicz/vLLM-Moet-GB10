# vLLM-Moet-GB10

**Run DeepSeek-V4-Flash-0731 (304B MoE: 284B main model + 20B built-in DSpark speculator head) on a single NVIDIA GB10 box** - NVIDIA
DGX Spark, ASUS Ascent GX10, and other GB10 systems with 121.6 GiB of unified
memory. Measured: **~14.3 tok/s decode** with speculative decoding, **256K
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

Upstream targets discrete-GPU boxes; this fork makes the same engine work on
**GB10 unified memory**, where several of its entirely reasonable assumptions
break. What changed and why: [`docs/CHANGES-VS-UPSTREAM.md`](docs/CHANGES-VS-UPSTREAM.md).
Per-release history: [`CHANGELOG.md`](CHANGELOG.md).

## Measured results (ASUS Ascent GX10, GB10, 121.6 GiB unified, aarch64)

| metric | value |
|---|---|
| decode, DSpark k=2 | **~14.3 tok/s** at tau=0.60/promote=64; ~15.8 with promote=16, ~16.5 ceiling with the gate quiet |
| decode, code at temperature 0 | drafter acceptance 90-97%, accepted throughput ~15 tok/s |
| decode, no speculation | ~7.7-8 tok/s |
| prefill | **~753 tok/s** on a 23.7K prompt (chunk 4096 + fused unpermute) |
| context | 256K served (KV pool ~660K tokens; model is natively 1M - the ceiling is prefill time, not memory) |
| first boot / later boots | ~35 min / **~10 min** (LOADER-SKIP from the planes cache: zero staging, zero requant) |
| memory at serve | ~117/121.6 GiB used, PSI ~ 0 |

Anti-recommendations (measured, resist the urge): **DSpark k=7** (the model
card's suggestion) gives 5.1 tok/s, because positions 4-7 accept ~0% and
seven sequential drafter passes eat the 273 GB/s of bandwidth - k=2 is the
optimum here. **`deepseek_mtp`** is incompatible with the 0731 head
architecture and fails at weight load.

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
  cleanly separated from `content`. This fork routes **every** `<think>`
  block there, including ones the model reopens mid-answer, so agent clients
  never see a raw `</think>` in the text.
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
2. **`Shmem` is GPU allocation.** `cudaMalloc` is backed by shmem; after
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
target - it boots, drafts and serves. However, the published speculator was
trained against **base** `deepseek-ai/DeepSeek-V4-Flash`, whose weights differ
from `-0731`, and draft acceptance collapses from 41.6% to ~3%. The moment a
0731-trained DFlash speculator exists, this stack should hit ~2x the current
decode. If you have the training capacity - the runtime is ready and we will
gladly test.

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
- `CHANGELOG.md`, `docs/CHANGES-VS-UPSTREAM.md` - per-release history and the
  full delta against upstream, with the measurements behind each item.
- `docs/UPSTREAM-README.md` - the original upstream README (worth reading:
  the architecture and the quality methodology are described there).

## License

Apache-2.0, same as upstream. See `LICENSE` and `NOTICE`.
