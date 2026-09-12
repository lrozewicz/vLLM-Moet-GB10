# DeepSeek-V4-Flash-0731 on one GB10

**Run DeepSeek-V4-Flash-0731 (304B MoE: 284B main model + 20B built-in DSpark
speculator head) on a single NVIDIA GB10 box.** Measured: **~21 tok/s decode
on code**, ~13 on sampled prose (the gap is how often the speculator guesses
right), **256K context**, quality-recovering FP4 tier, tool calling and split
reasoning for agent use. First boot ~35 min (one-time quantization); every
boot after that ~10 min.

This was the first model of this fork (releases v0.1.0-v0.2.1). What changed
against upstream and why: [`../CHANGES-VS-UPSTREAM.md`](../CHANGES-VS-UPSTREAM.md);
quality methodology and results: [`../quality.md`](../quality.md).

## Measured results (ASUS Ascent GX10, GB10, 121.6 GiB unified, aarch64)

| metric | value |
|---|---|
| decode, code at temperature 0 (DSpark k=2) | **~21 tok/s** (n=6, 20.5-22.9); drafter acceptance 81%, 2.62 accepted tokens/step; the confidence gate fires on 2-4% of steps |
| decode, sampled prose at temperature 1.0 | **~12.9 tok/s** (PL 12.8, EN 12.9); the gate fires on 12-14% of steps - its worst case, and what the tuning grid below was swept on |
| decode, gate operating point | shipped default is tau=0.60/promote=64, [deliberately](../quality.md#gate-operating-point-why-we-stay-at-tau060--promote64); the cheap point (0.30/16) buys 22.6 tok/s on code and 15.6 PL / 17.5 EN on prose |
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

## DFlash status (help wanted)

[RedHatAI/DeepSeek-V4-Flash-speculator.dflash](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-speculator.dflash)
is a 2B parallel-drafting speculator with excellent numbers on its target.
This fork contains the complete runtime support for running it against an MLA
target - it boots, drafts and serves. However, the published speculator was
trained against **base** `deepseek-ai/DeepSeek-V4-Flash`, whose weights differ
from `-0731`, and acceptance does not transfer: measured on 0731 at k=7,
**11.8%** on code at greedy (position 1: 45.7%, 1.82 accepted tokens/step) and
**2.7%** on prose at temperature 1.0, against 41.6% / 78.8% / 3.9 on the base
target. Two integration notes for anyone reproducing this: the published
`config.json` leaves `target_hidden_size` at `null` (it needs `16384` for
`hc_mult=4`, otherwise the fc is sized for a quarter of its input), and
acceptance is strongly workload-dependent, so a single prose-at-temperature-1
number understates the checkpoint by ~4x. The moment a 0731-trained DFlash
speculator exists, this stack should hit ~2x the current decode. If you have
the training capacity - the runtime is ready and we will gladly test.
