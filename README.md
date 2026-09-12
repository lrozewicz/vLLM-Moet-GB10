# vLLM-Moet-GB10

**Frontier Mixture-of-Experts models on a single NVIDIA GB10 box** - NVIDIA
DGX Spark, ASUS Ascent GX10 and other GB10 systems with 121.6 GiB of unified
memory. Models whose checkpoints are two to three times larger than the
machine's memory are served with vLLM at interactive speed. The routed
experts become 2-bit planes computed by hand-written SASS kernels, and a
runtime FP4 tier restores precision where it matters. Speculative decoding
supplies the speed, and a set of fixes covers what unified memory breaks.

## Supported models

| model | size | decode, code | context | quality, measured | guide |
|---|---|---:|---:|---|---|
| **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** | 321B total / 18B active; FP8 checkpoint 306 GiB | **~29 tok/s** (DFlash2) | **200K** | HumanEval **97.0%**, HumanEval+ **93.9%** (the original on z.ai: 95.1% / 89.6%) | [`docs/models/glm-5.3-flash.md`](docs/models/glm-5.3-flash.md) |
| **[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)** | 304B (284B + 20B DSpark head); ~156 GB | ~21 tok/s (DSpark k=2) | 256K | [`docs/quality.md`](docs/quality.md) | [`docs/models/deepseek-v4-flash-0731.md`](docs/models/deepseek-v4-flash-0731.md) |

Each guide covers what had to change for that model, what was measured, why
the quality holds, and a step-by-step setup. Per-release history:
[`CHANGELOG.md`](CHANGELOG.md) (latest: **v0.3.0**, GLM-5.3-Flash).

## How it works

The same engine serves every model; what differs per model is the vLLM base
and the overlays needed to make that model run at all.

- **2-bit expert planes** (upstream vLLM-Moet). The routed experts are ~97%
  of a large MoE's parameters, and each token uses only a few of them. They
  are stored as codes {-4, -1, 1, 4} times one scale per 32 weights, and
  hand-written SASS GEMMs compute directly on them. Everything else
  (attention, shared experts, dense layers, embeddings, router) keeps the
  checkpoint's own precision.
- **FP4 delta tier** (upstream). A pool of the hottest experts, served in FP4
  instead of 2-bit and chosen at runtime from the actual routing. The full
  FP4 copy lives on NVMe, and promotions read from it while the model serves.
- **Persistent planes cache.** Quantization runs once, on the first boot;
  every later boot loads the planes directly.
- **Speculative decoding.** DSpark (DeepSeek's built-in head) for DS4, the
  DFlash2 block-diffusion drafter for GLM-5.3-Flash.
- **Unified-memory engineering** (this fork). Streaming plane builds, the FP4
  store on NVMe instead of pinned host RAM, page-cache handling, memory
  guards, and the runner hooks ported to vLLM's V2 model runner. The full
  delta against upstream: [`docs/CHANGES-VS-UPSTREAM.md`](docs/CHANGES-VS-UPSTREAM.md).

DeepSeek-V4-Flash-0731 runs on the official `vllm/vllm-openai:v0.24.0` image
plus the generated patch. GLM-5.3-Flash needs vLLM code that exists only in
the vendor `glm53-flash` image (0.29-dev), so there the engine is extracted
from the same patch and installed onto that image
([`tools/apply_w2_overlay.py`](tools/apply_w2_overlay.py)).

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

GLM-5.3-Flash on one box also stands on community work:

- **[chriswritescode-dev/glm-5.3-flash-sm120](https://github.com/chriswritescode-dev/glm-5.3-flash-sm120)**
  (from [vllm#53963](https://github.com/vllm-project/vllm/issues/53963)):
  the NoPE sparse-MLA fix for consumer Blackwell. It zero-pads the 512-wide
  latent into the existing 576-wide GLM_NSA geometry - exact, and no CUDA.
  This is what unblocked the model on sm_12x.
- **[tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)**:
  the DFlash2 overlay (a port of vLLM PR #52816) and its GLM glue.
- **[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)**:
  the drafter itself (CC-BY-NC-ND-4.0, non-commercial).
- **[darkdatter/gb10-repo](https://github.com/darkdatter/gb10-repo)**: the
  HumanEval harness pattern and the lesson that on this hardware the drafter
  is the largest single-stream lever.

The NoPE overlay and the GLM glue carry no license, so this repository does
not redistribute them: the build scripts download them at build time.

## Quick start

Common prerequisites: a GB10 box with the NVIDIA driver (tested: 580.159.03,
CUDA 13.0), docker with the NVIDIA container toolkit, network access for the
one-time downloads, and the disk space listed in the model's guide. Then
follow the guide of the model you want:

- **GLM-5.3-Flash** -> [`docs/models/glm-5.3-flash.md`](docs/models/glm-5.3-flash.md#setup-step-by-step)
- **DeepSeek-V4-Flash-0731** -> [`docs/models/deepseek-v4-flash-0731.md`](docs/models/deepseek-v4-flash-0731.md#quick-start)

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
   hence the documented consent valves in the compose files.
5. **Page cache blocks the next boot.** vLLM's start-up check effectively
   compares `MemFree`, not `MemAvailable`, with the memory it wants. A boot
   that reads a 300 GB checkpoint leaves tens of GiB of page cache behind,
   and the next boot fails. `posix_fadvise(DONTNEED)` on the files releases
   it without root ([`drop_page_cache.py`](deploy/gb10-glm53/drop_page_cache.py),
   measured +43.6 GiB).
6. **`docker`'s `mem_limit` does not protect the host.** Measured during a
   GLM load: the container's cgroup reported 9.6 GiB while the host had
   118 GB in use - GPU allocations go through the driver and are not charged
   to the cgroup.
7. **A wedged box may still answer SSH's banner but never log you in.** Arm
   a guard before every first boot. [`hostguard.py`](deploy/gb10-glm53/hostguard.py)
   is a resident process that reads `/proc/meminfo` and kills the container
   through the docker socket without forking. A guard that shells out per
   check (like `memguard.sh`) cannot fork once the box thrashes, and it slept
   through two wedges.
8. **A checkpoint on a USB SSD: raise read-ahead.** At the default
   `read_ahead_kb` (1024 on our enclosure) a drive with ~20 ms per request
   read 99 MiB/s during a boot. At 16384 it read ~470 MiB/s, and a boot fell
   from ~45 to ~17 min (`/sys/block/<dev>/queue/read_ahead_kb`, persisted
   with a udev rule).

## Roadmap

- GLM-5.3-Flash: a static FP4 pool (a served model that is a fixed
  function), a hybrid KDA state (fp32 working state, fp16 checkpoints -
  smaller KV block, two long streams with DFlash2), GPTQ for the down
  projection, and an FP8 reference measurement on larger hardware.
- DeepSeek-V4-Flash-0731: a 0731-trained DFlash speculator (see its guide).
- More big-MoE models on single-GB10.

## Repository layout

- `patch/vllm-moet-v0.24.0.patch` - the whole engine as a patch onto the
  official `vllm/vllm-openai:v0.24.0` arm64 image (upstream's distribution
  model, preserved; regenerated with `tools/check_patch_files.py`).
- `deploy/gb10/` - DeepSeek-V4-Flash-0731: Dockerfile, the annotated compose
  file, memory guard, preflight checks, checkpoint downloader.
- `deploy/gb10-glm53/` - GLM-5.3-Flash: image build scripts, the serving
  compose file, `boot.sh`, `hostguard.py`, the checkpoint downloader, and
  `patches/` (mounted and baked-in vLLM patches, with `build_mounts.sh`).
- `kernels/cubins-sm120/` - upstream's hand-written SASS kernels (run
  bit-identically on sm_121).
- `tools/` - `apply_w2_overlay.py` (installs the engine on a non-0.24 vLLM
  tree), `build_fp4_sections.py` (offline FP4 sections), KV-geometry and
  prefix-cache probes; `tools/glm53/` - the GLM measurement tools (KLD
  harness, HumanEval, loop and quality probes, benchmark) and the GPTQ
  builder.
- `docs/models/` - per-model guides. `docs/GLM-5.3-FLASH.md` - the
  GLM-5.3-Flash bring-up log. `docs/CHANGES-VS-UPSTREAM.md`,
  `docs/quality.md`, `docs/UPSTREAM-README.md` (the original upstream README,
  worth reading for the architecture and the quality methodology).
- `CHANGELOG.md` - per-release history with the measurements behind each item.

## License

Apache-2.0, same as upstream. See `LICENSE` and `NOTICE`. Model weights are
not included. Components the build scripts download keep their own terms:
the GLM-5.3-Flash NoPE overlay and DFlash2 glue carry no license (downloaded
at build time, never redistributed here), and the DFlash2 drafter weights are
CC-BY-NC-ND-4.0.
