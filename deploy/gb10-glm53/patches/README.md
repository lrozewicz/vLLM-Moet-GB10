# Patches for GLM-5.3-Flash on GB10

Two kinds of patches are needed on top of the vendor image, and they are
applied differently on purpose.

**Mounted patches** (this directory, applied by `build_mounts.sh`). The
original file is copied OUT of the image, patched on the host and bind-mounted
read-only over the original by `boot.sh`. The image itself is never modified,
so every one of these can be switched off for an A/B (`MOUNT_PATCHES=0`), and
an image update is one re-run of the script.

```bash
./deploy/gb10-glm53/patches/build_mounts.sh            # -> ~/glm53-run/patch
```

**Baked-in patches** (8, 9, 9b). These are applied at image build time by
`dflash2.dockerfile` (see `build_dflash2.sh`), because the DFlash2 overlay they
extend is itself installed at build time.

Every script checks its own marker (running it twice is a no-op), validates
the syntax of the result, and fails loudly when an anchor has moved rather
than silently doing nothing. The numbers are historical (the order in which
the problems were found); several scripts carry their working notes in Polish.

## What each patch fixes

| # | script | target | status | what it fixes |
|---|---|---|---|---|
| 1 | `patch_swiglu.py` | `moe_w2_cubit.py` | kept, no-op | Reads the SwiGLU clamp from `moe_config` when the layer does not carry it, and logs it. The hypothesis (the checkpoint's `swiglu_limit` 10.0 was silently lost on the expert path) was **rejected**: the limit is read correctly. Kept so the output matches production byte for byte. |
| 2 | `patch_partial_fp4.py` | `moe_w2_cubit.py` | **required** | A layer without FP4 sections in the planes cache loads as plain 2-bit instead of forcing a requantization from the checkpoint. That requant, with 40 other layers already resident, dies at 0.4-0.8 GiB free (measured three times) - a deadlock on unified memory. |
| 5 | `patch_store_skip_read.py` | `moe_w2_cubit.py` | **required** | Does not read the FP4 sections from the planes cache when the NVMe store already holds that layer. The read fed nothing, cost a 3.4 GiB transient buffer per layer and 145 GiB of I/O per boot. Anchors in code left by patch 2, so it must run after it. |
| 3 | `patch_delta_skip.py` | `moe_w2_delta.py` | **required** | Never promotes a layer that has no rows in the FP4 pack. `DeltaTier` iterated `range(n_layers)` blindly; the missing rows are holes in a sparse file, and reading them ended in `illegal memory access`. |
| - | `patch_v2_runner.py` | `v1/worker/gpu/model_runner.py` | **required** | The FP4 tier's per-step lifecycle in the **V2** model runner: `wait_manager_idle()` before the forward, `step_end()` after it. vLLM selects V2 for this model and speculative decoding on a hybrid model works only on V2, while upstream's hooks live in V1. Without them the tier accumulates pins until it silently stops promoting. |
| 4 | `patch_mtp_token.py` | `v1/spec_decode/llm_base_proposer.py` | MTP mode | Upstream typo: the code reads `config.image_token_index`, `Glm5NextConfig` has `image_token_id`. Blocks the MTP head on this model. Harmless with DFlash2. |
| 6 | `patch_pdl_off.py` | `platforms/cuda.py` | switch, off | `VLLM_DISABLE_PDL=1` turns off Programmatic Dependent Launch, which vLLM enables for every major >= 9 including the unvalidated sm_121. Default unchanged; kept as an A/B switch. |
| 7 | `patch_kda_ssm_dtype.py` | `model_executor/layers/mamba/mamba_utils.py` | switch, off | `VLLM_KDA_SSM_DTYPE` sets the dtype of the KDA recurrent state (default f32, unchanged). Measured: bf16 breaks code, fp16 accumulates error in decode (KLD x2, perplexity +0.4-0.5%). Leave it empty. |
| 12 | `patch_mamba_align_block.py` | `v1/core/sched/scheduler.py` | **required** | Aligns prefill chunks to the KDA state block (7168 tokens) instead of `cache_config.block_size`. EngineCore lowers the latter to the smallest prefix-caching group block, which with the DFlash2 drafter (patch 8) is 1024, so a chunk could run across a state boundary: the KDA checkpoint was cached under the boundary's hash while holding the state of an earlier chunk end. Restoring it caused rare, deterministic gibberish-from-the-first-token turns in long agent sessions (reproduced by replaying a captured session; clean without prefix-cache hits). `PATCH_SCHED=0` switches it off for an A/B. |
| 13 | `patch_w8a16_loader.py` + `glm53_w8a16.py` | `model_executor/model_loader/utils.py` + new module `model_executor/layers/quantization/utils/glm53_w8a16.py` | **on by default** (`W8A16_RE`) | Block-128 FP8 W8A16 for the BF16 linears that dominate a decode step on GB10 (46% of it, nsys): KDA q/k/v/o and lm_head (BF16 in the FP8 release), the MLA q_b/o/fused_qkv_a (FP8 in the checkpoint, dequantized to BF16 on load by the NoPE overlay), and the whole DFlash2 drafter. The loader hook gives every module whose name matches `VLLM_GLM53_W8A16_RE` the W8A16 method before its `process_weights_after_loading`; the module requantizes the weight and keeps ONE plain FP8 copy. Decode runs a Triton W8A16 kernel on it (BF16 activations, as fast as FP8 Marlin, which needs a second repacked copy), prefill (M >= 128) dequantizes it to BF16 (bit-exact) for the stock GEMM. KDA gates, `kv_b_proj`, indexer and router stay BF16. Measured: decode +25-34% on code, +29-33% on prose, weights 99.6 -> 92.8 GiB; quality in [the model doc](../../../docs/models/glm-5.3-flash.md). `W8A16_RE=` switches it off. |
| 14 | `patch_moe_loop.py` + `kernels/cubins-sm120-loop` | `moe_w2_cubit.py` + the cubin dir (`$PATCH_DIR/cubit` over `/cubit-share`) | **on by default** (`LOOP_GRID`, `FUSE_GLUE`) | The decode MoE launch had `grid.y` = the moe_align table length, and each tier's empty pairs (m_rows = 0) were CTAs that loaded their desc and exited: 4-5 ms per step on GB10. The desc build compacts each tier's live pairs to the front of its table and looped cubins (the shipped kernel bodies, schedule unchanged, plus a grid-stride loop) walk it from a fixed grid (16 / 12). Also fuses the MoE glue: the token quant, row gather, scale copy and the FP4 tier's seen flags in one Triton kernel, the SwiGLU and the second quant in another. Bit-exact with the old path (`tools/glm53/loop_check.py`, `desc_check.py`, `glue_bench.py`); code 39.2 -> 42.2 tok/s. `LOOP_GRID= FUSE_GLUE=0` = the old launch. Runs after patch 5. |
| 15 | `patch_adaptive_k.py` + `patch_adaptive_graphs.py` | `v1/core/sched/scheduler.py` + `v1/worker/gpu/cudagraph_utils.py` | **on by default** (`ADAPTIVE`) | The draft depth per step from live acceptance. A decode step here costs ~81 + 7.4 k ms (k = 3 / 5 / 7: 103 / 117 / 133 ms), so with prose's acceptance (~2.2 tokens) verifying 4 tokens instead of 8 is 15% faster, while code (acceptance 5.5) needs k = 7. The scheduler keeps a per-request EMA of the conditional acceptance p and picks the k in `ADAPTIVE_KS` (3,7) that maximizes E[tokens](k) / T(k) (Strata's cost-driven controller, ported from the MiMo deployment); it rides on vLLM's dynamic speculative decoding (`num_speculative_tokens_per_batch_size`, set by boot.sh). The V2 runner captures FULL decode graphs only for the k values that table lists, so `patch_adaptive_graphs.py` adds the KS values. Lossless (verification decides). `ADAPTIVE=0` = fixed k. |
| 10 | `patch_xdump.py` | `moe_w2_cubit.py` (variant) | GPTQ only | Dumps the MoE inputs of selected layers for the GPTQ Hessians (`tools/glm53/gptq_build.py`). Produces `moe_w2_cubit.xdump.py`, mounted only with `boot.sh XDUMP=1`. Records only while `$XDUMP_DIR/ARM` exists. |
| 8 | `patch_glm5_draft_small_block.py` | `v1/core/kv_cache_utils.py` | baked, **required** | DFlash2 drafter KV at TP=1: the MLA page is 656 B/token = 16 x 41, so the overlay's exact-fit mode can never apply, and its standalone fallback charges a 7168-token drafter page to every pool block (-58% KV capacity). Gives the drafter a 1024-token block (-16%) and strips the `page_size_padded` stamped by `--kv-cache-dtype-skip-layers`. |
| 9 | `patch_skip_layers_kv_dtype.py` | `layers/attention/attention.py` | baked, switch | `VLLM_KV_SKIP_LAYERS_DTYPE`: fp8 e4m3 KV for the skipped (drafter) layers. Computes correctly, but FlashInfer's fp8 workspace eats what it saves (83 vs 81 blocks). Default `auto` (bf16). |
| 9b | `patch_flashinfer_builder_dtype.py` | `v1/attention/backends/flashinfer.py` | baked | Companion of 9: the FlashInfer builder uses the per-layer KV dtype instead of the global `fp8_ds_mla`. |

Worth reporting upstream: the `image_token_index` typo (4) to vLLM, and the
missing layer check in the `DeltaTier` promotion loop (3) to vLLM-Moet.

## Reference checksums

On the reference box, `build_mounts.sh` against `vllm-moet-glm53:dflash2`
(built on the pinned base `vllm/vllm-openai@sha256:905c0293...`, image config
`d42649063a2b`) produces files byte-identical to the ones the production
server mounts. If your `.orig.py` checksums differ, the base image changed:
the patches may still apply (they anchor on code, not on line numbers), but
re-validate.

| file | `.orig.py` (from the image) | patched |
|---|---|---|
| `moe_w2_cubit.py` | `436d0352` | `265c3cc0` |
| `moe_w2_cubit.xdump.py` | - | `4b6d3eca` |
| `moe_w2_delta.py` | `ed93b97e` | `77606ddb` |
| `llm_base_proposer.py` | `17d9b24e` | `20525b32` |
| `v2_model_runner.py` | `de7d7416` | `b4d3ebbf` |
| `cuda_platform.py` | `59ccbb45` | `782d8c12` |
| `mamba_utils.py` | `270d7d1b` | `7ec86aa2` |
| `scheduler.py` | `3d8e5c39` | `a0153420` |
| `model_loader/utils.py` (`loader_utils.py`) | `dc8831e4` | `ffa1aef2` |
| `v1/worker/gpu/cudagraph_utils.py` (`cudagraph_utils.py`) | `b74ee64f` | `41e36d80` |
| `glm53_w8a16.py` (new, copied) | - | `3fcac2a6` |

(first 8 hex digits of md5; before patch 14 the patched `moe_w2_cubit.py` was
`822cf242` and its xdump variant `db09d1e0`). `$PATCH_DIR/cubit` is the image's
`/cubit-share` (80 cubins) with the four files of `kernels/cubins-sm120-loop`
swapped in (checksums in its README); `$PATCH_DIR/cubit.orig` is the
untouched copy that `MOUNT_PATCHES=0` mounts.
