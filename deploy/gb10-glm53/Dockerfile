# vLLM-Moet-GB10 — serving image for GLM-5.3-Flash on a single NVIDIA GB10
# (DGX Spark class: NVIDIA DGX Spark, ASUS Ascent GX10), sm_121, aarch64.
#
# Build FROM THE REPO ROOT (the patch, the overlay tool and the cubins are
# copied from the build context):
#
#   docker build -f deploy/gb10-glm53/Dockerfile -t vllm-moet-glm53:latest .
#
# Why a different base than deploy/gb10/Dockerfile
# ------------------------------------------------
# The DS4-Flash image is `vllm/vllm-openai:v0.24.0` + the full generated
# patch.  GLM-5.3-Flash cannot use it: its `glm5next` implementation (KDA
# linear attention, NoPE sparse MLA, mHC hyper-connections, the MTP head and
# the vision tower) does not exist in 0.24.0 and, as of 2026-08, is not in the
# public vLLM repo at all — it ships only in the `glm53-flash` images
# (vLLM 0.29-dev).  So we invert the layering: take the vendor image that
# already knows the model, and add only the 2-bit expert engine on top.
#
# That also means most of the v0.24.0 patch is unwanted here.  Its SM120 base
# fixes are upstream by 0.29, its sparse-MLA/nvfp4-KV backends are DS4-family
# specific, and its DSpark/DFlash speculators target DeepSeek heads.  What
# transfers is the part that is genuinely model-agnostic: the 2-bit planes,
# their SASS GEMMs, and the FP8->2-bit loader hook.  `tools/apply_w2_overlay.py`
# extracts exactly that from the patch, so the patch stays the single source of
# truth for W2 code (AGENTS.md) and a regeneration propagates here for free.
#
# Memory budget this image is built around (121.6 GiB unified, measured shapes
# from the checkpoint, see docs/GLM-5.3-FLASH.md):
#   routed experts, 43 layers x 288 x 3 x 2048 x 4096 at 2 bits ... 72.6 GiB
#   everything else (FP8 dense/MLA, BF16 KDA, embeddings, vision)  15.4 GiB
#   ------------------------------------------------------------------------
#   weights                                                        88.0 GiB
# leaving roughly 33 GiB for KV, activations and graphs.  The KV side is
# unusually cheap for a 320B model: only 12 of 45 layers are attention with a
# growing cache (kv_lora_rank 512, NoPE, so 512 elements/token/layer), the
# other 34 are KDA with a constant-size recurrent state.

ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130
FROM ${BASE}

# --- 0. fail-fast sanity of the base: it must be arm64 and it must already
#        carry the model implementation we are building on top of.
#        (torch.cuda.get_arch_list() is NOT checked — `docker build` has no
#        GPU and returns an empty list, which looks like a failure and is not.)
RUN echo "=== arch: $(uname -m) ===" \
    && python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
    && python3 -c "import vllm; print('vllm', vllm.__version__)" \
    && python3 -c "import vllm.models.glm5next as m; print('glm5next present:', m.__file__)" \
    && python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

# --- 1. the moe_w2 engine, extracted from the generated patch and hooked into
#        this tree by anchor.  The tool is idempotent and fails loudly if an
#        upstream call site moved, rather than guessing.
COPY patch/vllm-moet-v0.24.0.patch /tmp/vllm-moet.patch
COPY tools/apply_w2_overlay.py /tmp/apply_w2_overlay.py
RUN python3 /tmp/apply_w2_overlay.py --patch /tmp/vllm-moet.patch \
    && python3 -c "from vllm.model_executor.layers.quantization.utils import moe_w2_cubit; print('moe_w2 hook OK')" \
    && python3 -c "from vllm.model_executor.layers.quantization.utils import moe_w2_delta, moe_w2_gate, moe_w2_planes, moe_w2_store; print('moe_w2 tiers OK')" \
    && python3 -c "import vllm.model_executor.layers.quantization.fp8 as f; import inspect; assert 'moe_w2' in inspect.getsource(f.Fp8MoEMethod.apply), 'fp8 apply hook missing'; print('fp8 forward hook OK')" \
    && rm /tmp/vllm-moet.patch /tmp/apply_w2_overlay.py

# --- 2. the hand-written SASS MoE GEMMs.  GLM-5.3-Flash needs K=4096
#        (gate/up: [2048, 4096]) and K=2048 (down: [4096, 2048]); both are in
#        the shipped `_a32` family, so this model needs no new kernel work.
#        sm_120 cubins are family-compatible with sm_121 (verified
#        bit-identical compute — see README).
COPY kernels/cubins-sm120/ /cubit-share/
ENV VLLM_MOE_W2_CUBIT_DIR=/cubit-share
RUN for k in 4096 2048; do \
        for f in moe_w2_mm_k${k}_a32 moe_w2_mm_mc4_k${k}_a32 moe_w2_mm_mc4afrag_k${k}_a32 moe_w4_mm_k${k}_a32; do \
            test -s "/cubit-share/${f}.cubin" || { echo "MISSING cubin: ${f}"; exit 1; }; \
        done; \
    done && echo "cubins for K=4096/2048 present"

EXPOSE 8000
