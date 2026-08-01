# vLLM-Moet-GB10 — serving image for NVIDIA GB10 (DGX Spark class: NVIDIA DGX
# Spark, ASUS Ascent GX10, and other GB10 boxes), sm_121, aarch64.
#
# Build FROM THE REPO ROOT (the patch and cubins are copied from the context):
#
#   docker build -f deploy/gb10/Dockerfile -t vllm-moet-gb10:latest .
#
# What this image does:
#   1. DeepGEMM nv-dev built from source on arm64/cu130 (the release pin has no
#      host paths for the 12x family).
#   2. flashinfer 0.6.14 (0.6.12 predates the DS4 attention API on SM120) with
#      a repaired flashinfer-cubin state (see the pitfall note below).
#   3. The vLLM-Moet patch applied onto the official arm64 vLLM v0.24.0 image —
#      including this fork's GB10 additions (sm_121 family unlock, mxfp4
#      stream-build, Model Runner V2 hooks for the W2 path, DFlash-on-MLA
#      fixes). No post-patch fixup scripts needed.
#   4. sm_120 cubins installed to /cubit-share (family-compatible with sm_121:
#      verified bit-identical compute, see README).
#
# What it deliberately does NOT build: the NVFP4 KV-cache extension. That step
# compiles with `arch=compute_120a,code=sm_120a` — the `a` suffix means
# device-SPECIFIC, not family — so it cannot start on sm_121. Serve with
# `--kv-cache-dtype=fp8` (as all DS4 recipes in the upstream repo do).

ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a

# =========================================================== DeepGEMM builder
# Built INSIDE the vLLM base image (nvcc 13.0 + torch + py3.12) so the
# python/glibc/torch ABI matches the runtime by construction. The v0.24.0 tag
# is a multi-arch manifest — docker picks the arm64 variant on GB10.
FROM vllm/vllm-openai:v0.24.0 AS deepgemm-builder
ARG DEEPGEMM_REF
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libcusparse-dev-13-0 libcusolver-dev-13-0 libcublas-dev-13-0 \
        cuda-nvrtc-dev-13-0 cuda-profiler-api-13-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages pybind11 setuptools wheel packaging
RUN git clone https://github.com/deepseek-ai/DeepGEMM.git /deepgemm \
    && cd /deepgemm && git checkout "$DEEPGEMM_REF" \
    && git submodule update --init --depth 1 third-party/cutlass third-party/fmt \
    && python3 setup.py bdist_wheel

# =========================================================== serving image
FROM vllm/vllm-openai:v0.24.0

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# --- 0. fail-fast sanity of the arm64 base.
# (torch.cuda.get_arch_list() is NOT checked here: `docker build` has no GPU
# and the call returns an empty list — it would look like a failure and is not.)
RUN echo "=== arch: $(uname -m) ===" \
    && nvcc --version | tail -2 \
    && python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- 1. DeepGEMM nv-dev (site-packages wins over vLLM's vendored copy)
COPY --from=deepgemm-builder /deepgemm/dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps --break-system-packages /tmp/deep_gemm-*.whl \
    && rm /tmp/deep_gemm-*.whl

# --- 2. flashinfer 0.6.14 (+ cu130 jit-cache; the aarch64 wheel exists)
#
# PITFALL (found on GX10): `pip uninstall flashinfer-cubin` removes the package
# metadata but LEAVES the flashinfer_cubin/ directory behind (cubins downloaded
# post-install are outside the RECORD). Python then treats the orphaned
# directory as an implicit namespace package: `import flashinfer_cubin`
# succeeds but has no __version__, flashinfer/jit/env.py raises
# AttributeError, and the whole flashinfer import fails — hours later, at model
# start. Remove the directory explicitly.
RUN pip uninstall -y --break-system-packages flashinfer-cubin || true \
    && SP="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && rm -rf "$SP/flashinfer_cubin" \
    && echo "removed orphaned $SP/flashinfer_cubin"
RUN pip install --no-cache-dir --break-system-packages flashinfer-python==0.6.14 \
    && pip install --no-cache-dir --break-system-packages \
        --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==0.6.14+cu130"

# Import MUST work now — without this check the regression surfaces only after
# a long quantization run.
RUN python3 -c "import flashinfer, flashinfer_jit_cache; \
print('flashinfer', flashinfer.__version__, '| jit-cache', flashinfer_jit_cache.__version__)"

# --- 3. the vLLM-Moet-GB10 patch (pure-python delta; compiled artifacts of the
#        base image stay untouched)
COPY patch/vllm-moet-v0.24.0.patch /tmp/vllm-moet.patch
RUN cd "$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" \
    && git apply --verbose /tmp/vllm-moet.patch \
    && python3 -c "from vllm.model_executor.layers.quantization.utils import moe_w2_cubit; print('moe_w2 hook OK')" \
    && python3 -c "import vllm.v1.worker.gpu.spec_decode.dspark.speculator; print('dspark OK')" \
    && python3 -c "import vllm.v1.worker.gpu.spec_decode.dflash.speculator; print('dflash OK')" \
    && rm /tmp/vllm-moet.patch

# --- 4. sm_120 cubins (family-compatible with sm_121)
COPY kernels/cubins-sm120/ /cubit-share/
ENV VLLM_MOE_W2_CUBIT_DIR=/cubit-share

EXPOSE 8000
