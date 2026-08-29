
# ============================== vLLM-Moet ==============================
# Doklejane przez deploy/gb10-glm53/build_combined.sh do pobranego overlaya
# NoPE. Ta czesc jest nasza; tamta pobierana jest przy budowie (patrz nota
# licencyjna w build_combined.sh).
#
# Silnik 2-bitowych ekspertow. Overlay NoPE naprawia UWAGE, ale nie rozmiar -
# zweryfikowano go na 4x RTX PRO 6000 (384 GB) z checkpointem FP8 306 GiB.
# Na JEDNYM GB10 (121.6 GiB) to sie nie miesci. W2 sprowadza eksperty do
# ~96 GiB planow 2-bit i dopiero razem daja model na pojedynczym boksie.
#
# Zbiory patchowanych plikow sa rozlaczne: oni ruszaja sparse-MLA, indexer
# kpool i platforms/cuda.py; my quantization/fp8.py, moe_w2_cubit.py, envs.py
# i gpu_model_runner.py. Baza to ta sama wersja vLLM, wiec kotwice trzymaja.
COPY patch/vllm-moet-v0.24.0.patch /tmp/vllm-moet.patch
COPY tools/apply_w2_overlay.py /tmp/apply_w2_overlay.py
RUN python3 /tmp/apply_w2_overlay.py --patch /tmp/vllm-moet.patch \
    && python3 -c "from vllm.model_executor.layers.quantization.utils import moe_w2_cubit; print('moe_w2 hook OK')" \
    && python3 -c "import vllm.model_executor.layers.quantization.fp8 as f; import inspect; assert 'moe_w2' in inspect.getsource(f.Fp8MoEMethod.apply), 'fp8 apply hook missing'; print('fp8 forward hook OK')" \
    && python3 -c "import inspect, vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 as m; assert 'rope_pad' in inspect.getsource(m), 'NoPE overlay missing'; print('NoPE sparse-MLA overlay OK')" \
    && rm /tmp/vllm-moet.patch /tmp/apply_w2_overlay.py

# Recznie pisane SASS-owe GEMM-y MoE. Ten model potrzebuje K=4096 (gate/up:
# [2048, 4096]) i K=2048 (down: [4096, 2048]) - oba sa w wydanym zestawie
# `_a32`, wiec nie wymaga nowych kerneli. Cubiny sm_120 licza bit-w-bit tak
# samo na sm_121 (ta sama rodzina 12.x).
COPY kernels/cubins-sm120/ /cubit-share/
ENV VLLM_MOE_W2_CUBIT_DIR=/cubit-share
RUN for k in 4096 2048; do \
        for f in moe_w2_mm_k${k}_a32 moe_w2_mm_mc4_k${k}_a32 moe_w2_mm_mc4afrag_k${k}_a32 moe_w4_mm_k${k}_a32; do \
            test -s "/cubit-share/${f}.cubin" || { echo "MISSING cubin: ${f}"; exit 1; }; \
        done; \
    done && echo "cubins for K=4096/2048 present"

EXPOSE 8000
