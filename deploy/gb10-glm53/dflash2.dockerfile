# DFlash2 speculative decoding dla GLM-5.3-Flash na JEDNYM GB10.
# Budowane przez deploy/gb10-glm53/build_dflash2.sh (kontekst = .build-dflash2,
# gdzie laduje pobrany overlay + nasza latka 8). Patrz nota w build-skrypcie.
#
# DLACZEGO. Glowica MTP tego modelu to PELNA warstwa MoE (288 ekspertow) i przy
# k=4 przechodzimy przez nia cztery razy sekwencyjnie na krok - zmierzone
# ~46% czasu kroku decode. DFlash2 to 1B drafter dyfuzyjny: JEDEN przebieg
# daje blok 8 tokenow. Na dwoch Sparkach (TP2, NVFP4) dal 21.8 -> 46.9 tok/s
# wobec MTP-4. Tutaj drafter zastepuje warstwe 45 z planow (-1.9 GiB), wiec
# bilans pamieci wag jest ~zero.
ARG BASE=vllm-moet-glm53:combined
FROM ${BASE}
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm

# 1. port DFlash2 z upstreamu (Apache-2.0)
COPY overlay/qwen3_dflash2.py $VLLM/model_executor/models/qwen3_dflash2.py
COPY overlay/dflash2/ $VLLM/v1/worker/gpu/spec_decode/dflash2/
COPY overlay/patch_registry_and_select.py /tmp/dflash2/
# 2. glue GLM-5.3 (pobrane, nie wendorowane)
COPY overlay/patch_glm_aux_capture.py overlay/patch_glm5_drafter_group.py /tmp/dflash2/
# 3. nasza latka 8
COPY overlay/patch_glm5_draft_small_block.py /tmp/dflash2/

RUN set -e; cd /tmp/dflash2 \
    && python3 patch_registry_and_select.py \
    && python3 patch_glm_aux_capture.py \
    && python3 patch_glm5_drafter_group.py \
    && python3 patch_glm5_draft_small_block.py $VLLM/v1/core/kv_cache_utils.py \
    && python3 -c "from vllm.model_executor.models.registry import ModelRegistry; assert 'DFlash2DraftModel' in ModelRegistry.get_supported_archs(); print('DFlash2DraftModel zarejestrowany')" \
    && python3 -c "import inspect, vllm.v1.core.kv_cache_utils as k; s=inspect.getsource(k); assert 'DFLASH2-DRAFTER-GROUP' in s and 'PATCH-DRAFT-SMALL-BLOCK' in s; print('grupa draftera + maly blok OK')" \
    && python3 -c "import inspect, vllm.models.glm5next.nvidia.model as m; assert 'DFLASH2-AUX-CAPTURE' in inspect.getsource(m); print('aux capture OK')" \
    && rm -rf /tmp/dflash2
