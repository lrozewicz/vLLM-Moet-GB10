# Łatki dla GLM-5.3-Flash na GB10

Skrypty nakładające łatki na pliki **wewnątrz obrazu** `vllm-moet-glm53:combined`.
Nie modyfikują obrazu — produkują pliki, które `boot.sh` montuje do kontenera
(`SWIGLU_PATCH=1`). Dzięki temu da się je włączać i wyłączać dla A/B, a po
aktualizacji obrazu odtworzyć jednym przebiegiem.

## Jak odtworzyć na czystym boksie

```bash
cd ~/glm53-run/patch
CID=$(docker create vllm-moet-glm53:combined)
D=/usr/local/lib/python3.12/dist-packages/vllm
docker cp $CID:$D/model_executor/layers/quantization/utils/moe_w2_cubit.py  moe_w2_cubit.orig.py
docker cp $CID:$D/model_executor/layers/quantization/utils/moe_w2_delta.py  moe_w2_delta.orig.py
docker cp $CID:$D/v1/spec_decode/llm_base_proposer.py                       llm_base_proposer.orig.py
docker cp $CID:$D/v1/worker/gpu/model_runner.py                             v2_model_runner.orig.py
docker rm $CID

cp moe_w2_cubit.orig.py moe_w2_cubit.py
python3 patch_swiglu.py       moe_w2_cubit.py     # diagnostyka (hipoteza odrzucona)
python3 patch_partial_fp4.py  moe_w2_cubit.py
python3 patch_store_skip_read.py moe_w2_cubit.py
cp moe_w2_delta.orig.py moe_w2_delta.py
python3 patch_delta_skip.py   moe_w2_delta.py
cp llm_base_proposer.orig.py llm_base_proposer.py
python3 patch_mtp_token.py    llm_base_proposer.py
cp v2_model_runner.orig.py v2_model_runner.py
python3 patch_v2_runner.py    v2_model_runner.py
```

Każdy skrypt jest idempotentny (sprawdza swój znacznik) i waliduje składnię.
Kotwice są dosłowne — po aktualizacji vLLM mogą wymagać przesunięcia; skrypt
wtedy **zawodzi głośno**, zamiast po cichu nie zrobić nic.

Kolejność dla `moe_w2_cubit.py` ma znaczenie: `patch_store_skip_read.py`
kotwiczy się w kodzie, który zostawia po sobie `patch_partial_fp4.py`.

Pełny kontekst: `gx10/docs/GLM-5.3-FLASH-STAN.md`.
