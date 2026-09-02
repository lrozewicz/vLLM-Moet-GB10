#!/bin/bash
# Buduje obraz z drafterem DFlash2 NA obrazie vllm-moet-glm53:combined.
#
#   ./deploy/gb10-glm53/build_dflash2.sh        # z katalogu glownego repo
#
# Sklad (trzy zrodla, trzy statusy licencyjne):
#   1. port DFlash2 z upstreamu vLLM (PR #52816, Apache-2.0): qwen3_dflash2.py,
#      dflash2/speculator.py, patch_registry_and_select.py - pliki z naglowkiem
#      SPDX Apache-2.0 w repo tonyd2wild;
#   2. glue GLM-5.3 -> drafter (tonyd2wild, BEZ pliku LICENSE i bez naglowka):
#      patch_glm_aux_capture.py (SupportsEagle3 + kontrakcja mHC),
#      patch_glm5_drafter_group.py (grupa KV draftera na szybkiej sciezce
#      GLM5-Next). POBIERANE przy budowie, NIE trzymane w repo - ta sama
#      zasada co dla overlayu NoPE w build_combined.sh;
#   3. nasza LATKA 8 (patches/patch_glm5_draft_small_block.py): maly blok
#      draftera w trybie standalone - bez niej pojemnosc KV spada 2.35x.
#
# Sam drafter (incoai/GLM-5.3-Flash-DFlash2) jest CC-BY-NC-ND-4.0: badania
# i uzytek wlasny tak, komercyjnie - kontakt z autorami.
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-$REPO_ROOT/.build-dflash2}"
TAG="${TAG:-vllm-moet-glm53:dflash2}"
BASE="${BASE:-vllm-moet-glm53:combined}"
UP="${UP:-https://raw.githubusercontent.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/main/overlay-dflash2}"

mkdir -p "$WORK/overlay/dflash2"
echo "== pobieram overlay DFlash2 (sekwencyjnie, jeden strumien; $WORK jest w .gitignore)"
for f in qwen3_dflash2.py dflash2/__init__.py dflash2/speculator.py \
         patch_registry_and_select.py patch_glm_aux_capture.py \
         patch_glm5_drafter_group.py; do
  [ -s "$WORK/overlay/$f" ] || curl -fsSL "$UP/$f" -o "$WORK/overlay/$f"
done
cp "$REPO_ROOT/deploy/gb10-glm53/patches/patch_glm5_draft_small_block.py" "$WORK/overlay/"
cp "$REPO_ROOT/deploy/gb10-glm53/patches/patch_skip_layers_kv_dtype.py" "$WORK/overlay/"
cp "$REPO_ROOT/deploy/gb10-glm53/dflash2.dockerfile" "$WORK/Dockerfile"
sed -i -E "s|^ARG BASE=.*|ARG BASE=${BASE}|" "$WORK/Dockerfile"

echo "== buduje $TAG na $BASE"
docker build -f "$WORK/Dockerfile" -t "$TAG" "$WORK"
echo "== gotowe: $TAG"
