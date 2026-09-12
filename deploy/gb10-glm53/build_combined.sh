#!/bin/bash
# Buduje obraz laczacy DWA overlaye na tej samej bazie vLLM:
#   1. NoPE sparse-MLA dla SM12x - chriswritescode-dev/glm-5.3-flash-sm120
#   2. silnik 2-bitowych ekspertow W2 - ten repozytorium
#
#   ./deploy/gb10-glm53/build_combined.sh          # uruchamiac z katalogu glownego repo
#
# DLACZEGO POBIERAMY, A NIE TRZYMAMY KOPII: repozytorium (1) nie ma pliku
# LICENSE, wiec domyslnie obowiazuje "wszelkie prawa zastrzezone" - nie wolno
# nam redystrybuowac go w repo na Apache-2.0. Pobranie przy budowie u siebie to
# zwykle uzycie cudzej pracy, nie jej rozpowszechnianie. Gdyby autor dodal
# licencje zgodna z Apache-2.0, mozna to uproscic do wendorowanej kopii.
#
# Co ten overlay robi (opisane w ich Dockerfile, warto przeczytac): GLM-5.3-Flash
# jest NoPE (qk_rope_head_dim=0), a jedyny sparse-MLA na SM12x zada 64-bitowej
# polowki rope. Zamiast pisac kernel - dopychaja 512-wymiarowy latent zerami do
# geometrii 576 GLM_NSA. Zerowy blok rope nie wnosi nic do iloczynu QK, wiec
# wynik jest dokladny. Zgloszenie z pelna analiza: vllm-project/vllm#53963.
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-$REPO_ROOT/.build-combined}"
TAG="${TAG:-vllm-moet-glm53:combined}"
# PINNED to what every published number was measured on. BASE is the arm64
# variant of the multi-arch digest their Dockerfile pins (image config
# sha256:d42649063a2b..., vLLM 0.1.dev20051+g487ecf187). The tag
# vllm/vllm-openai:glm53-flash-arm64-cu130 was re-pushed with a DIFFERENT
# image on 2026-09-09, so a tag here would silently build on an unvalidated
# base. The overlay is pinned to the commit that was current for our builds
# (identical bytes to what we used). Override either to try newer versions.
BASE="${BASE:-vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce}"
UPSTREAM_URL="${UPSTREAM_URL:-https://raw.githubusercontent.com/chriswritescode-dev/glm-5.3-flash-sm120/dc6b4fdd68005ab6ee0b1decfa4ebb8384393d37/Dockerfile}"

mkdir -p "$WORK"
echo "== pobieram overlay NoPE (nie trafia do repo: $WORK jest w .gitignore)"
curl -fsSL "$UPSTREAM_URL" -o "$WORK/upstream.dockerfile"

# Ich Dockerfile pinuje baze po digescie listy multi-arch; podmieniamy na nasz
# wariant arm64 TEJ SAMEJ listy (ten sam vLLM 0.1.dev20051+g487ecf187).
sed -E "s|^ARG BASE=.*|ARG BASE=${BASE}|" "$WORK/upstream.dockerfile" > "$WORK/Dockerfile"
grep -q "^ARG BASE=${BASE}$" "$WORK/Dockerfile" || { echo "!!! nie podmienilem BASE"; exit 1; }

# ...i doklejamy nasza czesc. Kolejnosc ma znaczenie tylko o tyle, ze oba
# overlaye sprawdzaja kotwice twardo - ewentualna kolizja wysypie BUILD,
# a nie start modelu 10 minut pozniej.
cat "$REPO_ROOT/deploy/gb10-glm53/w2-tail.dockerfile" >> "$WORK/Dockerfile"

echo "== buduje $TAG"
docker build -f "$WORK/Dockerfile" -t "$TAG" "$REPO_ROOT"
echo "== gotowe: $TAG"
