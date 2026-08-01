#!/bin/bash
# Pobranie DeepSeek-V4-Flash-0731 (~167 GB, 48 shardow) na GX10.
#
# Uruchamiane w tmux:  tmux new-session -d -s ds4dl ~/moet-preflight/download_ds4_0731.sh
# Podglad:             tmux attach -t ds4dl      (odlaczenie: Ctrl-B, potem D)
# Log:                 tail -f ~/moet-preflight/download.log
#
# hf CLI nie ma na hoscie (brak huggingface_hub), wiec jedzie w kontenerze z naszego obrazu
# (huggingface_hub 1.21.0). Token montowany z ~/.cache/huggingface — repo jest publiczne,
# ale token nie przeszkadza.
#
# --max-workers 2  -> spojnie z max-concurrent-downloads:2 w Dockerze. Przy 16 router w domu
#                     sie zatyka (conntrack/bufferbloat) i cala LAN laguje.
# HF_HUB_DISABLE_XET=1 -> Xet potrafil wieszac pobieranie shardow; klasyczne HTTPS ma resume.

set -u

REPO=deepseek-ai/DeepSeek-V4-Flash-0731
OUT=/mnt/adata/ai/models/deepseek-v4-flash-0731
IMG=vllm-moet-gb10:v024
LOG="$HOME/moet-preflight/download.log"
SHARDS=48

mkdir -p "$OUT" "${OUT}-cache" "${OUT}-packs" "$(dirname "$LOG")"

{
  echo "================================================================"
  echo "=== START $(date -Is)  repo=$REPO"
  echo "=== cel: $OUT   (wolne: $(df -h "$OUT" | tail -1 | awk '{print $4}'))"
  echo "================================================================"
} | tee -a "$LOG"

rc=1
for attempt in 1 2 3 4 5 6; do
  echo "--- proba $attempt/6  $(date -Is) ---" | tee -a "$LOG"
  docker run --rm --name hf-dl-ds4 \
    -v "$OUT":/out \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -e HF_HUB_DISABLE_XET=1 \
    --entrypoint hf "$IMG" \
    download "$REPO" --local-dir /out --max-workers 2 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && break
  echo "!!! proba $attempt zwrocila rc=$rc — ponawiam za 60 s (pobrane pliki sie nie powtorza)" \
    | tee -a "$LOG"
  sleep 60
done

n=$(ls "$OUT"/model-*.safetensors 2>/dev/null | wc -l)
sz=$(du -sh "$OUT" 2>/dev/null | cut -f1)
part=$(find "$OUT" -name '*.incomplete' -o -name '*.part' 2>/dev/null | wc -l)

{
  echo "================================================================"
  echo "=== KONIEC $(date -Is)  rc=$rc"
  echo "=== shardow: $n/$SHARDS   rozmiar: $sz   niedokonczonych plikow: $part"
  if [ "$rc" -eq 0 ] && [ "$n" -eq "$SHARDS" ] && [ "$part" -eq 0 ]; then
    echo "=== WYNIK: KOMPLET OK"
  else
    echo "=== WYNIK: NIEKOMPLETNE — uruchom skrypt ponownie, wznowi od miejsca przerwania"
  fi
  echo "================================================================"
} | tee -a "$LOG"
