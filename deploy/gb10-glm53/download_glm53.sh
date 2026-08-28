#!/bin/bash
# Pobranie GLM-5.3-Flash (FP8, ~306 GiB, 62 shardy) na GX10.
#
# Uruchomienie odporne na rozlaczenie SSH (pobieranie trwa kilkanascie godzin):
#   setsid nohup deploy/gb10-glm53/download_glm53.sh < /dev/null > /dev/null 2>&1 &
# Podglad:  tail -f ~/glm53-run/dl-ckpt.log
#
# JEDEN strumien + twardy limit pasma. Domowy router w tej sieci pada przy
# rownoleglych strumieniach HTTPS (przepelnia conntrack -> "No route to host"
# na calej LAN, przy zywym pingu). To NIE jest przesadna ostroznosc: 2 workery
# `hf download` polozyly siec po kilku minutach, a `docker pull` po ~10.
# Sprawdzona recepta to `curl -C - --limit-rate 6M` na plik, sekwencyjnie.
#
# Skrypt jest WZNAWIALNY: kazdy plik jest weryfikowany po rozmiarze z API HF,
# a `curl -C -` dociaga urwane. Bezpiecznie uruchomic ponownie po awarii.

set -u

REPO="${REPO:-zai-org/GLM-5.3-Flash}"
DEST="${DEST:-/mnt/adata/ai/models/glm-5.3-flash}"
RATE="${RATE:-6M}"
LOG="${LOG:-$HOME/glm53-run/dl-ckpt.log}"
LIST="${LIST:-$HOME/glm53-run/glm53_files.txt}"

# PULAPKA (zaliczona 2026-08-28): /mnt/adata/ai/models jest root-owned, wiec
# `mkdir` tutaj konczy sie "Permission denied", a skrypt bez tej kontroli
# leci dalej i pobiera w prozne. sudo wymaga hasla, ale docker jest bez sudo -
# katalogi tworzy sie kontenerem:
#   docker run --rm -v /mnt/adata/ai/models:/m alpine:latest sh -c \
#     'mkdir -p /m/glm-5.3-flash && chown 1000:1000 /m/glm-5.3-flash'
# Uwaga: NIE dodawac tu `set -e` - w petli ponowien jest `[ ... ] && break`,
# ktore przy nieudanej probie zwraca 1 i wywalilo by caly skrypt.
mkdir -p "$DEST" "${DEST}-cache" "${DEST}-packs" "$(dirname "$LOG")" || {
  echo "!!! nie moge utworzyc $DEST - patrz uwaga o uprawnieniach wyzej"; exit 1; }

# Lista plikow z rozmiarami prosto z API (jeden lekki request) - rozmiar jest
# jedynym kryterium kompletnosci, wiec musi pochodzic ze zrodla, nie z zalozen.
if [ ! -s "$LIST" ]; then
  curl -sL "https://huggingface.co/api/models/$REPO?blobs=true" \
    | python3 -c '
import json,sys
m=json.load(sys.stdin)
rows=[(s["rfilename"], s.get("size") or 0) for s in m["siblings"]]
small=[r for r in rows if not r[0].endswith(".safetensors")]
big=sorted([r for r in rows if r[0].endswith(".safetensors")])
for n,s in small+big:            # male pliki najpierw: config/tokenizer
    print(n, s)                  # pozwalaja sprawdzic model zanim zejda shardy
' > "$LIST"
fi

exec >> "$LOG" 2>&1
echo "=== START $(date -Is)  repo=$REPO -> $DEST"
echo "=== wolne miejsce: $(df -h "$DEST" | tail -1 | awk '{print $4}')"

n=0; total=$(wc -l < "$LIST")
while read -r name size; do
  n=$((n+1))
  out="$DEST/$name"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
    echo "[$n/$total] OK (jest) $name"; continue
  fi
  echo "$(date +%H:%M:%S) [$n/$total] $name ($((size/1048576)) MB)"
  for try in 1 2 3 4 5; do
    curl -sL -C - --limit-rate "$RATE" --retry 20 --retry-delay 15 --retry-all-errors \
         -o "$out" "https://huggingface.co/$REPO/resolve/main/$name"
    [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" = "$size" ] && break
    echo "   rozmiar sie nie zgadza, ponawiam ($try/5)"; sleep 20
  done
  [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" = "$size" ] || {
    echo "!!! NIEUDANE: $name"; exit 1; }
done < "$LIST"

echo "=== CHECKPOINT KOMPLETNY $(date -Is)"
du -sh "$DEST"
