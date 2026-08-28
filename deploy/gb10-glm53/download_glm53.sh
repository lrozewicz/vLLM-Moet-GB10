#!/bin/bash
# Pobranie GLM-5.3-Flash (FP8, ~306 GiB, 62 shardy) na GX10.
#
# Uruchomienie odporne na rozlaczenie SSH (pobieranie trwa kilkanascie godzin):
#   setsid ~/glm53-run/run-dl.sh </dev/null >/dev/null 2>&1 & disown
# Podglad:  tail -f ~/glm53-run/dl-ckpt.log
#
# JEDEN strumien + twardy limit pasma. Domowy router w tej sieci pada przy
# rownoleglych strumieniach HTTPS (przepelnia conntrack -> "No route to host"
# na calej LAN). Sprawdzona recepta: `curl -C - --limit-rate 6M` na plik,
# sekwencyjnie. NIE pobieraj rownoczesnie niczego innego - dotyczy to takze
# pobierania na INNEJ maszynie w tej samej sieci (zaliczone 2026-08-28:
# rownolegle ciagniecie warstwy obrazu na pop-osie polozylo LAN na ~5 min).
#
# WERYFIKACJA: rozmiar + sha256 z metadanych LFS HuggingFace. Sam rozmiar nie
# wystarcza przy 306 GiB - a przy wznawianiu wrecz myli: `curl -C -` dokleja
# od biezacego konca pliku, wiec plik uszkodzony (np. przez dwie rownolegle
# instancje piszace w to samo miejsce - zaliczone) rosnie PONAD rozmiar
# zrodla i zadna liczba ponowien go nie naprawi. Dlatego kazdy plik o zlym
# rozmiarze jest KASOWANY i pobierany od zera, a po pobraniu liczona jest
# suma. Zweryfikowany plik dostaje marker `.ok`, zeby restart nie czytal
# 306 GiB z dysku od nowa.

set -u

REPO="${REPO:-zai-org/GLM-5.3-Flash}"
DEST="${DEST:-/mnt/adata/ai/models/glm-5.3-flash}"
RATE="${RATE:-6M}"
RUNDIR="${RUNDIR:-$HOME/glm53-run}"
LOG="${LOG:-$RUNDIR/dl-ckpt.log}"
LIST="${LIST:-$RUNDIR/glm53_files.txt}"

mkdir -p "$RUNDIR"

# Jedna instancja na raz. Bez tego dwa curl-e dopisuja do tego samego pliku.
exec 9>"$RUNDIR/.dl.lock"
flock -n 9 || { echo "$(date -Is) inna instancja juz dziala, wychodze"; exit 0; }

# PULAPKA (zaliczona 2026-08-28): /mnt/adata/ai/models jest root-owned, wiec
# `mkdir` konczy sie "Permission denied". sudo wymaga hasla, ale docker jest
# bez sudo - katalogi tworzy sie kontenerem:
#   docker run --rm -v /mnt/adata/ai/models:/m alpine:latest sh -c \
#     'mkdir -p /m/glm-5.3-flash && chown 1000:1000 /m/glm-5.3-flash'
mkdir -p "$DEST" "${DEST}-cache" "${DEST}-packs" "$(dirname "$LOG")" || {
  echo "!!! nie moge utworzyc $DEST - patrz uwaga o uprawnieniach wyzej"; exit 1; }

# Lista "nazwa rozmiar sha256" prosto z API (jeden lekki request). sha256 jest
# w metadanych LFS; male pliki (bez LFS) dostaja "-" i sa sprawdzane rozmiarem.
if [ ! -s "$LIST" ]; then
  curl -sL "https://huggingface.co/api/models/$REPO?blobs=true" \
    | python3 -c '
import json,sys
m=json.load(sys.stdin)
rows=[(s["rfilename"], s.get("size") or 0,
       (s.get("lfs") or {}).get("sha256") or "-") for s in m["siblings"]]
small=[r for r in rows if not r[0].endswith(".safetensors")]
big=sorted([r for r in rows if r[0].endswith(".safetensors")])
for n,s,h in small+big:          # male pliki najpierw: pozwalaja obejrzec
    print(n, s, h)               # config/tokenizer zanim zejda shardy
' > "$LIST"
fi

exec >> "$LOG" 2>&1
echo "=== START $(date -Is)  repo=$REPO -> $DEST"
echo "=== wolne miejsce: $(df -h "$DEST" | tail -1 | awk '{print $4}')"

verify() {  # $1=plik $2=sha256 ("-" = pomin)
  [ "$2" = "-" ] && return 0
  [ "$(sha256sum "$1" | cut -d" " -f1)" = "$2" ]
}

n=0; total=$(wc -l < "$LIST")
while read -r name size sha; do
  n=$((n+1))
  out="$DEST/$name"
  mkdir -p "$(dirname "$out")"

  if [ -f "$out.ok" ] && [ "$(stat -c%s "$out" 2>/dev/null)" = "$size" ]; then
    echo "[$n/$total] OK (zweryfikowany) $name"; continue
  fi

  ok=0
  for try in 1 2 3 4 5; do
    have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    # Za duzy = uszkodzony. Wznawianie go tylko utrwala blad.
    if [ "$have" -gt "$size" ]; then
      echo "   $name: $have > $size (uszkodzony), kasuje i pobieram od zera"
      rm -f "$out"; have=0
    fi
    if [ "$have" -lt "$size" ]; then
      echo "$(date +%H:%M:%S) [$n/$total] $name ($((size/1048576)) MB, mam $((have/1048576)) MB)"
      curl -sL -C - --limit-rate "$RATE" --retry 20 --retry-delay 15 \
           --retry-all-errors -o "$out" \
           "https://huggingface.co/$REPO/resolve/main/$name"
      have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    fi
    if [ "$have" = "$size" ] && verify "$out" "$sha"; then
      touch "$out.ok"; ok=1; break
    fi
    echo "   $name: proba $try/5 nieudana (rozmiar $have/$size, suma niezgodna?)"
    rm -f "$out"          # zla suma = bajty smieciowe, nie ma czego wznawiac
    sleep 20
  done
  [ "$ok" = 1 ] || { echo "!!! NIEUDANE: $name"; exit 1; }
done < "$LIST"

echo "=== CHECKPOINT KOMPLETNY $(date -Is)"
du -sh "$DEST"
