#!/bin/bash
# Czarna skrzynka bring-upu: zapisuje przebieg BOOTU lokalnie na GX10, co 2 s,
# zeby dane przetrwaly utrate SSH.
#
# Powod: przy tym bring-upie pamiec potrafi spasc z 93 GB do 13 GB w mniej niz
# 10 s. Zanim zdazy sie odczytac `docker logs` przez SSH, maszyna juz nie
# wpuszcza na sesje - i diagnostyka przepada razem z proba. Ten skrypt jest
# rezydentny, pisze do pliku i nie forkuje w petli poza jednym `docker logs`.
#
#   setsid ./blackbox.sh glm53-moet > /dev/null 2>&1 &
#   # po wszystkim:  cat ~/glm53-run/blackbox.log
set -u
C="${1:-glm53-moet}"
OUT="${OUT:-$HOME/glm53-run/blackbox.log}"
: > "$OUT"
echo "=== blackbox start $(date -Is) container=$C" >> "$OUT"
prev=""
while :; do
  a=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  # ostatnia linia DBG - tylko gdy sie zmienila, zeby plik nie spuchl
  cur=$(docker logs --tail 40 "$C" 2>&1 | grep -E "DBG |planes built|BUILD" | tail -1 | sed 's/.*DBG/DBG/' | cut -c1-140)
  st=$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null || echo gone)
  echo "$(date +%H:%M:%S) avail=${a}G state=$st | $cur" >> "$OUT"
  [ "$st" = "gone" ] && { echo "=== kontener zniknal, koniec" >> "$OUT"; exit 0; }
  if [ "$st" = "exited" ]; then
    echo "=== kontener zakonczony; zrzucam pelny log DBG" >> "$OUT"
    docker logs "$C" 2>&1 | grep -E "DBG |planes built|BUILD|Traceback|Error" \
      | sed 's/.*DBG/DBG/' | cut -c1-160 >> "$OUT"
    exit 0
  fi
  sleep 2
done
