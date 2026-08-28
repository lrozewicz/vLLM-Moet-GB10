#!/bin/bash
# Nadzorca pobierania: wznawia download_glm53.sh po kazdym zerwaniu.
#
#   setsid ~/glm53-run/run-dl.sh </dev/null >/dev/null 2>&1 & disown
#
# Przetrwa rozlaczenie SSH i wylaczenie maszyny, z ktorej to uruchomiono.
# Wznawianie jest bezpieczne: kazdy plik jest weryfikowany rozmiarem i sha256,
# uszkodzone sa kasowane i pobierane od zera. Blokada jednej instancji siedzi
# w samym download_glm53.sh (flock) - tutaj jej NIE powielamy, bo dwa flocki
# na ten sam plik konczylyby sie natychmiastowym, cichym wyjsciem.
DL="${DL:-$HOME/glm53-run/dl-ckpt.sh}"
LOG="${LOG:-$HOME/glm53-run/dl-ckpt.log}"

exec >> "$LOG" 2>&1
i=0
until "$DL"; do
  i=$((i+1))
  echo "$(date -Is) === proba $i nieudana, wznawiam za 120 s ==="
  sleep 120
done
echo "$(date -Is) === POBIERANIE ZAKONCZONE ==="
