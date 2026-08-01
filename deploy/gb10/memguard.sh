#!/bin/bash
# Straznik boota vLLM-Moet na GX10 — wersja 2.
#
# WERSJA 1 PRZESPALA 40-MINUTOWY ZAKLESZCZ (2026-08-01): pilnowala "RAM + swap",
# a swapu bylo 137 GiB wolnego, wiec prog sie nie spelnil — mimo ze maszyna od
# pol godziny stala z PSI full 75% i zero postepu. Wniosek: przy thrashingu na
# stronach PLIKOWYCH ilosc wolnej pamieci NIE jest sygnalem. Sa nim:
#
#   1. PSI full (/proc/pressure/memory) utrzymujace sie wysoko — system stoi
#      zablokowany na pamieci, niezaleznie od tego ile "wolnego" raportuje free
#   2. BRAK POSTEPU w logu modelu przez N minut — bo o to naprawde chodzi
#   3. wyczerpanie RAM-u ORAZ swapu naraz (stary warunek, jako ostatnia deska)
#
# Uruchomienie:  nohup ~/moet-preflight/memguard.sh deepseek-moet &
# Podglad:       tail -f ~/moet-preflight/memguard.log

set -u
C=${1:-deepseek-moet}
THRESH_GIB=${2:-8}          # podloga RAM+swap
PSI_LIMIT=${3:-60}          # % PSI full avg60
STALL_MIN=${4:-8}           # minut bez postepu w logu
LOG="$HOME/moet-preflight/memguard.log"

THRESH_KB=$((THRESH_GIB * 1024 * 1024))
TICK=15
STALL_TICKS=$((STALL_MIN * 60 / TICK))

echo "=== memguard v2 start $(date -Is): $C | podloga=${THRESH_GIB}G | PSI>${PSI_LIMIT}% | stall=${STALL_MIN}min ===" >> "$LOG"

stop_now() {
    { echo "$(date -Is) !!! $1"; echo "$(date -Is) !!! zatrzymuje $C"; } >> "$LOG"
    docker stop -t 15 "$C" >> "$LOG" 2>&1 \
        || { echo "$(date -Is) docker stop nie zadzialal — SIGKILL na proces" >> "$LOG"
             sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
             sudo pkill -9 -f "bin/vllm serve" 2>/dev/null; }
    echo "$(date -Is) ostatnie 15 linii modelu:" >> "$LOG"
    docker logs --tail 15 "$C" >> "$LOG" 2>&1
    exit 1
}

low_streak=0; psi_streak=0; stall_ticks=0; prev_mark=""
while true; do
    docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q true || {
        echo "$(date -Is) kontener $C nie dziala — koniec czuwania" >> "$LOG"; exit 0; }

    eval "$(awk '/^MemAvailable:/{a=$2} /^MemFree:/{f=$2} /^Cached:/{c=$2}
                 /^SwapFree:/{s=$2} /^SwapTotal:/{st=$2}
                 END{printf "avail_kb=%d free_kb=%d cached_kb=%d swf_kb=%d swt_kb=%d",a,f,c,s,st}' \
             /proc/meminfo)"
    psi=$(awk '/^full/{for(i=1;i<=NF;i++) if($i ~ /^avg60=/){split($i,x,"="); printf "%d", x[2]}}' \
          /proc/pressure/memory 2>/dev/null)
    psi=${psi:-0}
    # BOOT-ONLY guard: gdy serwer wstal, detektor "braku postepu" nie ma sensu
    # (znaczniki bootu przestaja sie zmieniac podczas serwowania) - konczymy.
    # Lekcja 2026-08-01: guard zostawiony na serwujacym procesie ubil go po
    # 20 min "stalla", zrywajac sesje uzytkownika (Bad Gateway na proxy).
    # --since od ostatniego startu kontenera: docker logs AKUMULUJE sie miedzy
    # restartami tego samego kontenera, a stary "startup complete" nie moze
    # konczyc czuwania nad swiezym bootem.
    STARTED=$(docker inspect -f "{{.State.StartedAt}}" "$C" 2>/dev/null)
    if docker logs --since "$STARTED" "$C" 2>&1 | grep -q "Application startup complete"; then
        echo "$(date -Is) serwer wstal - koniec czuwania (guard jest boot-only)" >> "$LOG"
        exit 0
    fi

    # znacznik postepu = ostatnia zbudowana warstwa + ostatni shard
    mark=$(docker logs "$C" 2>&1 | grep -oE "layer [0-9]+ planes built|Completed \| [0-9]+/[0-9]+" | tail -2 | tr '\n' ' ')

    echo "$(date -Is) avail=$((avail_kb/1024/1024))G cache=$((cached_kb/1024/1024))G swap_uzyty=$(( (swt_kb-swf_kb)/1024/1024 ))G PSI=${psi}% | ${mark}" >> "$LOG"

    # --- 1. PSI: system stoi na pamieci
    if [ "$psi" -ge "$PSI_LIMIT" ]; then
        psi_streak=$((psi_streak + 1))
        [ "$psi_streak" -ge 8 ] && stop_now "PSI full=${psi}% przez ~2 min — system stoi na pamieci"
    else
        psi_streak=0
    fi

    # --- 2. brak postepu
    if [ "$mark" = "$prev_mark" ] && [ -n "$mark" ]; then
        stall_ticks=$((stall_ticks + 1))
        [ "$stall_ticks" -ge "$STALL_TICKS" ] && stop_now "brak postepu przez ${STALL_MIN} min (${mark})"
    else
        stall_ticks=0; prev_mark="$mark"
    fi

    # --- 3. ostatnia deska: RAM i swap naraz
    if [ $((avail_kb + swf_kb)) -lt "$THRESH_KB" ]; then
        low_streak=$((low_streak + 1))
        [ "$low_streak" -ge 2 ] && stop_now "RAM+swap ponizej ${THRESH_GIB} GiB"
    else
        low_streak=0
    fi

    sleep "$TICK"
done
