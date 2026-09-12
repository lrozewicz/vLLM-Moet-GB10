#!/bin/bash
# HISTORY (bring-up, 2026-08-29): this is the script that ran the very first
# boot on the reference box, with the bring-up compose file and memguard.sh.
# It is kept as a record. For a new install use deploy/gb10-glm53/boot.sh,
# described in docs/models/glm-5.3-flash.md.
#
# Pierwszy boot GLM-5.3-Flash: czeka az checkpoint bedzie kompletny, zwalnia
# pamiec, uzbraja straznika i startuje serwer.
#
#   setsid ~/glm53-run/first-boot.sh </dev/null >/dev/null 2>&1 & disown
#   tail -f ~/glm53-run/first-boot.log
#
# Dlaczego to skrypt, a nie recznie: pobieranie konczy sie w srodku nocy, a
# pierwszy boot trwa ~35 min (kwantyzacja 311e9 parametrow ekspertow) i jest
# najbardziej ryzykownym momentem calego stacku - to wtedy pamiec zunifikowana
# potrafi wejsc w zakleszczenie, z ktorego nie ma powrotu bez twardego resetu.
# Dlatego straznik jest uzbrajany ZAWSZE, zanim ktokolwiek na to spojrzy.

set -u
RUNDIR="${RUNDIR:-$HOME/glm53-run}"
REPO="${REPO:-$RUNDIR/repo}"
DEST="${DEST:-/mnt/adata/ai/models/glm-5.3-flash}"
LOG="${LOG:-$RUNDIR/first-boot.log}"
COMPOSE="$REPO/deploy/gb10-glm53/docker-compose.glm-5.3-flash.yml"
CONTAINER=glm53-moet
# Klucz API: tylko z otoczenia, nigdy wpisany w plik.
: "${VLLM_API_KEY:?export VLLM_API_KEY first}"
export VLLM_API_KEY

exec >> "$LOG" 2>&1
echo "=== START $(date -Is)"

# --- 1. czekaj na kompletny checkpoint -------------------------------------
# Kryterium to marker .ok na kazdym shardzie (sha256 sie zgodzil), a nie sam
# rozmiar katalogu: niedokonczony shard tez zajmuje miejsce.
want=62
while :; do
  have=$(ls "$DEST"/model-*.safetensors.ok 2>/dev/null | wc -l)
  [ "$have" -ge "$want" ] && break
  if ! pgrep -f 'glm53-run/(run-dl|dl-ckpt)[.]sh' > /dev/null; then
    echo "$(date -Is) !!! pobieranie NIE dziala, a shardow jest $have/$want - przerywam"
    exit 1
  fi
  echo "$(date +%H:%M) czekam na checkpoint: $have/$want shardow"
  sleep 300
done
echo "$(date -Is) checkpoint kompletny ($have/$want), startuje boot"

# --- 2. zwolnij pamiec ------------------------------------------------------
# 109 z 121 GiB trzyma serwer Qwena. Autoheal MUSI pojsc pierwszy, inaczej
# natychmiast podniesie to, co przed chwila zatrzymalismy.
for c in autoheal-qwen38 qwen38-sglang; do
  if docker ps --format '{{.Names}}' | grep -qx "$c"; then
    echo "$(date +%H:%M) zatrzymuje $c (przywrocenie: docker start $c)"
    docker stop "$c" > /dev/null
    echo "$c" >> "$RUNDIR/.stopped-for-glm53"
  fi
done

# drop_caches i dirty_bytes wymagaja roota; sudo chce hasla, ale docker nie -
# wiec przez uprzywilejowany kontener. dirty_bytes: zapis cache planow na
# wolniejszy dysk nie moze zalegac w RAM-ie (zaliczone na DS4).
docker run --rm --privileged --pid=host alpine:latest sh -c \
  'sync; echo 3 > /proc/sys/vm/drop_caches; echo 2147483648 > /proc/sys/vm/dirty_bytes' \
  > /dev/null 2>&1 && echo "$(date +%H:%M) cache zrzucony, dirty_bytes=2G"
free -g | head -2

# --- 3. start + straznik ----------------------------------------------------
echo "$(date +%H:%M) docker compose up"
docker compose -f "$COMPOSE" up -d || { echo "!!! compose up nieudany"; exit 1; }

# Straznik ubija kontener przy utrzymujacym sie PSI zamiast pozwolic maszynie
# wejsc w zakleszczenie. Uzbrajany PO starcie kontenera (v1 przespala 40 min
# zakleszczenia, bo pilnowala wolnej pamieci, a nie PSI i postepu w logu).
nohup "$REPO/deploy/gb10/memguard.sh" "$CONTAINER" 8 60 20 \
  > "$RUNDIR/memguard.log" 2>&1 &
echo "$(date -Is) straznik uzbrojony (PSI 60%, brak postepu 20 min)"

# --- 4. postep --------------------------------------------------------------
echo "$(date -Is) sledze boot; pelny log: docker logs -f $CONTAINER"
docker logs -f "$CONTAINER" 2>&1 \
  | grep --line-buffered -E "moe_w2|planes|Completed|startup complete|Error|error|Traceback|OOM|CUDA" \
  | while read -r line; do echo "$(date +%H:%M:%S) $line"; done
