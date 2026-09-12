#!/bin/bash
# Boot GLM-5.3-Flash on ONE GB10 box (NVIDIA DGX Spark, ASUS Ascent GX10, ...)
# in the configuration documented in docs/models/glm-5.3-flash.md.
#
#   export VLLM_API_KEY=...             # required; clients send it as Bearer
#   export DATA_DIR=/path/to/models     # where the checkpoint and caches live
#   deploy/gb10-glm53/boot.sh [timeout_minutes]        # default 40
#
# Every setting below is an environment variable whose default is the
# production value; set one to override it. Examples:
#   DELTA_GB=0 boot.sh 120        # the very first boot (builds the planes cache)
#   DFLASH=0 boot.sh              # MTP k=4 instead of DFlash2 (two long streams)
#   XDUMP=1 boot.sh               # arm the GPTQ calibration hook (see the doc)
#   DRY_RUN=1 boot.sh             # print the resolved compose file and exit
#
# What it does, in order. Each step is here because skipping it cost us a
# boot, or the whole box, at least once:
#   1. arms hostguard.py - on unified memory a runaway allocation wedges the
#      box within seconds, and docker's mem_limit does not see GPU memory
#   2. removes the previous container
#   3. drops the page cache of the checkpoint, the caches and (optionally) the
#      docker image layers - vLLM's start-up check counts page cache as used
#      memory on GB10, so without this every second boot fails
#   4. waits until MemFree clears MEM_MIN
#   5. starts the container (adding the DFlash2-only flag) and waits for
#      /health, then prints the KV numbers worth checking after every boot
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
TMO_MIN="${1:-40}"
CONTAINER=glm53-moet
PROJECT=glm53
die() { echo "!!! $*" >&2; exit 1; }
say() { echo "[$(date +%T)] $*"; }

[ -n "${VLLM_API_KEY:-}" ] || die "export VLLM_API_KEY first (any secret string; clients send it as 'Authorization: Bearer ...')"

# ---------------------------------------------------------------- paths
# Defaults are the reference box layout; DATA_DIR moves all three at once.
export DATA_DIR="${DATA_DIR:-/mnt/adata/ai/models}"
export MODEL_DIR="${MODEL_DIR:-$DATA_DIR/glm-5.3-flash}"        # FP8 checkpoint, 306 GiB
export CACHE_DIR="${CACHE_DIR:-$DATA_DIR/glm-5.3-flash-cache}"  # torch.compile + FlashInfer JIT, ~4 GB
export PACKS_DIR="${PACKS_DIR:-$DATA_DIR/glm-5.3-flash-packs}"  # planes caches, 227 GB (+48 GB GPTQ)
export NVME_DIR="${NVME_DIR:-/var/lib/glm53-packs-nvme}"        # FP4 store, 146 GB - INTERNAL NVMe
export RUN_DIR="${RUN_DIR:-$HOME/glm53-run}"                    # logs + temporary compose file
export PATCH_DIR="${PATCH_DIR:-$RUN_DIR/patch}"                 # output of patches/build_mounts.sh
mkdir -p "$RUN_DIR"

# ------------------------------------------------------- configuration
# Planes: the GPTQ-on-code variant when it has been built, else the RTN cache.
# (-L: on the reference box it is a symlink that only resolves inside the
# container, because the GPTQ planes live on the NVMe mounted at /packs-nvme.)
if [ -z "${PLANES_DIR:-}" ]; then
  if [ -L "$PACKS_DIR/planes-cache-gptq-code" ] || [ -d "$PACKS_DIR/planes-cache-gptq-code" ]; then
    PLANES_DIR=planes-cache-gptq-code
  else
    PLANES_DIR=planes-cache-fp4
  fi
fi
export PLANES_DIR
export USE_V2="${USE_V2:-1}"            # V2 model runner (the FP4-tier hooks are ported to it)
export DELTA_GB="${DELTA_GB:-2}"        # FP4 delta pool, GiB (0 = bare 2-bit)
export GATE="${GATE:-0}"                # confidence gate: needs ~12.8 GiB of pool, does not fit
export PROMO_HYST="${PROMO_HYST:-1.25}" # promotion hysteresis, stops pool ping-pong (+15% decode)
export BF12="${BF12:-1}"                # lossless 12-bit packing of the BF16 remainder (-2.4 GiB)
export MM_IMG="${MM_IMG:-0}" MM_VID="${MM_VID:-0}"  # 1 = keep the vision tower's profiling budget
export MAX_LEN="${MAX_LEN:-204800}"     # context window
export UTIL="${UTIL:-0.94}"             # gpu-memory-utilization
export MNBT="${MNBT:-2048}"             # max-num-batched-tokens (prefill chunk)
export CG_EST="${CG_EST:-0}"            # do not reserve ~4.9 GiB for the CUDA-graph estimate
export MAX_LOGPROBS="${MAX_LOGPROBS:-20}"  # 64 for tools/glm53/kld_glm.py
MEM_MIN="${MEM_MIN:-105.0}"             # GiB of MemFree required before starting

export DFLASH="${DFLASH:-1}"
if [ "$DFLASH" = 1 ]; then
  # DFlash2 block diffusion drafter: one pass proposes a block of 8, so k=7.
  export IMAGE="${IMAGE:-vllm-moet-glm53:dflash2}"
  export SPEC_METHOD=dflash MTP_K="${MTP_K:-7}"
  export SPEC_EXTRA=',"model":"/models/dflash2-draft"'
  export DRAFT_DIR="${DRAFT_DIR:-$RUN_DIR/dflash2-draft}"      # incoai/GLM-5.3-Flash-DFlash2
  [ -f "$DRAFT_DIR/model.safetensors" ] || die "no drafter in $DRAFT_DIR (see the doc: hf download incoai/GLM-5.3-Flash-DFlash2)"
else
  # The model's own MTP head (a full MoE layer): slower, but pins fewer KV
  # blocks per request, so two long streams fit. Needs layer 45 in the planes.
  export IMAGE="${IMAGE:-vllm-moet-glm53:combined}"
  export SPEC_METHOD=mtp MTP_K="${MTP_K:-4}" SPEC_EXTRA=""
fi

# Patched vLLM files, mounted read-only over the image (patches/README.md).
# MOUNT_PATCHES=0 mounts the image's own files instead (the .orig.py copies
# build_mounts.sh saved) - never nothing: the compose file's /dev/null
# default would put an EMPTY module in place of each file.
if [ "${MOUNT_PATCHES:-1}" = 1 ]; then SUF=.py; else SUF=.orig.py; fi
CUBIT=moe_w2_cubit$SUF
if [ "${XDUMP:-0}" = 1 ]; then
  CUBIT=moe_w2_cubit.xdump.py
  export XDUMP_LAYERS="${XDUMP_LAYERS:-$(seq -s, 0 41)}"      # layer keys 0-41 = layers 3-44
  export XDUMP_DIR="${XDUMP_DIR:-/packs-nvme/xdump}"          # container path; ARM file gates it
fi
# PATCH_SCHED=0 switches patch 12 (prefill chunks aligned to the KDA state
# block) off on its own, for an A/B of just that patch. Without it long agent
# sessions can restore a KDA state from the wrong position (gibberish turns).
SCHED=scheduler$SUF
[ "${PATCH_SCHED:-1}" = 1 ] || SCHED=scheduler.orig.py
for f in "$CUBIT" moe_w2_delta$SUF llm_base_proposer$SUF v2_model_runner$SUF cuda_platform$SUF mamba_utils$SUF "$SCHED"; do
  [ -f "$PATCH_DIR/$f" ] || die "missing $PATCH_DIR/$f - run deploy/gb10-glm53/patches/build_mounts.sh"
done
export SWIGLU_MOUNT="$PATCH_DIR/$CUBIT"
export DELTA_MOUNT="$PATCH_DIR/moe_w2_delta$SUF"
export PROPOSER_MOUNT="$PATCH_DIR/llm_base_proposer$SUF"
export V2RUNNER_MOUNT="$PATCH_DIR/v2_model_runner$SUF"
export CUDAPLAT_MOUNT="$PATCH_DIR/cuda_platform$SUF"
export MAMBAUTILS_MOUNT="$PATCH_DIR/mamba_utils$SUF"
export SCHED_MOUNT="$PATCH_DIR/$SCHED"

# The DFlash2 drafter's sliding-window layers have no fp8_ds_mla backend on
# sm_121, so their KV must be skipped from quantization. The flag cannot live
# in the compose file as a variable: an empty value parses to [""], which is a
# truthy list and switches the padded-spec logic on for every layer.
CF="$HERE/docker-compose.quality-fp4.yml"
if [ "$DFLASH" = 1 ]; then
  CF2="$RUN_DIR/compose-dflash.yml"
  sed 's|^\(\s*\)- "--tensor-parallel-size=1"|\1- "--tensor-parallel-size=1"\n\1- "--kv-cache-dtype-skip-layers=sliding_window"|' "$HERE/docker-compose.quality-fp4.yml" > "$CF2"
  grep -q 'skip-layers=sliding_window' "$CF2" || die "could not insert --kv-cache-dtype-skip-layers"
  CF="$CF2"
fi

say "image $IMAGE, spec $SPEC_METHOD k=$MTP_K, planes $PLANES_DIR, pool ${DELTA_GB} GiB, context $MAX_LEN, util $UTIL"
if [ "${DRY_RUN:-0}" = 1 ]; then
  docker compose -p "$PROJECT" -f "$CF" config
  exit 0
fi
[ -e "$PACKS_DIR/$PLANES_DIR" ] || [ -L "$PACKS_DIR/$PLANES_DIR" ] || \
  say "NOTE: $PACKS_DIR/$PLANES_DIR does not exist yet - this boot quantizes the checkpoint (~35 min) and writes it"

# ----------------------------------------------------------- 1. guard
if ! pgrep -f "[h]ostguard.py" >/dev/null; then
  setsid nohup python3 "$HERE/hostguard.py" "$CONTAINER" "${GUARD_GIB:-1.5}" 2 \
    </dev/null >"$RUN_DIR/hostguard.log" 2>&1 &
  sleep 2
  pgrep -f "[h]ostguard.py" >/dev/null || die "hostguard did not start, see $RUN_DIR/hostguard.log"
  say "hostguard armed (kills $CONTAINER below ${GUARD_GIB:-1.5} GiB available)"
fi

# ----------------------------------------------- 2. previous container
docker compose -p "$PROJECT" -f "$CF" down >/dev/null 2>&1
docker rm -f "$CONTAINER" >/dev/null 2>&1
sync

# ------------------------------------------------------ 3. page cache
python3 "$HERE/drop_page_cache.py" "$MODEL_DIR" "$CACHE_DIR" "$PACKS_DIR" "$NVME_DIR"
if [ "${DROP_DOCKER_LAYERS:-1}" = 1 ]; then
  docker run --rm --entrypoint python3 -v /var/lib/docker:/d:ro -v "$HERE":/s:ro "$IMAGE" \
    /s/drop_page_cache.py --min-size 1048576 /d/overlay2 /d/containers /d/image /d/volumes /d/buildkit \
    | sed 's/^/  docker layers: /'
fi

# ---------------------------------------------------- 4. memory gate
memfree() { awk '/^MemFree:/ {printf "%.2f", $2 / 1048576}' /proc/meminfo; }
ok=0
for i in $(seq 1 40); do
  if awk -v a="$(memfree)" -v b="$MEM_MIN" 'BEGIN { exit !(a > b) }'; then ok=1; break; fi
  [ "$i" = 1 ] && say "waiting for memory: MemFree $(memfree) GiB < $MEM_MIN"
  sleep 15
done
[ "$ok" = 1 ] || die "MemFree $(memfree) GiB still below $MEM_MIN after 10 min - stop other GPU/memory users first"
say "MemFree $(memfree) GiB - starting"

# ---------------------------------------------------- 5. start + wait
docker compose -p "$PROJECT" -f "$CF" up -d || die "docker compose up failed"
for i in $(seq 1 $((TMO_MIN * 6))); do
  if curl -s -o /dev/null -w '%{http_code}' localhost:8000/health 2>/dev/null | grep -q 200; then
    say "READY after $((i * 10)) s"
    docker logs "$CONTAINER" 2>&1 | grep -aE "Using V2 Model Runner|Available KV cache|GPU KV cache size|LOADER-SKIP" | tail -6
    curl -s localhost:8000/metrics | grep '^vllm:cache_config_info' | tr ',' '\n' | grep -E 'num_gpu_blocks|block_size'
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    say "CONTAINER EXITED after $((i * 10)) s:"
    docker logs "$CONTAINER" 2>&1 | grep -aE "Error|Traceback|assert|larger than available KV" | tail -20
    exit 1
  fi
  sleep 10
  [ $((i % 6)) -eq 0 ] && say "...$((i * 10)) s: $(docker logs "$CONTAINER" 2>&1 | tail -1 | cut -c1-140)"
done
# Do NOT restart on a timeout: a slow disk only makes loading slower, and a
# restart throws away everything read so far. The container keeps loading.
say "TIMEOUT after $TMO_MIN min - the container is still loading; follow: docker logs -f $CONTAINER"
exit 1
