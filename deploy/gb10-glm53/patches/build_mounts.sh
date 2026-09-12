#!/bin/bash
# Build the patched vLLM files that boot.sh bind-mounts into the container.
#
#   deploy/gb10-glm53/patches/build_mounts.sh [OUT_DIR]
#
# OUT_DIR defaults to $PATCH_DIR, then $HOME/glm53-run/patch. IMAGE defaults
# to vllm-moet-glm53:dflash2 (the files below are identical in :combined).
#
# The files are copied out of the image, patched on the host and mounted
# read-only over the originals - the image itself is never modified. That
# keeps every patch switchable for an A/B (boot.sh MOUNT_PATCHES=0) and makes
# an image update a matter of re-running this script: every patch checks its
# own marker (idempotent), validates the syntax, and fails LOUDLY when an
# anchor moved instead of silently doing nothing.
#
# Reference check: on our box the outputs are byte-identical to the files the
# production server mounts (md5 table in patches/README.md).
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-${PATCH_DIR:-$HOME/glm53-run/patch}}"
IMAGE="${IMAGE:-vllm-moet-glm53:dflash2}"
V=/usr/local/lib/python3.12/dist-packages/vllm

# name in OUT -> path inside the vLLM package
NAMES="moe_w2_cubit moe_w2_delta llm_base_proposer v2_model_runner cuda_platform mamba_utils"
src_of() {
  case "$1" in
    moe_w2_cubit)      echo model_executor/layers/quantization/utils/moe_w2_cubit.py ;;
    moe_w2_delta)      echo model_executor/layers/quantization/utils/moe_w2_delta.py ;;
    llm_base_proposer) echo v1/spec_decode/llm_base_proposer.py ;;
    v2_model_runner)   echo v1/worker/gpu/model_runner.py ;;
    cuda_platform)     echo platforms/cuda.py ;;
    mamba_utils)       echo model_executor/layers/mamba/mamba_utils.py ;;
  esac
}

mkdir -p "$OUT"
echo "== copying the originals out of $IMAGE -> $OUT"
CID=$(docker create "$IMAGE")
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT
for n in $NAMES; do
  docker cp -q "$CID:$V/$(src_of "$n")" "$OUT/$n.orig.py"
done
docker rm "$CID" >/dev/null
trap - EXIT

apply() {  # apply <patch script> <target>
  echo "-- $1 -> $(basename "$2")"
  python3 "$HERE/$1" "$2"
}

echo "== applying patches"
# moe_w2_cubit.py: ORDER MATTERS - patch_store_skip_read anchors in code that
# patch_partial_fp4 leaves behind. patch_swiglu is diagnostics only (the
# hypothesis was rejected) but stays in so the result matches production.
cp "$OUT/moe_w2_cubit.orig.py" "$OUT/moe_w2_cubit.py"
apply patch_swiglu.py           "$OUT/moe_w2_cubit.py"
apply patch_partial_fp4.py      "$OUT/moe_w2_cubit.py"
apply patch_store_skip_read.py  "$OUT/moe_w2_cubit.py"
cp "$OUT/moe_w2_delta.orig.py" "$OUT/moe_w2_delta.py"
apply patch_delta_skip.py       "$OUT/moe_w2_delta.py"
cp "$OUT/llm_base_proposer.orig.py" "$OUT/llm_base_proposer.py"
apply patch_mtp_token.py        "$OUT/llm_base_proposer.py"
cp "$OUT/v2_model_runner.orig.py" "$OUT/v2_model_runner.py"
apply patch_v2_runner.py        "$OUT/v2_model_runner.py"
cp "$OUT/cuda_platform.orig.py" "$OUT/cuda_platform.py"
apply patch_pdl_off.py          "$OUT/cuda_platform.py"
cp "$OUT/mamba_utils.orig.py" "$OUT/mamba_utils.py"
apply patch_kda_ssm_dtype.py    "$OUT/mamba_utils.py"
# Optional variant for GPTQ calibration (boot.sh XDUMP=1): production file +
# the MoE-input dump hook. Never mounted unless asked for.
cp "$OUT/moe_w2_cubit.py" "$OUT/moe_w2_cubit.xdump.py"
apply patch_xdump.py            "$OUT/moe_w2_cubit.xdump.py"

echo "== result (md5, compare with patches/README.md)"
for n in $NAMES moe_w2_cubit.xdump; do
  f="$OUT/$n.py"
  [ "$n" = moe_w2_cubit.xdump ] && f="$OUT/moe_w2_cubit.xdump.py"
  printf '  %-24s %s\n' "$(basename "$f")" "$(md5sum < "$f" | cut -c1-8)"
done
