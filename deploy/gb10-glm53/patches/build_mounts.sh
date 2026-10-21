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
NAMES="moe_w2_cubit moe_w2_delta llm_base_proposer v2_model_runner cuda_platform mamba_utils scheduler loader_utils cudagraph_utils kda_layer mamba_hybrid"
src_of() {
  case "$1" in
    scheduler)         echo v1/core/sched/scheduler.py ;;
    moe_w2_cubit)      echo model_executor/layers/quantization/utils/moe_w2_cubit.py ;;
    moe_w2_delta)      echo model_executor/layers/quantization/utils/moe_w2_delta.py ;;
    llm_base_proposer) echo v1/spec_decode/llm_base_proposer.py ;;
    v2_model_runner)   echo v1/worker/gpu/model_runner.py ;;
    cuda_platform)     echo platforms/cuda.py ;;
    mamba_utils)       echo model_executor/layers/mamba/mamba_utils.py ;;
    loader_utils)      echo model_executor/model_loader/utils.py ;;
    cudagraph_utils)   echo v1/worker/gpu/cudagraph_utils.py ;;
    kda_layer)         echo models/glm5next/nvidia/kda.py ;;
    mamba_hybrid)      echo v1/worker/gpu/model_states/mamba_hybrid.py ;;
  esac
}

mkdir -p "$OUT"
echo "== copying the originals out of $IMAGE -> $OUT"
CID=$(docker create "$IMAGE")
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT
for n in $NAMES; do
  docker cp -q "$CID:$V/$(src_of "$n")" "$OUT/$n.orig.py"
done
# Patch 14: the image's cubin dir with the decode moe_w2_mm / moe_w4_mm
# swapped for the looped builds (kernels/cubins-sm120-loop).
rm -rf "$OUT/cubit" "$OUT/cubit.orig"
docker cp -q "$CID:/cubit-share" "$OUT/cubit.orig"
cp -a "$OUT/cubit.orig" "$OUT/cubit"
docker rm "$CID" >/dev/null
trap - EXIT
LOOP_DIR="$HERE/../../../kernels/cubins-sm120-loop"
for f in "$LOOP_DIR"/*.cubin; do
  cp "$f" "$OUT/cubit/"
done
echo "== cubit: $(ls "$OUT/cubit" | wc -l) cubins, looped: $(cd "$LOOP_DIR" && ls *.cubin | tr '\n' ' ')"

apply() {  # apply <patch script> [mode] <target>
  local script=$1; shift
  echo "-- $script -> $(basename "${@: -1}")"
  python3 "$HERE/$script" "$@"
}

echo "== applying patches"
# moe_w2_cubit.py: ORDER MATTERS - patch_store_skip_read anchors in code that
# patch_partial_fp4 leaves behind. patch_swiglu is diagnostics only (the
# hypothesis was rejected) but stays in so the result matches production.
cp "$OUT/moe_w2_cubit.orig.py" "$OUT/moe_w2_cubit.py"
apply patch_swiglu.py           "$OUT/moe_w2_cubit.py"
apply patch_partial_fp4.py      "$OUT/moe_w2_cubit.py"
apply patch_store_skip_read.py  "$OUT/moe_w2_cubit.py"
apply patch_moe_loop.py         "$OUT/moe_w2_cubit.py"
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
cp "$OUT/scheduler.orig.py" "$OUT/scheduler.py"
apply patch_mamba_align_block.py "$OUT/scheduler.py"
apply patch_adaptive_k.py       "$OUT/scheduler.py"
cp "$OUT/cudagraph_utils.orig.py" "$OUT/cudagraph_utils.py"
apply patch_adaptive_graphs.py  "$OUT/cudagraph_utils.py"
cp "$OUT/kda_layer.orig.py" "$OUT/kda_layer.py"
apply patch_kda_commit.py layer "$OUT/kda_layer.py"
cp "$OUT/mamba_hybrid.orig.py" "$OUT/mamba_hybrid.py"
apply patch_kda_commit.py state "$OUT/mamba_hybrid.py"
cp "$OUT/loader_utils.orig.py" "$OUT/loader_utils.py"
apply patch_w8a16_loader.py     "$OUT/loader_utils.py"
# Not a patch but a new module (patch 13 imports it): copied as is.
cp "$HERE/glm53_w8a16.py" "$OUT/glm53_w8a16.py"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$OUT/glm53_w8a16.py"
# Patch 16: the KDA verify/commit module the two hooks above import.
cp "$HERE/kda_commit.py" "$OUT/kda_commit.py"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$OUT/kda_commit.py"
# Optional variant for GPTQ calibration (boot.sh XDUMP=1): production file +
# the MoE-input dump hook. Never mounted unless asked for.
cp "$OUT/moe_w2_cubit.py" "$OUT/moe_w2_cubit.xdump.py"
apply patch_xdump.py            "$OUT/moe_w2_cubit.xdump.py"

echo "== result (md5, compare with patches/README.md)"
for n in $NAMES moe_w2_cubit.xdump glm53_w8a16 kda_commit; do
  f="$OUT/$n.py"
  [ "$n" = moe_w2_cubit.xdump ] && f="$OUT/moe_w2_cubit.xdump.py"
  printf '  %-24s %s\n' "$(basename "$f")" "$(md5sum < "$f" | cut -c1-8)"
done
