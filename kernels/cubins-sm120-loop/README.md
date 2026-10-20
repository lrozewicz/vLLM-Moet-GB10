# Looped decode cubins (patch 14)

`moe_w2_mm` and `moe_w4_mm` for K = 4096 and 2048 (GLM-5.3-Flash's w13 and
w2 contractions) with a grid-stride loop over the pair table, for
`deploy/gb10-glm53/patches/patch_moe_loop.py`. `build_mounts.sh` copies them
over the image's `/cubit-share` into `$PATCH_DIR/cubit`, which `boot.sh`
mounts.

**Why.** The decode launch has `grid.y` = the moe_align table length (64 at
T = 8), but a tier only has its own live pairs in it. The rest are CTAs that
load their desc and EXIT: ~0.5 us (w2) / ~0.8 us (w4) per empty pair per
launch on GB10, 4-5 ms per decode step. Patch 14 compacts each tier's live
pairs to the front of its table, and these kernels, launched with a small
fixed `grid.y` (16 for w2, 12 for w4), walk `y = ctaid.y + k * y_stride`
and stop at the first empty entry.

**ABI.** Two u32 parameters after `n_kb128`: `n_pairs` (the table length) and
`y_stride` (the launch `grid.y`). `n_pairs = y_stride = grid.y` is exactly
the old one-pair-per-CTA launch, so an uncompacted table stays valid. The
unpatched launcher passes only five parameters and must NOT be paired with
these cubins (`boot.sh MOUNT_PATCHES=0` mounts the image's own cubins).

**Measured on GB10** (`tools/glm53/loop_check.py`, `variant_time.py`, server
stopped): bit-identical output to the shipped cubins for every shape, live
pair count (1-64) and grid; at equal work (no compaction) the same speed
(+-1%); with compaction 10-60 us less per launch. End to end: code step
140.6 -> 133 ms (with the patch-14 glue fusion 131.5 ms), 39.2 -> 42.2 tok/s.

## How they were built

The body of each kernel is the shipped cubin's, instruction for instruction
and control code for control code; only the loop instructions are added
(`kernels/gen/sched_loopify.py` has the details). Letting cubit schedule a
looped version of the generator output instead changed the scoreboard
allocation of the whole K-march and cost ~20% - do not rebuild them that way.

1. cubit at `1f7a5aa` (the `5912400` pin in `kernels/MANIFEST.md` no longer
   exists upstream) with `cubit-local.patch`:
   - QMMA-containing kernels lost their Mercury (capmerc) sections even with an
     explicit `--mercury-stub`; such a cubin does not load on sm_121
     (`CUresult 209`). The patch honours the stub.
   - `CUBIT_SMEM_CAP=0` drops the sm_103a 1 KiB reserved-smem cap, as the old
     cubit did for sm_120. It does not change the speed; it only keeps the ELF
     closer to the shipped one.
2. Per kernel (run from the cubit checkout, which reads `tables/` relative):
   ```bash
   cubit disassemble kernels/cubins-sm120/X.cubin > X.dis
   python3 kernels/gen/sched_loopify.py X.dis kernels/sass/X.sass X.sass
   sed -i -E '/QMMA/ s/(R[0-9]+)\.(ROW|COL)/\1/g' X.sass   # see below
   CUBIT_SMEM_CAP=0 CUBIT_MERC13=0 cubit asm X.sass -o X.cubin \
       --kernel moe_w2_mm --mercury-stub kernels/sass/qmma_e4m3.merc.stub
   ```
   The current cubit refuses the `.ROW`/`.COL` operand suffixes on `QMMA.SF`
   that the old one silently dropped (its BUG-272); without them the QMMA words
   are identical.

The same cubit reproduces all 72 live shipped cubins of `cubins-sm120`
instruction for instruction from `kernels/sass` (the differences are an
equivalent `ISETP ..., RZ` vs immediate `0x0` encoding and the ELF metadata
above). The two remaining files, `moe_w2_mm_k2048` and `moe_w2_mm_mc2_k2048`,
are retired variants whose cubin does not match the saved SASS.

| file | md5 (first 8) |
|---|---|
| `moe_w2_mm_k4096_a32.cubin` | `98bbc1c0` |
| `moe_w2_mm_k2048_a32.cubin` | `972d055a` |
| `moe_w4_mm_k4096_a32.cubin` | `c2a00eeb` |
| `moe_w4_mm_k2048_a32.cubin` | `2772e35e` |
