#!/usr/bin/env python3
"""PATCH 12: align prefill chunks to the KDA (Mamba) block, not to cache_config.block_size.

SYMPTOM. In long agent sessions one turn in a few dozen produced gibberish
from its FIRST token (draft acceptance 0.0%), deterministically: replaying the
captured session turn by turn broke at the same step every time, and that
step replayed without prefix-cache hits was clean. The broken step was the
first one to restore the KDA state from the checkpoint at 43,008 tokens.

MECHANISM. In mamba_cache_mode="align" a cached Mamba block p must hold the
recurrent state after EXACTLY (p + 1) * block_size tokens - the block hash
says so. `Scheduler._mamba_block_aligned_split` guarantees it by ending prefill
chunks on block boundaries, using `self.cache_config.block_size` as the block.
That equals the Mamba block (7168 here, forced by the KDA page size) only
until `EngineCore` overwrites `cache_config.block_size` with the MINIMUM block
size of the groups that take part in prefix caching (v1/engine/core.py). The
DFlash2 drafter group has a 1024-token block (patch 8), so the scheduler
aligned chunks to 1024 while the KDA state lives in 7168-token blocks. A
chunk could then run across a 7168 boundary: the block kept the state of an
earlier chunk end (41,984 instead of 43,008) and was cached under the hash of
the boundary. Restoring it drops 1024 tokens of recurrent state -> gibberish.
(core.py already excludes the kpool tail group for exactly this reason - "would
otherwise ... desync it from mamba" - the drafter group was not excluded.)

PATCH. Use the block size of the MambaSpec KV groups for the split (cached on
first use), falling back to cache_config.block_size when there is none. Only
the split changes; the drafter keeps its small block and hashing granularity.

    python3 patch_mamba_align_block.py scheduler.py
"""
import ast
import sys

MARK = "[PATCH-MAMBA-ALIGN-BLOCK]"

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if MARK in src:
    print("already patched")
    sys.exit(0)

OLD = """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
"""
NEW = """        # [PATCH-MAMBA-ALIGN-BLOCK] The KDA/Mamba state block, not
        # cache_config.block_size: EngineCore lowers the latter to the smallest
        # prefix-caching group block (the DFlash2 drafter's 1024), which would let
        # a chunk run across a state boundary. See patch_mamba_align_block.py.
        block_size = getattr(self, "_mamba_align_block_size", None)
        if block_size is None:
            from vllm.v1.kv_cache_interface import MambaSpec as _MambaSpec

            _sizes = {
                g.kv_cache_spec.block_size
                for g in self.kv_cache_config.kv_cache_groups
                if isinstance(g.kv_cache_spec, _MambaSpec)
            }
            assert len(_sizes) <= 1, f"mixed Mamba block sizes: {_sizes}"
            block_size = _sizes.pop() if _sizes else self.cache_config.block_size
            self._mamba_align_block_size = block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
"""

n = src.count(OLD)
if n != 1:
    print(f"ANCHOR NOT FOUND EXACTLY ONCE ({n}x): _mamba_block_aligned_split", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD, NEW, 1)
ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print("Mamba align split uses the Mamba group block size, syntax OK")
