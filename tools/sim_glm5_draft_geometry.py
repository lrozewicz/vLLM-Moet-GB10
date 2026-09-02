#!/usr/bin/env python3
"""Symulacja geometrii KV dla GLM-5.3-Flash + drafter DFlash2 na JEDNYM GB10.

Uruchamiac WEWNATRZ obrazu z nalozonym overlayem DFlash2 (bez GPU):

    docker run --rm --entrypoint python3 -v $PWD/tools:/t:ro \\
        vllm-moet-glm53:dflash2 /t/sim_glm5_draft_geometry.py

Buduje specyfikacje KV o naszej geometrii (34 KDA + 11 MLA po 656 B/token
z dopchanym latentem + 11 indexerow + 11 ogonow kpool + 5 warstw SWA
draftera), przepuszcza je przez zalatane `kv_cache_utils` i drukuje: tryb
grupy draftera, blok, bajty na blok puli, liczbe blokow dla zadanego budzetu
i wymagane minimum. Jeden przebieg zamiast 15-minutowego bootu na probe.
"""
import argparse
from types import SimpleNamespace

import torch

from vllm.v1.core import kv_cache_utils as K
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

ap = argparse.ArgumentParser()
ap.add_argument("--mla-block", type=int, default=7168)
ap.add_argument("--mla-bytes", type=int, default=656, help="B/token warstwy MLA")
ap.add_argument("--draft-kv-heads", type=int, default=8)
ap.add_argument("--draft-dtype", default="uint8", choices=["uint8", "bfloat16"])
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--avail-gib", type=float, default=6.0)
ap.add_argument("--max-len", type=int, default=204800)
ap.add_argument("--mnbt", type=int, default=2048)
args = ap.parse_args()

vllm_config = SimpleNamespace(
    parallel_config=SimpleNamespace(pipeline_parallel_size=1, decode_context_parallel_size=1),
    model_config=SimpleNamespace(max_model_len=args.max_len),
    cache_config=SimpleNamespace(num_gpu_blocks_override=None, mamba_cache_mode="align"),
    max_in_flight_tokens=args.mnbt,
    speculative_config=None,
)

spec: dict = {}
for i in range(34):
    spec[f"model.layers.{i}.kda"] = MambaSpec(
        block_size=args.mla_block, shapes=((64, 128, 128),), dtypes=(torch.float32,)
    )
for i in range(11):
    spec[f"model.layers.{i}.mla"] = MLAAttentionSpec(
        block_size=args.mla_block, num_kv_heads=1, head_size=args.mla_bytes, dtype=torch.uint8
    )
    spec[f"model.layers.{i}.indexer"] = MLAAttentionSpec(
        block_size=args.mla_block, num_kv_heads=1, head_size=132, dtype=torch.uint8, compress_ratio=4
    )
    spec[f"model.layers.{i}.kpool_tail"] = KpoolTailSpec(
        block_size=4, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, sliding_window=4
    )
draft_dtype = getattr(torch, args.draft_dtype)
for i in range(5):
    spec[f"drafter.layers.{i}.attn"] = SlidingWindowSpec(
        block_size=args.mla_block, num_kv_heads=args.draft_kv_heads, head_size=128,
        dtype=draft_dtype, sliding_window=args.window,
    )

groups = K._get_kv_cache_groups_glm5_next(vllm_config, spec)
assert groups is not None, "sciezka glm5_next odrzucila model"
layout = K._glm5_next_tensor_layout(groups)
assert layout is not None, "wykrywanie layoutu padlo"
draft_g = layout[8]
mla_page = layout[4]
print(f"grupy: {len(groups)}  mla_page={mla_page} B ({mla_page/2**20:.2f} MiB)")
for g in groups:
    s = g.kv_cache_spec
    inner = s.kv_cache_specs if isinstance(s, UniformTypeKVCacheSpecs) else {"": s}
    any_s = next(iter(inner.values()))
    print(f"  grupa {len(g.layer_names):2d} warstw  blok={s.block_size:5d}  page={any_s.page_size_bytes:>9d}  "
          f"padded={getattr(any_s, 'page_size_padded', None)}  typ={type(any_s).__name__}")
assert draft_g is not None, "brak grupy draftera"
dpage = next(iter(draft_g.kv_cache_spec.kv_cache_specs.values())).page_size_bytes
mode = "EXACT-FIT" if dpage == mla_page else "STANDALONE"
print(f"drafter: tryb {mode}, blok {draft_g.kv_cache_spec.block_size}, strona {dpage/2**20:.2f} MiB/warstwe")

avail = int(args.avail_gib * 2**30)
pb = K._pool_bytes_per_block(vllm_config, groups)
cfg = K.get_kv_cache_config_from_groups(vllm_config, groups, avail)
need = K._max_memory_usage_bytes_from_groups(vllm_config, groups)
sched_bs, hash_bs = K.resolve_kv_cache_block_sizes(
    SimpleNamespace(kv_cache_groups=groups),
    SimpleNamespace(cache_config=SimpleNamespace(block_size=args.mla_block, enable_prefix_caching=True, prefix_match_unit=None),
                    parallel_config=vllm_config.parallel_config, kv_transfer_config=None),
)
print(f"bajty/blok puli: {pb/2**20:.2f} MiB  (bez draftera: {(11*mla_page + 11*layout[5])/2**20:.2f} MiB)")
print(f"budzet {args.avail_gib} GiB -> {cfg.num_blocks} blokow = {cfg.num_blocks*args.mla_block} tokenow MLA")
print(f"minimum na 1 request: {need/2**30:.2f} GiB   tensorow: {len(cfg.kv_cache_tensors)}")
print(f"scheduler_block={sched_bs} hash_block={hash_bs}")
per_req_draft_blocks = draft_g.kv_cache_spec.max_memory_usage_pages(vllm_config)
print(f"drafter: {per_req_draft_blocks} blokow puli na request = {per_req_draft_blocks*args.mla_block} tokenow MLA 'zuzytych' id")
covered = set()
for t in cfg.kv_cache_tensors:
    covered.update(t.shared_by)
assert covered == set(spec), set(spec) - covered
print("OK: kazda warstwa ma tensor")
