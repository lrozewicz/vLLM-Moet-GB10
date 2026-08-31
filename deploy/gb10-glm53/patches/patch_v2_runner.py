#!/usr/bin/env python3
"""LATKA 5: cykl zycia warstwy FP4 w runnerze V2.

PO CO. Hooki silnika W2 sa zainstalowane w runnerze V1, a MTP na tym modelu
DZIALA WYLACZNIE na V2 (hybryda 34 KDA + 11 MLA rozjezdza drafter na dwie
grupy KV, czego proposer V1 nie dopuszcza - AssertionError). Zeby miec
JAKOSC (warstwa FP4) i SZYBKOSC (MTP) naraz, tier musi umiec zyc na V2.

ZAKRES. Tylko DWA hooki, bo bramka pewnosci jest na tym boksie nieosiagalna
(prog puli 12.8 GiB bez spekulacji, ~51 GiB z MTP k=3 - czlon `(1 + n_spec)`),
a to ona wymagala pozostalych trzech (logity w zasiegu, snapshot stanu KDA,
check_pool_floor). Sama warstwa FP4 potrzebuje wylacznie granicy kroku:

  1. wait_manager_idle() PRZED forwardem - commity rezydencji nie moga
     nachodzic na czytelnikow grafu CUDA;
  2. step_end() PO kroku - inaczej piny z force_promote kumuluja sie i tier
     po cichu przestaje promowac.

Roznica wobec V1: V1 mial kotwice "# Run the model. # Use persistent buffers
for CUDA graphs."; V2 przepisal ten fragment na jawny wybor PIECEWISE vs
eager wewnatrz `with set_forward_context(...)`. Wpinamy sie przed ten blok.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[PATCH-V2-TIER]" in src:
    print("juz zalatane"); sys.exit(0)

IMPORT_OLD = "import vllm.envs as envs\n"
IMPORT_NEW = """import vllm.envs as envs

# [PATCH-V2-TIER] cykl zycia warstwy FP4 (moe_w2_delta) na runnerze V2.
# Importowane defensywnie: bez silnika W2 runner ma dzialac bez zmian.
try:
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_delta as _w2_delta)
except Exception:  # noqa: BLE001
    _w2_delta = None
"""

DRAIN_OLD = """            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,"""
DRAIN_NEW = """            # [PATCH-V2-TIER] Commity rezydencji sa planowane po poprzednim
            # kroku. Drenujemy je PRZED tym forwardem, zeby mutacja tablicy
            # slotow / puli nie nachodzila na czytelnikow grafu CUDA.
            if _w2_delta is not None:
                for _t in (_w2_delta._BASE_TIER, _w2_delta._TIER):
                    if _t is not None:
                        _t.wait_manager_idle()

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,"""

STEP_END_OLD = """        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
            routed_experts=routed_experts,
        )
"""
STEP_END_NEW = STEP_END_OLD + """
        # [PATCH-V2-TIER] Zamknij okno SEEN/pin tiera na KAZDYM kroku, nie
        # tylko gdy cos promowano: piny z force_promote inaczej kumuluja sie
        # w nieskonczonosc i tier po cichu przestaje promowac (upstream
        # zmierzyl, ze marki z prefillu pinuja cala pule przy nastepnym
        # decode i psuja pierwsze tokeny).
        if _w2_delta is not None and not self.is_pooling_model:
            try:
                for _t in (_w2_delta._BASE_TIER, _w2_delta._TIER):
                    if _t is not None:
                        _t.step_end()
            except Exception as _e:  # noqa: BLE001 - nigdy nie wywracaj zadania
                logger.warning_once("moe_w2 step_end pominiety: %s", _e)
"""

for old, new, name in ((IMPORT_OLD, IMPORT_NEW, "import"),
                       (DRAIN_OLD, DRAIN_NEW, "drenaz przed forwardem"),
                       (STEP_END_OLD, STEP_END_NEW, "step_end po kroku")):
    if src.count(old) != 1:
        print(f"kotwica '{name}': {src.count(old)} wystapien (oczekiwano 1)",
              file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)

open(path, "w", encoding="utf-8").write(src)
import ast; ast.parse(src)
print("zalatane 3 miejsca (import + 2 hooki), skladnia OK")
