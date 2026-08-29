#!/usr/bin/env python3
"""Install the moe_w2 engine onto an installed vLLM tree that is NOT v0.24.0.

Why this exists
---------------
`patch/vllm-moet-v0.24.0.patch` is a byte-exact `git diff` against the v0.24.0
tag and cannot be applied to any other tree.  Bringing a new model up on a
newer vLLM (GLM-5.3-Flash needs the `glm5next` implementation that first ships
in the `vllm/vllm-openai:glm53-flash*` images, vLLM 0.29-dev) therefore needs a
different delivery: the *self-contained* half of the patch is copied out
verbatim, and the handful of call sites it hooks are re-applied by anchor.

The split is what makes this cheap.  The patch has two halves:

  * ~7.7k lines of NEW files (``moe_w2_*.py``) that import only long-stable
    vLLM APIs.  Verified against 0.29-dev: every imported symbol still
    resolves.  These are copied byte-for-byte - no porting, no divergence.
  * a small number of hooks in existing files.  ``Fp8MoEMethod``'s signatures
    (``create_weights`` / ``process_weights_after_loading`` / ``apply`` /
    ``supports_eplb``) are unchanged between 0.24.0 and 0.29-dev, so the hooks
    transfer literally; only their surrounding whitespace moved.

**The patch stays the single source of truth for W2 code.**  This tool never
carries its own copy of a ``moe_w2_*`` module - it extracts them from the
patch at run time, so a patch regeneration (AGENTS.md) propagates here with no
second edit.  That is the same "iron rule" the rest of the repo follows.

What is deliberately NOT installed
----------------------------------
The DS4-specific half: DSpark/DFlash speculators, the v0.24.0 SM120 base
fixes (fixed upstream by 0.29), sparse-MLA backends and the nvfp4 KV cache.
GLM-5.3-Flash gets its attention, MTP and multimodal stack from the base image.
Runner-side hooks (base-cache replay, confidence gate) are staged separately -
see ``--stage``.

Usage (inside the image build; idempotent, safe to re-run):

    python3 tools/apply_w2_overlay.py --patch patch/vllm-moet-v0.24.0.patch
    python3 tools/apply_w2_overlay.py --verify-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Files copied verbatim out of the patch.  Each is a "new file" hunk there, so
# reconstruction is exact rather than a merge.
W2_MODULES = (
    "vllm/model_executor/layers/quantization/utils/moe_w2_bf12.py",
    "vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py",
    "vllm/model_executor/layers/quantization/utils/moe_w2_delta.py",
    "vllm/model_executor/layers/quantization/utils/moe_w2_gate.py",
    "vllm/model_executor/layers/quantization/utils/moe_w2_looka.py",
    "vllm/model_executor/layers/quantization/utils/moe_w2_planes.py",
    "vllm/model_executor/layers/quantization/utils/moe_w2_planes_cache.py",
    "vllm/model_executor/layers/quantization/utils/moe_w2_store.py",
    "vllm/model_executor/layers/quantization/utils/prefill_timers.py",
)

MARK = "VLLM_MOE_W2 overlay"


class ApplyError(RuntimeError):
    pass


# --------------------------------------------------------------- patch reader
def extract_new_files(patch: Path) -> dict[str, str]:
    """Reconstruct every 'new file' from a git patch: path -> contents.

    A new-file diff has no context lines, so the '+' side IS the file; this is
    reconstruction, not a merge, and cannot silently drift from the patch.
    """
    text = patch.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for chunk in re.split(r"^diff --git ", text, flags=re.M)[1:]:
        header, _, rest = chunk.partition("\n")
        name = header.split(" b/")[-1].strip()
        preamble = rest.split("@@", 1)[0]
        if "new file mode" not in preamble:
            continue
        body, started = [], False
        for line in rest.split("\n"):
            if line.startswith("@@"):
                started = True
                continue
            if started and line.startswith("+"):
                body.append(line[1:])
        out[name] = "\n".join(body) + "\n"
    return out


# ------------------------------------------------------------- anchored edits
def _splice(text: str, anchor: str, insert: str, *, where: str, after: bool = True) -> str:
    """Insert `insert` at the single occurrence of `anchor`.

    Hard-anchored on purpose: a missing or ambiguous anchor means the upstream
    call site moved and the hook must be re-read by a human, never guessed.
    """
    hits = text.count(anchor)
    if hits != 1:
        raise ApplyError(
            f"{where}: anchor found {hits} times, expected exactly 1.\n"
            f"  anchor: {anchor.strip()[:120]!r}\n"
            "  The upstream call site moved - re-read it before editing."
        )
    idx = text.index(anchor)
    at = idx + len(anchor) if after else idx
    return text[:at] + insert + text[at:]


FP8_STAGE_HOOK = '''
        # {mark}: the raw fp8 experts of ALL layers cannot be staged in
        # memory here (GLM-5.3-Flash: ~306 GiB against 121.6 GiB of unified
        # memory), so take the same three-step path the mxfp4/NVFP4 methods
        # take, in order of preference:
        #   1. LOADER-SKIP - a pack/planes cache already holds this layer,
        #      so the checkpoint tensors are never read at all;
        #   2. STREAM-BUILD - requantize each layer the moment its last
        #      expert tensor lands, so peak staging is one layer (~6 GiB),
        #      not the checkpoint;
        #   3. classic host staging - the upstream FP8 behaviour, correct
        #      only where host RAM is a separate, larger pool.
        from vllm.model_executor.layers.quantization.utils import moe_w2_cubit

        if self.block_quant and moe_w2_cubit.is_w2_layer(
            getattr(layer, "layer_name", "")
        ):
            # Remembered for `is_monolithic` below: quant methods are
            # per-layer, and this is the earliest point where the layer name
            # is known.
            self._moe_w2_force_modular = True
            if moe_w2_cubit.plan_pack_skip(layer):
                return
            if moe_w2_cubit.arm_stream_build(layer):
                return
            for pname in (
                "w13_weight",
                f"w13_{self.weight_scale_name}",
                "w2_weight",
                f"w2_{self.weight_scale_name}",
            ):
                p_ = getattr(layer, pname)
                attrs = dict(p_.__dict__)  # weight_loader, quant_method, ...
                newp = torch.nn.Parameter(p_.data.cpu(), requires_grad=False)
                layer.register_parameter(pname, newp)
                set_weight_attrs(newp, attrs)
'''

FP8_BUILD_HOOK = '''        # {mark}: adopt the 2-bit planes and skip the stock kernel setup.
        # The layer may already be built - LOADER-SKIP served it from a pack,
        # or STREAM-BUILD requantized it during load - so only build here
        # when neither happened. The key comes from create_weights, which is
        # what the stream builder used.
        from vllm.model_executor.layers.quantization.utils import moe_w2_cubit

        if self.block_quant and moe_w2_cubit.is_w2_layer(
            getattr(layer, "layer_name", "")
        ):
            key = getattr(layer, "_moe_w2_create_key", len(moe_w2_cubit._LAYERS))
            if not getattr(layer, "_moe_w2_stream_built", False):
                moe_w2_cubit.build_layer_planes_fp8(
                    layer, key, scale_suffix=self.weight_scale_name
                )
            layer._moe_w2_key = key
            return

'''

FP8_APPLY_HOOK = '''        # {mark} routed-expert path (cubit moe_w2_mm, 2-bit planes).
        # Shared experts stay with the MoE runner, matching the other
        # non-modular applies: this returns the routed output only.
        w2_key = getattr(layer, "_moe_w2_key", None)
        if w2_key is not None:
            from vllm.model_executor.layers.quantization.utils import moe_w2_cubit

            if layer.expert_map is not None or layer.apply_router_weight_on_input:
                raise RuntimeError(
                    "VLLM_MOE_W2 does not support expert maps/EPLB or "
                    "router-weight-on-input"
                )
            return moe_w2_cubit.moe_w2_forward(x, topk_weights, topk_ids, w2_key)

'''

FP8_MONOLITHIC_GUARD = '''        # {mark}: unreachable by construction (is_monolithic returns False
        # for W2 layers) - kept because the failure it guards is silent.
        if getattr(layer, "_moe_w2_key", None) is not None:
            raise RuntimeError(
                "VLLM_MOE_W2 layer reached the monolithic MoE kernel; its "
                "expert weights were replaced by 2-bit planes and the stock "
                "kernel would compute on freed tensors"
            )
'''


FP8_EPLB_OLD = """    def supports_eplb(self) -> bool:
        return True

    def apply_monolithic("""

FP8_EPLB_NEW = '''    def supports_eplb(self) -> bool:
        # {mark}: the 2-bit planes are built per physical layer and carry no
        # expert map, so EPLB rebalancing cannot be honoured.
        from vllm.model_executor.layers.quantization.utils import moe_w2_cubit

        return not moe_w2_cubit.enabled()

    @property
    def is_monolithic(self) -> bool:
        # {mark}: force the MODULAR path for W2 layers.
        #
        # This is load-bearing, not defensive.  `is_monolithic` falls back to
        # `experts_cls.is_monolithic()` whenever `moe_kernel is None` - which
        # is exactly our state, because process_weights_after_loading returns
        # before `_setup_kernel`.  Several FP8 backends advertise a monolithic
        # class first (FLASHINFER_TRTLLM returns
        # [TrtLlmFp8ExpertsMonolithic, TrtLlmFp8ExpertsModular]), and a
        # monolithic dispatch calls `apply_monolithic`, which routes
        # internally and reads `w13_weight`/`w2_weight` - tensors the plane
        # builder has already replaced.  The W2 GEMM needs the modular form
        # anyway: it consumes topk_weights/topk_ids computed by the router.
        if getattr(self, "_moe_w2_force_modular", False):
            return False
        return super().is_monolithic

    def apply_monolithic('''


def patch_fp8(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        return False

    # 1. host-stage the experts at create_weights time
    text = _splice(
        text,
        "            layer.w13_input_scale = None\n            layer.w2_input_scale = None\n",
        FP8_STAGE_HOOK.replace("{mark}", MARK),
        where="fp8.Fp8MoEMethod.create_weights",
    )

    # 2. build the planes instead of the stock kernel setup
    text = _splice(
        text,
        "    def process_weights_after_loading(self, layer: RoutedExperts) -> None:\n",
        FP8_BUILD_HOOK.replace("{mark}", MARK),
        where="fp8.Fp8MoEMethod.process_weights_after_loading",
    )

    # 3. route the forward through the 2-bit GEMM.  It must land BEFORE the
    #    stock asserts: when W2 is on, process_weights_after_loading returned
    #    early, so `self.moe_kernel` was never set up and `is_monolithic` is
    #    not guaranteed false - asserting first would fire on our own path.
    text = _splice(
        text,
        "        assert not self.is_monolithic\n"
        "        assert self.moe_kernel is not None\n"
        "        return self.moe_kernel.apply(\n",
        FP8_APPLY_HOOK.replace("{mark}", MARK),
        where="fp8.Fp8MoEMethod.apply",
        after=False,
    )

    # 3b. a W2 layer must never reach the monolithic kernel; if the override
    #     above is ever bypassed, fail loudly instead of silently computing on
    #     tensors the plane builder replaced.
    text = _splice(
        text,
        "        assert self.is_monolithic\n        assert self.moe_kernel is not None\n",
        FP8_MONOLITHIC_GUARD.replace("{mark}", MARK),
        where="fp8.Fp8MoEMethod.apply_monolithic",
        after=False,
    )

    # 4. EPLB is incompatible with the plane layout
    if FP8_EPLB_OLD not in text:
        raise ApplyError("fp8.Fp8MoEMethod.supports_eplb: anchor not found")
    text = text.replace(FP8_EPLB_OLD, FP8_EPLB_NEW.replace("{mark}", MARK), 1)

    path.write_text(text, encoding="utf-8")
    return True


ENVS_HELPERS = '''

# {mark}: the W2 engine reads its knobs straight from the environment rather
# than the typed registry, so upstream's unknown-variable validator must not
# reject them, and compile-cache identity must still include them - they change
# graph shapes, residency and numerics.
_MOET_EXTENSION_ENV_PREFIXES = ("VLLM_MOE_W2",)


def _is_moet_extension_env(name: str) -> bool:
    return name.startswith(_MOET_EXTENSION_ENV_PREFIXES)

'''


def patch_envs(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        return False

    text = _splice(
        text,
        "def validate_environ(hard_fail: bool) -> None:\n",
        "",
        where="envs.validate_environ",
    )
    text = text.replace(
        "def validate_environ(hard_fail: bool) -> None:\n",
        ENVS_HELPERS.replace("{mark}", MARK).lstrip("\n")
        + "\ndef validate_environ(hard_fail: bool) -> None:\n",
        1,
    )
    old = '        if env.startswith("VLLM_") and env not in environment_variables:'
    if old not in text:
        raise ApplyError("envs.validate_environ: unknown-variable anchor not found")
    text = text.replace(
        old,
        '        if (\n'
        '            env.startswith("VLLM_")\n'
        "            and env not in environment_variables\n"
        "            and not _is_moet_extension_env(env)\n"
        "        ):",
        1,
    )

    # compile-cache identity: fold every explicitly set extension knob in
    anchor = "        factors[factor] = normalize_value(raw)\n"
    if anchor not in text:
        raise ApplyError("envs.compute_hash: factors anchor not found")
    text = text.replace(
        anchor,
        anchor
        + "    for factor, raw in os.environ.items():\n"
        "        if _is_moet_extension_env(factor):\n"
        "            factors[factor] = normalize_value(raw)\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    return True


RUNNER_IMPORT_HOOK = '''
# {mark}: the FP4 delta tier and the confidence gate are driven from the
# runner (residency commits must not overlap graph readers, and the gate's
# signal - confidence - exists only once logits are computed).  Imported
# defensively so a partial deployment still boots; None => no gate.
from vllm.model_executor.layers.quantization.utils import moe_w2_delta

try:
    from vllm.model_executor.layers.quantization.utils import moe_w2_gate
except Exception:  # noqa: BLE001
    import os as _os  # this module does not import os at top level

    if _os.getenv("VLLM_MOE_W2_GATE", "0") == "1":
        raise
    moe_w2_gate = None
'''

RUNNER_DRAIN_HOOK = '''        # {mark}: residency commits are scheduled after the previous step.
        # Drain them before this forward so slot-table/pool mutation cannot
        # overlap graph readers.
        for _w2_tier in (moe_w2_delta._BASE_TIER, moe_w2_delta._TIER):
            if _w2_tier is not None:
                _w2_tier.wait_manager_idle()

'''

# Single-GB10 profile: TP=1 and PP=1, so every all-reduce in the upstream
# gate collapses to the local decision and the PP re-forward path is dead
# code.  Keeping them out is not a shortcut - a silently mis-ported
# collective desyncs or hangs, and there is no second rank here to justify
# carrying one.  Multi-rank deployments are refused rather than approximated.
RUNNER_GATE_HOOK = '''        # {mark}: confidence-gated FP4 re-forward (VLLM_MOE_W2_GATE=1;
        # default OFF, so the serving path is unchanged when unset).  When
        # this step's 2-bit top-1 is low-confidence, pull its routed COLD
        # experts up to FP4 and replay the forward so the step is re-decided
        # at FP4.  Fires on real DECODE only: a spec-decode verify step, or a
        # pure single-token decode; prefill is excluded.
        #
        # FIXED-POINT fire: one replay is only first-order - experts upgraded
        # early re-route later layers onto still-cold experts INSIDE the
        # replay, which then decide the token at 2 bits.  Iterate
        # promote -> replay until nothing new is promoted or the bound trips.
        #
        # This is the single-GPU form of the upstream hook.  GB10 is one
        # device: TP=PP=1, every rank-consensus reduce collapses to the local
        # decision.  Multi-rank is refused above rather than approximated.
        # Fail-safe: any error keeps the 2-bit logits.
        if (
            moe_w2_gate is not None
            and moe_w2_gate.enabled()
            and logits is not None
            and not self.is_pooling_model
            and (spec_decode_metadata is not None or max_num_scheduled_tokens <= 1)
        ):
            if get_pp_group().world_size > 1 or get_tp_group().world_size > 1:
                raise RuntimeError(
                    "VLLM_MOE_W2_GATE on this build supports single-rank "
                    "serving only (the re-forward is a collective); use the "
                    "v0.24.0 patch for TP/PP deployments"
                )
            try:
                if moe_w2_gate.should_reforward(logits, spec=spec_decode_metadata):
                    # Recurrent layers advance their state in place, so a
                    # replay needs the step's ENTRY state restored first.
                    # Refuse to fire rather than corrupt the sequence if the
                    # handles were never discovered.
                    _w2_rec = getattr(self, "_moe_w2_recurrent_caches", None)
                    if _w2_rec is None and self.kv_cache_config.has_mamba_layers:
                        raise RuntimeError(
                            "recurrent-state tensors unknown; the gate cannot "
                            "safely replay a step on a hybrid model"
                        )
                    _w2_snap = [t.clone() for t in (_w2_rec or [])]
                    _gate_iters = 0
                    _step_budget = moe_w2_gate.step_promote_budget()
                    while _gate_iters < moe_w2_gate.fire_fp_max():
                        # GATE_MAX_PROMOTE is a per-STEP budget shared across
                        # the fixed-point passes; reading it per call would
                        # multiply the cap by the loop count and cycle the
                        # whole pool on every fire.
                        n_promoted = moe_w2_gate.force_promote_step(
                            max_promote=_step_budget
                        )
                        if _step_budget is not None:
                            _step_budget = max(_step_budget - n_promoted, 0)
                        # Replay only if a cold expert was actually upgraded:
                        # otherwise the result cannot change and the replay is
                        # wasted HBM bandwidth.
                        if not (n_promoted > 0 and moe_w2_gate.reforward_enabled()):
                            break
                        _gate_iters += 1
                        # Rewind recurrent state to this step's entry value:
                        # the forward we are about to repeat already advanced
                        # it once.
                        for _t, _snap in zip(_w2_rec or [], _w2_snap):
                            _t.copy_(_snap)
                        with set_forward_context(
                            attn_metadata,
                            self.vllm_config,
                            num_tokens=num_tokens_padded,
                            num_tokens_across_dp=num_tokens_across_dp,
                            cudagraph_runtime_mode=cudagraph_mode,
                            batch_descriptor=batch_desc,
                            ubatch_slices=ubatch_slices_padded,
                            slot_mapping=slot_mappings,
                            skip_compiled=has_encoder_input,
                        ):
                            regated_output = self._model_forward(
                                input_ids=input_ids,
                                positions=positions,
                                intermediate_tensors=intermediate_tensors,
                                inputs_embeds=inputs_embeds,
                                **model_kwargs,
                            )
                        if self.use_aux_hidden_state_outputs:
                            hidden_states, aux_hidden_states = regated_output
                        else:
                            hidden_states = regated_output
                        sample_hidden_states = hidden_states[logits_indices]
                        logits = self.model.compute_logits(sample_hidden_states)
            except Exception as e:  # noqa: BLE001 - never fail a request
                logger.warning_once("moe_w2 gate re-forward skipped: %s", e)

        # {mark}: close each tier's per-step SEEN/pin window on EVERY step,
        # not just when the gate fired.  force_promote's hit-pins otherwise
        # accumulate forever and the gate silently stops promoting; upstream
        # measured a prefill's accumulated marks pinning the whole pool at the
        # next decode step, corrupting the first decode tokens.
        if (
            moe_w2_delta._BASE_TIER is not None or moe_w2_delta._TIER is not None
        ) and not self.is_pooling_model:
            try:
                for _w2_tier in (moe_w2_delta._BASE_TIER, moe_w2_delta._TIER):
                    if _w2_tier is not None:
                        _w2_tier.step_end()
            except Exception as e:  # noqa: BLE001 - never fail a request
                logger.warning_once("moe_w2 step_end skipped: %s", e)

'''

RUNNER_RECURRENT_HOOK = '''        # {mark}: remember the recurrent-state tensors (KDA / Mamba-class
        # layers).  The confidence gate re-runs a step's forward, and unlike an
        # attention KV write - same slots, same values, idempotent - a
        # recurrent layer READS its state and writes the advanced state back
        # in place.  Replaying a step would therefore advance it TWICE and
        # corrupt every later token of that sequence.  GLM-5.3-Flash has 34
        # such layers out of 45, so this is not an edge case here.
        #
        # Keeping the handles lets the gate snapshot and restore them around a
        # replay.  Only Mamba-spec groups are collected: the attention KV
        # cache is tens of GiB and must never be cloned.
        self._moe_w2_recurrent_caches = []
        try:
            from vllm.v1.kv_cache_interface import MambaSpec

            for _g in kv_cache_config.kv_cache_groups:
                if isinstance(_g.kv_cache_spec, MambaSpec):
                    for _ln in _g.layer_names:
                        _t = kv_caches.get(_ln)
                        if _t is not None:
                            self._moe_w2_recurrent_caches.append(_t)
        except Exception as e:  # noqa: BLE001 - never break cache init
            logger.warning("moe_w2: recurrent-state discovery failed: %s", e)
            self._moe_w2_recurrent_caches = []
        if self._moe_w2_recurrent_caches:
            logger.info(
                "moe_w2: tracking %d recurrent-state tensors for gate replay",
                len(self._moe_w2_recurrent_caches),
            )

'''


RUNNER_FLOOR_HOOK = '''        # {mark}: refuse to serve a delta/base pool sized below its per-step
        # working set - routed experts would keep zeroed contributions, which
        # is silent quality corruption rather than a visible failure.  Runs
        # here because the plane builder has registered every MoE layer by
        # end-of-load and the spec/scheduler configs are final.
        # VLLM_MOE_W2_FORCE_POOL=1 downgrades the failure to a warning.
        if (
            moe_w2_delta._BASE_TIER is not None or moe_w2_delta._TIER is not None
        ) and not load_dummy_weights:
            try:
                _n_spec = 0
                if self.speculative_config is not None:
                    _n_spec = int(
                        getattr(self.speculative_config, "num_speculative_tokens", 0)
                        or 0
                    )
                _topk = int(
                    getattr(self.model_config.hf_text_config, "num_experts_per_tok", 0)
                    or 0
                )
                if _topk > 0:
                    moe_w2_delta.check_pool_floor(
                        _topk, _n_spec, int(self.scheduler_config.max_num_seqs)
                    )
            except ValueError:
                raise
            except Exception as e:  # noqa: BLE001 - guard must not break boot
                logger.warning("moe_w2 pool-floor check skipped: %s", e)

'''

RUNNER_FILE = "vllm/v1/worker/gpu_model_runner.py"


def patch_runner(path: Path) -> bool:
    """Install the tier-lifecycle and confidence-gate hooks in the V1 runner.

    Only the hooks the single-GB10 profile actually needs.  Deliberately NOT
    ported: the base-cache miss replay and its PP segment variant (the 2-bit
    planes are GPU-resident here, so `_BASE_TIER` stays None), the PP
    full-pipeline `gate_reforward`, and the LOOKA router-lookahead arming.
    """
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        return False

    text = _splice(
        text,
        "import vllm.envs as envs\n",
        RUNNER_IMPORT_HOOK.replace("{mark}", MARK),
        where="gpu_model_runner imports",
    )
    text = _splice(
        text,
        "        # Run the model.\n        # Use persistent buffers for CUDA graphs.\n",
        "",
        where="gpu_model_runner forward site",
    )
    text = text.replace(
        "        # Run the model.\n        # Use persistent buffers for CUDA graphs.\n",
        RUNNER_DRAIN_HOOK.replace("{mark}", MARK)
        + "        # Run the model.\n        # Use persistent buffers for CUDA graphs.\n",
        1,
    )
    text = _splice(
        text,
        "        self.execute_model_state = ExecuteModelState(\n",
        RUNNER_GATE_HOOK.replace("{mark}", MARK),
        where="gpu_model_runner gate site",
        after=False,
    )
    text = _splice(
        text,
        "        bind_kv_cache(\n",
        RUNNER_RECURRENT_HOOK.replace("{mark}", MARK),
        where="gpu_model_runner kv-cache init",
        after=False,
    )
    text = _splice(
        text,
        "            format_gib(self.model_memory_usage),\n"
        "            time_after_load - time_before_load,\n        )\n",
        RUNNER_FLOOR_HOOK.replace("{mark}", MARK),
        where="gpu_model_runner pool-floor site",
    )
    path.write_text(text, encoding="utf-8")
    return True



# --------------------------------------------------- moe_w2_cubit extensions
# The one place this tool edits a W2 module rather than copying it verbatim.
#
# Upstream's streaming first-boot exists for the modelopt-NVFP4 and mxfp4
# checkpoint paths, but NOT for FP8 block-quant: Fp8MoEMethod stages the whole
# expert checkpoint in host RAM and requantizes afterwards. On a discrete GPU
# that is merely wasteful (host RAM is a separate pool); on GB10, where host
# RAM IS VRAM, it is fatal - GLM-5.3-Flash's experts are ~306 GiB against
# 121.6 GiB of memory. Measured on the first boot attempt: memory climbed
# ~12 GiB/min through 86 GiB with no plane ever built, and the run had to be
# stopped before the box wedged.
#
# DS4-Flash never exposed this: its checkpoint is FP4 and takes the mxfp4
# path, which IS armed. So the two functions that dispatch on checkpoint
# format learn the FP8 variant here.

CUBIT_FILE = "vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py"

CUBIT_SKIP_OLD = '''    for pname in ("w13_weight", "w13_weight_scale",
                  "w2_weight", "w2_weight_scale"):
'''

CUBIT_SKIP_NEW = '''    # {mark}: FP8 block-quant names its scales *_weight_scale_inv, so the
    # fixed tuple below raised AttributeError on that path instead of
    # skipping the layer.
    _sfx = (
        "weight_scale_inv"
        if getattr(layer, "w13_weight_scale_inv", None) is not None
        else "weight_scale"
    )
    for pname in ("w13_weight", f"w13_{_sfx}",
                  "w2_weight", f"w2_{_sfx}"):
'''

CUBIT_ARM_OLD = '''    if all(getattr(layer, _n, None) is not None
           for _n in ("w13_weight_scale_2", "w2_weight_scale_2")):
'''

CUBIT_ARM_NEW = '''    # {mark}: FP8 block-quant variant. Its scales are *_weight_scale_inv and
    # its builder takes the suffix, so neither existing branch fits; without
    # this the FP8 path falls back to staging the whole checkpoint, which
    # cannot fit in unified memory.
    if getattr(layer, "w13_weight_scale_inv", None) is not None:
        import functools as _ft

        expected = {"w13_weight": 2 * E, "w13_weight_scale_inv": 2 * E,
                    "w2_weight": E, "w2_weight_scale_inv": E}
        layer._moe_w2_stream_builder = _ft.partial(
            build_layer_planes_fp8, scale_suffix="weight_scale_inv")
    elif all(getattr(layer, _n, None) is not None
             for _n in ("w13_weight_scale_2", "w2_weight_scale_2")):
'''

CUBIT_BIG_OLD = '''    big = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
'''

CUBIT_BIG_NEW = '''    # {mark}: derive the big params from `expected` instead of naming them,
    # so the FP8 (*_scale_inv) branch above stubs the right four.
    big = tuple(n for n in expected if not n.endswith("_scale_2"))
'''


CUBIT_EMPTYCACHE_OLD = '''    # fp8 experts are 4x the bytes of the mxfp4 path and the requant makes f32
    # temporaries -> smaller H2D chunks, per-expert quantize.
    chunk = 8
    for e0 in range(0, E, chunk):
'''

CUBIT_EMPTYCACHE_NEW = '''    # fp8 experts are 4x the bytes of the mxfp4 path and the requant makes f32
    # temporaries -> smaller H2D chunks, per-expert quantize.
    chunk = 8
    for e0 in range(0, E, chunk):
        # {mark}: return the allocator's cached blocks each chunk.
        #
        # MEASURED on GB10, and it is the difference between booting and not:
        # without this the requant loop consumed ~1.33 GiB per expert that
        # never came back - 85 GiB gone in 64 experts, with the box wedging
        # before a single layer finished. It is not a leak. The FP8 dequant
        # path is f64-heavy (w.double(), the block scales repeat_interleave'd
        # to a full [N, K] f64, then w * s, then u and u.abs() inside
        # _f64_to_codes_scales), so each expert churns several ~134 MiB f64
        # temporaries; the caching allocator kept growing instead of reusing
        # them, and on unified memory those bytes are the machine's only RAM.
        # With this call: allocated tracks the planes alone (+~1.8 GiB per
        # layer, which is what they weigh), free memory stays flat at ~75-85
        # GiB, and layers complete in ~14 s each.
        #
        # The other quantization paths never hit this because their dequant
        # is a lookup, not f64 arithmetic over the whole expert.
        torch.cuda.empty_cache()
'''

def patch_moe_w2_cubit(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        return False
    for old, new, where in (
        (CUBIT_SKIP_OLD, CUBIT_SKIP_NEW, "moe_w2_cubit.plan_pack_skip"),
        (CUBIT_ARM_OLD, CUBIT_ARM_NEW, "moe_w2_cubit.arm_stream_build format branch"),
        (CUBIT_BIG_OLD, CUBIT_BIG_NEW, "moe_w2_cubit.arm_stream_build big tuple"),
        (CUBIT_EMPTYCACHE_OLD, CUBIT_EMPTYCACHE_NEW,
         "moe_w2_cubit.build_layer_planes_fp8 requant loop"),
    ):
        if text.count(old) != 1:
            raise ApplyError(
                f"{where}: anchor found {text.count(old)} times, expected 1"
            )
        text = text.replace(old, new.replace("{mark}", MARK), 1)
    path.write_text(text, encoding="utf-8")
    return True


# ------------------------------------------------------------------ front end
def vllm_root() -> Path:
    import vllm  # noqa: PLC0415

    return Path(vllm.__file__).resolve().parent.parent


def verify(root: Path, with_runner: bool = True) -> list[str]:
    problems = []
    for rel in W2_MODULES:
        if not (root / rel).exists():
            problems.append(f"missing module: {rel}")
    checked = [
        CUBIT_FILE,
        "vllm/model_executor/layers/quantization/fp8.py",
        "vllm/envs.py",
    ]
    if with_runner:
        checked.append(RUNNER_FILE)
    for rel in checked:
        if MARK not in (root / rel).read_text(encoding="utf-8"):
            problems.append(f"hook not applied: {rel}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch", type=Path, default=Path("patch/vllm-moet-v0.24.0.patch"))
    ap.add_argument("--root", type=Path, default=None, help="site-packages root")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument(
        "--no-runner",
        action="store_true",
        help="skip the runner hooks: installs the 2-bit base path only "
        "(GPU-resident experts, no FP4 delta tier and no confidence gate)",
    )
    args = ap.parse_args()

    root = args.root or vllm_root()
    print(f"vLLM tree: {root}")

    if args.verify_only:
        problems = verify(root, not args.no_runner)
        for p in problems:
            print(f"  FAIL {p}")
        print("overlay verified" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    new_files = extract_new_files(args.patch)
    missing = [m for m in W2_MODULES if m not in new_files]
    if missing:
        raise ApplyError(
            "the patch no longer carries these modules as new files: "
            + ", ".join(missing)
            + "\nRegenerate it from the fork branch (AGENTS.md) before building."
        )

    copied = 0
    for rel in W2_MODULES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        body = new_files[rel]
        if dst.exists() and dst.read_text(encoding="utf-8") == body:
            continue
        dst.write_text(body, encoding="utf-8")
        copied += 1
    print(f"modules: {copied} written, {len(W2_MODULES) - copied} already current")

    edits = [
        (CUBIT_FILE, patch_moe_w2_cubit),
        ("vllm/model_executor/layers/quantization/fp8.py", patch_fp8),
        ("vllm/envs.py", patch_envs),
    ]
    if not args.no_runner:
        edits.append((RUNNER_FILE, patch_runner))
    for name, fn in edits:
        changed = fn(root / name)
        print(f"hooks: {name} {'patched' if changed else 'already patched'}")

    problems = verify(root, not args.no_runner)
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        return 1
    print("overlay installed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ApplyError as e:
        print(f"apply_w2_overlay: {e}", file=sys.stderr)
        sys.exit(2)
