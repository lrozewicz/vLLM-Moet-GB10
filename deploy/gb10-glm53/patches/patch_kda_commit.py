#!/usr/bin/env python3
"""PATCH 16: KDA spec verify without per-draft-token state stores.

Hooks kda_commit.py (mounted as model_executor/layers/mamba/ops/kda_commit.py)
into two files; inert unless VLLM_KDA_COMMIT=1.

  python3 patch_kda_commit.py layer <models/glm5next/nvidia/kda.py>
  python3 patch_kda_commit.py state <v1/worker/gpu/model_states/mamba_hybrid.py>

layer: the spec-verify call of the GLM5-Next KDA layer goes through
       KdaCommit.verify (logs the rank-1 updates instead of storing 8 states).
state: prepare_attn tells KdaCommit which batch rows verify; postprocess_state
       commits the accepted state (KdaCommit.materialize) right after the
       accepted counts are scattered and BEFORE the align-mode copies, which
       read that slot.
"""
import ast
import sys

MARK = "[PATCH 16]"


def sub(src, old, new, what):
    if src.count(old) != 1:
        sys.exit(f"patch_kda_commit: anchor ({what}) moved: {src.count(old)} matches")
    return src.replace(old, new)


def layer(src):
    src = sub(
        src,
        "from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig\n",
        "from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig\n"
        f"from vllm.model_executor.layers.mamba.ops import kda_commit as _kda_commit  # {MARK}\n",
        "import",
    )
    src = sub(
        src,
        "        self._conv_state_dim_first = is_conv_state_dim_first()\n",
        "        self._conv_state_dim_first = is_conv_state_dim_first()\n"
        f"        # {MARK} spec verify without the per-token state stores\n"
        "        self._kda_commit_layer = -1\n"
        "        if _kda_commit.ENABLED:\n"
        "            assert self.kda_safe_gate, \"kda_commit implements the safe gate only\"\n"
        "            self._kda_commit_layer = _kda_commit.KdaCommit.register_layer()\n"
        "            _kda_commit.KdaCommit.set_max_rows(\n"
        "                vllm_config.scheduler_config.max_num_seqs\n"
        "            )\n",
        "__init__ tail",
    )
    old = """            core_attn_out_spec, _ = fused_recurrent_kda(
                q=_rearr(q_spec),
                k=_rearr(k_spec),
                v=_rearr(v_spec),
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )
"""
    new = f"""            if self._kda_commit_layer >= 0:  # {MARK}
                core_attn_out_spec = _kda_commit.KdaCommit.verify(
                    self._kda_commit_layer,
                    _rearr(q_spec),
                    _rearr(k_spec),
                    _rearr(v_spec),
                    g1_spec,
                    beta_spec,
                    recurrent_state,
                    spec_query_start_loc[: num_spec_decodes + 1],
                    spec_state_indices_tensor,
                    num_accepted_tokens,
                    spec_out,
                    self.A_log,
                    self.dt_bias,
                    lower_bound,
                )
            else:
""" + "\n".join(("    " + l) if l else l for l in old.split("\n"))
    return sub(src, old, new, "spec fused_recurrent_kda call")


def state(src):
    src = sub(
        src,
        "from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata\n",
        "from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata\n"
        f"from vllm.model_executor.layers.mamba.ops import kda_commit as _kda_commit  # {MARK}\n",
        "import",
    )
    src = sub(
        src,
        "            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)\n",
        "            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)\n"
        f"            if _kda_commit.ENABLED:  # {MARK} rows the KDA verify kernel runs\n"
        "                _kda_commit.KdaCommit.set_spec_rows(\n"
        "                    np.flatnonzero(num_decode_draft_tokens_np >= 0).tolist()\n"
        "                )\n",
        "prepare_attn draft counts",
    )
    src = sub(
        src,
        "        # Align: save the running state to the block-aligned position when\n",
        f"        # {MARK} commit the accepted KDA state before anything copies it\n"
        "        if _kda_commit.ENABLED:\n"
        "            _kda_commit.KdaCommit.materialize(\n"
        "                idx_mapping,\n"
        "                self.num_accepted_tokens_gpu,\n"
        "                self.num_accepted_tokens_gpu\n"
        "                if num_computed_tokens is None\n"
        "                else num_computed_tokens,\n"
        "                # the block size the align postprocess below uses\n"
        "                self._mamba_ctx.block_size\n"
        "                if self._align_mode\n"
        "                and num_computed_tokens is not None\n"
        "                and self._mamba_ctx is not None\n"
        "                else 0,\n"
        "            )\n"
        "\n"
        "        # Align: save the running state to the block-aligned position when\n",
        "postprocess align comment",
    )
    return src


def main():
    kind, path = sys.argv[1], sys.argv[2]
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print("already patched")
        return
    src = {"layer": layer, "state": state}[kind](src)
    ast.parse(src)
    open(path, "w", encoding="utf-8").write(src)
    print(f"kda_commit hooks ({kind}) added, syntax OK")


if __name__ == "__main__":
    main()
