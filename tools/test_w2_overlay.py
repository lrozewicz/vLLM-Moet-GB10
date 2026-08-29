#!/usr/bin/env python3
"""Contract tests for tools/apply_w2_overlay.py.

Runs anywhere - no GPU, no vLLM install, no checkpoint. The point is to catch
the failure modes that would otherwise surface an hour into an image build, or
worse, at weight load:

  * the patch stopping to carry a W2 module as a new file (a regeneration that
    dropped work - AGENTS.md's recurring incident),
  * an anchor that no longer matches, or matches twice, after a vLLM bump,
  * a second run of the tool double-applying its hooks.

The vLLM tree is synthesised from the anchors the tool expects, so these tests
assert the CONTRACT, not a particular vLLM release. When a real bump breaks an
anchor, the Dockerfile's `--verify-only` and this file's `test_missing_anchor`
both say so loudly; what they cannot do is guess the new call site.

    python3 tools/test_w2_overlay.py
"""

import ast
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_w2_overlay as ov  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "patch" / "vllm-moet-v0.24.0.patch"

# Minimal stand-ins carrying exactly the anchor text the tool splices against.
FP8_STUB = '''class Fp8MoEMethod(FusedMoEMethodBase):
    def create_weights(self, layer):
        if something:
            pass
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def _setup_kernel(self):
        pass

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight

    def apply_monolithic(
        self,
        layer: RoutedExperts,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(x)

    @property
    def supports_eplb(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer,
    ) -> torch.Tensor:
        return None

    def apply(
        self,
        layer: RoutedExperts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
        )
'''

ENVS_STUB = '''import os

environment_variables = {}


def validate_environ(hard_fail: bool) -> None:
    for env in os.environ:
        if env.startswith("VLLM_") and env not in environment_variables:
            if hard_fail:
                raise ValueError("unknown")


def compute_hash() -> str:
    from vllm.config.utils import normalize_value

    factors = {}
    for factor, raw in os.environ.items():
        factors[factor] = normalize_value(raw)
    return str(factors)
'''

RUNNER_STUB = '''import functools

import vllm.envs as envs

logger = init_logger(__name__)


class GPUModelRunner:
    def execute_model(self):
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        model_output = self._model_forward()
        logits = self.model.compute_logits(sample_hidden_states)

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
        )

    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        kv_caches = {}
        bind_kv_cache(
            kv_caches,
        )
        return kv_caches

    def load_model(self, load_dummy_weights: bool = False) -> None:
        logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
        )
'''


def make_tree(root: Path) -> None:
    q = root / "vllm/model_executor/layers/quantization"
    (q / "utils").mkdir(parents=True, exist_ok=True)
    (q / "fp8.py").write_text(FP8_STUB)
    (root / "vllm/envs.py").write_text(ENVS_STUB)
    w = root / "vllm/v1/worker"
    w.mkdir(parents=True, exist_ok=True)
    (w / "gpu_model_runner.py").write_text(RUNNER_STUB)


class TestExtraction(unittest.TestCase):
    def test_patch_still_carries_every_module(self):
        """A module vanishing here means a regeneration dropped work."""
        if not PATCH.exists():
            self.skipTest("patch not present")
        new_files = ov.extract_new_files(PATCH)
        for rel in ov.W2_MODULES:
            self.assertIn(rel, new_files, f"{rel} is no longer a new file in the patch")
            self.assertGreater(len(new_files[rel]), 200, f"{rel} reconstructed empty")

    def test_extracted_modules_parse(self):
        if not PATCH.exists():
            self.skipTest("patch not present")
        new_files = ov.extract_new_files(PATCH)
        for rel in ov.W2_MODULES:
            ast.parse(new_files[rel])  # raises on a truncated reconstruction


class TestHooks(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        make_tree(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _apply_all(self):
        ov.patch_fp8(self.dir / "vllm/model_executor/layers/quantization/fp8.py")
        ov.patch_envs(self.dir / "vllm/envs.py")
        ov.patch_runner(self.dir / ov.RUNNER_FILE)

    def test_hooks_apply_and_parse(self):
        self._apply_all()
        for rel in (
            "vllm/model_executor/layers/quantization/fp8.py",
            "vllm/envs.py",
            ov.RUNNER_FILE,
        ):
            text = (self.dir / rel).read_text()
            self.assertIn(ov.MARK, text, f"{rel} not marked")
            ast.parse(text)  # a mis-indented splice fails here

    def test_idempotent(self):
        self._apply_all()
        before = {
            rel: (self.dir / rel).read_text()
            for rel in (
                "vllm/model_executor/layers/quantization/fp8.py",
                "vllm/envs.py",
                ov.RUNNER_FILE,
            )
        }
        self._apply_all()  # second run must be a no-op
        for rel, text in before.items():
            self.assertEqual(text, (self.dir / rel).read_text(), f"{rel} double-applied")

    def test_w2_forward_precedes_stock_asserts(self):
        """The hook must run BEFORE `assert not self.is_monolithic`.

        With W2 on, process_weights_after_loading returned early, so
        moe_kernel was never set up - asserting first fires on our own path.
        """
        self._apply_all()
        text = (self.dir / "vllm/model_executor/layers/quantization/fp8.py").read_text()
        hook = text.index("moe_w2_forward")
        assert_pos = text.index("assert not self.is_monolithic")
        self.assertLess(hook, assert_pos, "W2 forward hook landed after the stock asserts")

    def test_modular_is_forced(self):
        self._apply_all()
        text = (self.dir / "vllm/model_executor/layers/quantization/fp8.py").read_text()
        self.assertIn("_moe_w2_force_modular", text)
        self.assertIn("def is_monolithic", text)

    def test_gate_rewinds_recurrent_state(self):
        """Hybrid safety: the replay must restore state it already advanced."""
        self._apply_all()
        text = (self.dir / ov.RUNNER_FILE).read_text()
        self.assertIn("_moe_w2_recurrent_caches", text)
        snap = text.index("_w2_snap = [t.clone()")
        rewind = text.index("_t.copy_(_snap)")
        forward = text.index("regated_output = self._model_forward")
        self.assertLess(snap, rewind, "snapshot must precede the rewind")
        self.assertLess(rewind, forward, "rewind must precede the re-forward")

    def test_step_end_is_unconditional(self):
        """step_end must not sit inside the gate's fire branch."""
        self._apply_all()
        text = (self.dir / ov.RUNNER_FILE).read_text()
        for line in text.splitlines():
            if "_w2_tier.step_end()" in line:
                self.assertLessEqual(
                    len(line) - len(line.lstrip()),
                    24,
                    "step_end is nested too deep to run on every step",
                )
                break
        else:
            self.fail("step_end hook missing")

    def test_missing_anchor_is_loud(self):
        f = self.dir / "vllm/model_executor/layers/quantization/fp8.py"
        f.write_text(FP8_STUB.replace("layer.w2_input_scale = None", "pass"))
        with self.assertRaises(ov.ApplyError):
            ov.patch_fp8(f)

    def test_ambiguous_anchor_is_loud(self):
        f = self.dir / "vllm/envs.py"
        f.write_text(ENVS_STUB + "\n\ndef validate_environ(hard_fail: bool) -> None:\n    pass\n")
        with self.assertRaises(ov.ApplyError):
            ov.patch_envs(f)


class TestCubitFp8StreamBuild(unittest.TestCase):
    """The FP8 path must reach STREAM-BUILD, not full-checkpoint staging.

    This is the bug that stopped the first boot: upstream arms streaming for
    the NVFP4 and mxfp4 formats only, so FP8 block-quant staged the whole
    306 GiB of experts in host memory - fatal where host RAM is VRAM.
    """

    def setUp(self):
        if not PATCH.exists():
            self.skipTest("patch not present")
        self.dir = Path(tempfile.mkdtemp())
        mod = ov.extract_new_files(PATCH)[ov.CUBIT_FILE]
        dst = self.dir / ov.CUBIT_FILE
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(mod)
        self.f = dst

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_patch_applies_and_parses(self):
        self.assertTrue(ov.patch_moe_w2_cubit(self.f))
        text = self.f.read_text()
        ast.parse(text)
        self.assertIn("weight_scale_inv", text)
        self.assertIn("build_layer_planes_fp8", text)

    def test_idempotent(self):
        ov.patch_moe_w2_cubit(self.f)
        before = self.f.read_text()
        ov.patch_moe_w2_cubit(self.f)
        self.assertEqual(before, self.f.read_text())

    def test_fp8_branch_precedes_nvfp4(self):
        """scale_inv must be checked first: an FP8 layer has no scale_2, so
        the old code fell through to the mxfp4 branch and the WRONG builder."""
        ov.patch_moe_w2_cubit(self.f)
        text = self.f.read_text()
        fp8 = text.index('w13_weight_scale_inv", None) is not None')
        nvfp4 = text.index('"w13_weight_scale_2", "w2_weight_scale_2"')
        self.assertLess(fp8, nvfp4)

    def test_big_params_derived_from_expected(self):
        ov.patch_moe_w2_cubit(self.f)
        text = self.f.read_text()
        self.assertIn('big = tuple(n for n in expected', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
