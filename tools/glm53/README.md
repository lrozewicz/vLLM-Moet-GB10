# GLM-5.3-Flash measurement and build tools

Every number in [`docs/models/glm-5.3-flash.md`](../../docs/models/glm-5.3-flash.md)
was produced with these scripts. They talk to the OpenAI-compatible endpoint
(`VLLM_BASE`, default `http://localhost:8000`; `VLLM_API_KEY`) and need only
the Python standard library, except `kld_glm.py dump/compare` (numpy) and
`gptq_build.py` (runs inside the serving image). Without numpy on the host,
run them in the serving image:

```bash
docker run --rm --network host -e VLLM_API_KEY -v "$PWD":/w -w /w \
  --entrypoint python3 vllm-moet-glm53:dflash2 tools/glm53/kld_glm.py ...
```

| script | what it answers |
|---|---|
| `bench_mtp.py` | Decode tok/s and per-position draft acceptance (any speculative method). The "code" figure is the mean of three code prompts at temperature 0. |
| `humaneval/` | HumanEval pass@1: `generate.py` on the server, `execute.py` in a network-less container, `report.py`. The dataset is downloaded, not vendored (see `generate.py`). `plus_rewrite.py` re-scores the same answers with the much stricter HumanEval+ (EvalPlus) tests. `zai_generate.py` produces the same answers from the ORIGINAL model on the z.ai API (key read from a file), so the local server and the original are scored by one harness. |
| `kld_glm.py` | Deterministic quality: KL divergence, top-1 agreement and perplexity of one configuration against another, on a fixed corpus, by teacher forcing through prompt logprobs. Needs `MAX_LOGPROBS=64` on the server. |
| `quality_pl.py` | Generation probe for the two failure modes of bare 2-bit: loops and invented words (detected through the tokenizer), on Polish prose with English and code as controls. |
| `loop_probe.py` | Long-code loop probe (8 tasks x N samples, fixed seeds), for comparing plane variants. |
| `loop_inspect.py` | Tells a real loop (the same line/block repeated in a row, repeated 20-grams) from code structure, which fools the prose repetition metric. |
| `reasoning_probe.py` | Splits the token budget between `reasoning` and `content` for one task. |
| `pcache_replay.py` | Replays a captured agent request (messages + tools + sampling, as the client sent it) turn by turn, so every step restores state from the prefix cache the previous steps wrote; a step that looks corrupted is re-run without cache hits (`cache_salt`). This is how the prefix-cache bug fixed by patch 12 was reproduced. |
| `needle.py` | Long-context retrieval: 8 "access code" notes spread over real source code at even depths (`--hard`: 24 notes, 12 first names x 2 surnames, 8 asked), ~105K and ~190K tokens; reports hits per depth. |
| `conc_bench.py` | One vs two concurrent streams on a code-edit task (per-stream and aggregate tok/s, draft acceptance), and whether two long prompts (`--cap-sizes`) decode together without preemption. |
| `w8a16_bench.py` | Tunes the Triton W8A16 kernel of patch 13 (`deploy/gb10-glm53/patches/glm53_w8a16.py`) per weight shape against BF16 and FP8 Marlin; run in the serving image with the server stopped. |
| `loop_check.py` | Patch 14 kernels: the looped `moe_w2_mm`/`moe_w4_mm` (`kernels/cubins-sm120-loop`) against the shipped ones - bit-exact output for every shape, live pair count and grid, and the launch time with and without compaction. Server stopped, serving image, `NEW_DIR=<cubin dir>`. |
| `variant_time.py` | Same-work timing (one pair per CTA, scattered live pairs) and bit comparison of any number of cubin directories, e.g. a rebuild against the shipped set. Imports `loop_check.py`. |
| `desc_check.py` | Patch 14 desc build: the COMPACT tables hold exactly the live entries of the scattered ones, in order, with an empty tail. Takes the patched `moe_w2_cubit.py`. |
| `glue_bench.py` | Patch 14 glue fusion: the fused gather+quant+seen and SwiGLU+quant kernels against the production ops (bit-exact) and their time. Takes the patched `moe_w2_cubit.py`. |
| `step_sum.py` | Step time (decode seconds per draft), acceptance and tok/s per category from `bench_mtp.py` JSONs - the robust way to compare configurations, since acceptance moves +-3% between boots of the same one (the FP4 pool changes the greedy text). |
| `kda_state_bench.py` | The ceiling of patch 16: the KDA spec-verify kernel with 8 per-token state stores vs one (and vs 16 tokens with one, the replay shape). Server stopped, serving image. |
| `kda_commit_check.py` | Patch 16: `kda_commit` (rank-1 log + replay) against the stock kernel over chained steps with random acceptance - outputs, the committed slot and the block-boundary slot compared bit for bit, per replay fma form - then the time of both paths. Mount `kda_commit.py` into the image. |
| `build_code_calib.py` | Builds the code calibration corpus for GPTQ from local clones of open-source repositories. |
| `gptq_build.py` | Builds GPTQ planes (w13) from the MoE-input dumps of `patches/patch_xdump.py`. |

The FP4-section builder used during setup is one directory up:
[`tools/build_fp4_sections.py`](../build_fp4_sections.py).

## How to compare two configurations without fooling yourself

These are the rules the published numbers follow. Each one exists because
breaking it produced a wrong conclusion at least once.

1. **Measure the noise floor first.** Dump the same configuration twice
   (`kld_glm.py dump` twice in one boot) and compare the two. With the
   dynamic FP4 pool the served model is not a fixed function: two dumps of
   the same text differ by KLD 0.010-0.019 (0.006-0.010 without the pool).
   A difference below that is invisible.
2. **KLD says how far, perplexity says which way.** A change that moves the
   model by KLD 0.11 can be better or worse; only the perplexity of the real
   tokens on a corpus the change never saw tells you which.
3. **Keep evaluation and calibration disjoint.** Our evaluation corpus
   consists of repositories created after 2026-08-16, arXiv abstracts from
   late August/September and a Polish Wikipedia slice - none of it in the
   GPTQ calibration.
4. **Speed: same prompts, warm server, repeated.** Boot-to-boot variance is
   ~8%; acceptance depends on content far more than on configuration.
5. **HumanEval has +-2 tasks of noise**, and its answers are ~150 tokens, so
   it cannot see loops. Long-code generation needs `loop_probe.py`, and code
   repetition must be judged with `loop_inspect.py`, not the prose metric.
