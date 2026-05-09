# PR #38325: swapAB SM120 CUTLASS blockwise FP8 GEMM

## What this is

A vendored copy of vLLM upstream PR [#38325](https://github.com/vllm-project/vllm/pull/38325)
("Add swapAB support for SM120 CUTLASS blockwise FP8 GEMM"), rewritten so it
applies cleanly to the v0.19.0 source tree we build from. Without this rewrite
the upstream `.diff` would fail because the PR was authored against a tree
where `csrc/libtorch_stable/...` and `torch::stable::Tensor` exist; v0.19.0
still has `csrc/quantization/...` and `torch::Tensor`.

## What it does

Adds an alternative CUTLASS GEMM dispatch (`Sm120BlockwiseScaleConfig` with
B-major weight layout — "swapAB") for the FP8 blockwise path on SM120 family
hardware (which includes SM121 / DGX Spark via `enable_sm120_family<...>`).
The runtime auto-selects swapAB whenever `M ≤ 64 || M % 4 != 0`, i.e. exactly
the decode shapes our `shared_expert` FP8 layers hit at batch size 1-4.

Single .cuh file changed (`scaled_mm_blockwise_sm120_fp8_dispatch.cuh`),
~150 lines net. No CUTLASS submodule bump, no other csrc, no Python changes.

## Default-on since 2026-05-09

Measured on Qwen3.5-122B/Spark with hybrid INT4+FP8 + INT8 LM Head v2 + MTP-2:

| Test          | Without PR (autotune only) | With PR | Δ      |
|---------------|----------------------------|---------|--------|
| Q&A 256       | 50.5                       | 50.75   | +0.5%  |
| Code 512      | 52.6                       | 53.4    | +1.5%  |
| JSON 1024     | 51.3                       | 51.6    | +0.6%  |
| Math 64       | 47.7                       | 48.1    | +0.8%  |
| LongCode 2048 | 54.85                      | 55.05   | +0.4%  |

Average +0.76% marginal contribution. Real but small — most of the
shared_expert FP8 GEMM time is bandwidth-bound on the 273 GB/s UMA, not
scheduling-bound. swapAB helps register pressure and access patterns, not
the bandwidth ceiling we're already pushing against.

Originally we shipped this as opt-in (`--with-pr38325` flag). Re-evaluated
on 2026-05-09: on a fresh install the user pays 30-60 min for `vllm-sm121`
build anyway (vLLM has no SM121 wheels). Adding PR #38325 to that build is
effectively zero extra time. Net win for new users; existing users with
cached `vllm-sm121:latest` from before this flip can either pass `--no-cache`
to rebuild (and pick up PR #38325) or pass `--no-pr38325` to keep their
existing base as-is and only pick up the autotune (which is a 1-second
thin-layer change, no recompile).

## How to disable

`./install.sh --no-pr38325` skips the patch entirely — useful if it ever
breaks a build for a future eugr/spark-vllm-docker pin or future torch
nightly. Cost: lose the marginal +0.76%, keep autotune.

## When to remove

When vLLM ≥ 0.20.0 is our build target, this patch becomes redundant — PR
#38325 was merged into upstream 0.20.x. At that point delete this directory
and the `--no-pr38325` flag from `install.sh`.
