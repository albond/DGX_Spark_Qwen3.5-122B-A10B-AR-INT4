# SGLang DFlash on DGX Spark — Feasibility Assessment

> Research-only document. Compiled from upstream PRs/issues, NVIDIA developer-forum threads, and the live `lmsysorg/sglang:spark` / `scitrera/dgx-spark-sglang` images. No code was changed. No images were built. No servers were started.

## 1. Verdict

**Feasible-with-effort: ~3–5 human-days for a working 122B SGLang-DFlash prototype, ~7–10 human-days for a production-grade Docker image and benchmark sweep equivalent to the master branch.** The fundamentally hard pieces (SGLang on SM121, DFlash core in SGLang) are already solved by other people; what remains is integration and verification on our specific stack (122B + Hybrid INT4+FP8 + INT8 LM Head + DeltaNet).

The single load-bearing piece of evidence: a community member already ran SGLang 0.5.6 + DFlash PR #16818 on a DGX Spark with Qwen3-Coder-30B-A3B and reported **50.1 tok/s** in [DFlash on DGX — Every Token Rate Scaled by Six](https://forums.developer.nvidia.com/t/dflash-on-dgx-every-token-rate-scaled-by-six/359971). The DFlash code path runs. The hard blocker is `fa4` and `trtllm_mha`, not SGLang itself — both can be replaced by `flashinfer` with the (forecast) cost of giving up Modal Labs' dual-Spark numbers.

## 2. SM121 support status

SGLang on DGX Spark / GB10 / sm_121a is **functional but unmerged**. The reference Docker image is `lmsysorg/sglang:spark` (built off branch `yvbbrjdr:sglang:spark`); a more actively-maintained alternative is `scitrera/dgx-spark-sglang:0.5.10` (PyTorch 2.10 + CUDA 13.1.1 + Triton 3.6 + FlashInfer 0.6.3). Both build sgl-kernel from source against `TORCH_CUDA_ARCH_LIST=12.1a` and patch around three known holes: (a) the trtllm-gen FMHA backend has no SM121 cubins ([TensorRT-LLM #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799), open), (b) Triton attention + CUDA graphs crashes with illegal memory access on first warmup ([SGLang #19799](https://github.com/sgl-project/sglang/issues/19799), open — workaround is `--disable-cuda-graph` or use FlashInfer instead), (c) FA4 (CuTeDSL) requires `tcgen05.mma` which SM121 does not have ([SGLang #15342](https://github.com/sgl-project/sglang/issues/15342)). The umbrella tracking issue is [SGLang #11658](https://github.com/sgl-project/sglang/issues/11658) — listed PRs #11299/#11606 are not merged. PyTorch wheels are nightly-only on cu130/cu131.

Concrete blockers for our pinned `torch 2.12.0.dev20260408+cu130`: the lmsys spark branch is on torch 2.9.0; the scitrera image is on torch 2.10.0. **Neither matches our 2.12 nightly.** We would need to either (a) downgrade vLLM-side master to a torch 2.10/2.9 image and live with whatever ABI breakage that entails, or (b) rebuild SGLang and sgl-kernel against torch 2.12 and accept a 1–2 day debug loop on signature changes. (b) is the realistic path.

## 3. Attention backend status

| Backend | SM121 status | DFlash compatible? | Notes |
|---|---|---|---|
| `trtllm_mha` | **Blocked.** `ValueError: TRTLLM MHA backend is only supported on Blackwell GPUs (SM100)` ([SGLang #14814](https://github.com/sgl-project/sglang/issues/14814), [#9140](https://github.com/sgl-project/sglang/issues/9140)). Root cause: TensorRT-LLM ships no SM120/SM121 cubins ([TRT-LLM #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799)). | Yes, but unreachable on SM121. | This is what Modal Labs uses for the **target** path on B200. We cannot use it. |
| `fa4` | **Blocked.** Requires `tcgen05.mma` + TMEM (SM100 datacenter only). nvidia-cutlass-dsl 4.4.1 has no SM121 kernel images. Confirmed unreachable on GB10 ([SGLang #10564 discussion](https://github.com/sgl-project/sglang/discussions/10564), [#15342](https://github.com/sgl-project/sglang/issues/15342)). | Yes, but unreachable on SM121. | This is what Modal Labs uses for the **draft** path on B200. We cannot use it. |
| `fa3` | Available (Hopper-focused, also runs on Blackwell) but uses fallback paths on SM121. Acceptance and quality are fine; throughput penalty exists. | Yes — DFlash PR #22077 lists it as a supported draft backend. | Likely the realistic draft backend on our hardware. |
| `flashinfer` | **Working.** Already used by every NVIDIA-published DGX Spark + SGLang config (NVIDIA's own [build.nvidia.com/spark/sglang](https://build.nvidia.com/spark/sglang) playbook hard-codes `--attention-backend flashinfer`). | Yes, with caveat: FlashInfer's plan-stream introduces a sync that limits `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` benefit (per [SGLang attention-backend docs](https://docs.sglang.io/advanced_features/attention_backend.html)). | Most likely production choice. |
| `triton` | Working but with `--disable-cuda-graph` ([#19799](https://github.com/sgl-project/sglang/issues/19799)). Lossy on throughput because of CPU launch overhead. | Yes. | Fallback if flashinfer breaks. |
| `torch_native` (SDPA) | Always works. | **No.** SDPA backend has no speculative-decoding hook in SGLang. | Not viable for DFlash. |

**Net:** the canonical Modal/B200 launch (`trtllm_mha` + `fa4`) is exactly the pair we cannot run. The realistic single-Spark stack is `--attention-backend flashinfer --speculative-draft-attention-backend fa3`. We lose the trtllm-gen verifier and CuTeDSL drafter; gain unknown.

## 4. Architectural differences SGLang has that vLLM doesn't

1. **Spec V2 with overlap scheduler.** `SGLANG_ENABLE_SPEC_V2=1` (default in newer SGLang; old flag `--enable-beta-spec` deprecated) fuses output sync with the next batch's kernel launch — CPU bookkeeping is hidden inside the GPU forward. Per the [SGLang speculative-decoding docs](https://docs.sglang.io/advanced_features/speculative_decoding.html): *"output sync/processing is delayed while the next batch's kernels launch early, so CPU overhead for batching and syncing is hidden in GPU forward"*. Initial skeleton: PR #11398. Verified backends: TRTLLM MLA/MHA, FA3, Triton, Ascend. FlashInfer is "limited" because of the plan-stream sync. vLLM's scheduler still sequences draft→verify as separate steps; this is the gap dcw02 (Modal) is closing in [SGLang PR #23000](https://github.com/sgl-project/sglang/pull/23000) ("[Feature] Spec V2 DFlash Support", open). PR #23000 reports BS=1 going from 845 → 1075 tok/s on B200:8 just from the V2 plumbing.

2. **`SGLANG_ENABLE_OVERLAP_PLAN_STREAM`.** A separate CUDA stream prepares the next batch's metadata (KV-cache pages, attention-index buffers) while the current step's GEMMs run. Removes a sync point during draft-verify alternation. Not yet a stable feature — PR #23000 author flags it "experimental, may not be stable."

3. **`--mamba-scheduler-strategy extra_buffer`.** Qwen3.5's Gated Delta Networks layers (DeltaNet) carry recurrent state across speculative branches; "extra_buffer" allocates a ping-pong buffer per request so a rejected speculative branch can roll back without re-running mamba. Default `no_buffer` mode bans page-size > 1, overlap scheduling, and speculation. **For Qwen3.5-122B-A10B with DeltaNet layers, extra_buffer is mandatory if you want speculative decoding.** This is a real architectural commitment SGLang has made and vLLM has not — vLLM 0.20 simply disables prefix-caching on hybrid attention models and auto-falls-back to non-speculative mamba on rollback.

4. **Modal-Labs scheduler optimizations** (PR #23000 author dcw02 quote): *"rewrote the fused kv helper, added some new triton ops, removed some syncs"*. Concretely: a new fused KV-pack kernel for DFlash drafter outputs and removal of a CPU-GPU sync in the draft-token integration step. Not visible from the public PR alone; would need to read the diff to assess portability.

## 5. Portability of those features to vLLM

| Feature | Difficulty to port to vLLM | Reasoning |
|---|---|---|
| Spec V2 overlap scheduler | **Hard.** | Touches the engine core (`Scheduler`, `EngineCore`, `ModelRunner`). vLLM v1's scheduler is not a drop-in for SGLang's. Would essentially require rewriting vLLM's speculative engine; this is in progress upstream as part of vLLM's own scheduler work but not landed. ETA per vLLM roadmap: months, not weeks. |
| OVERLAP_PLAN_STREAM | **Medium.** | Cleanly factored as a separate CUDA stream + event. The hard part is identifying every sync point in vLLM's draft path; Modal's diff in #23000 is the reference. Maybe 1–2 weeks of focused work for a vLLM expert. |
| `--mamba-scheduler-strategy extra_buffer` | **Hard.** | vLLM's hybrid-attention path isn't designed to checkpoint+rollback recurrent state; it currently **disables** prefix-caching on Qwen3.5 entirely (we hit this on master — see "Prefix Caching — Broken on Qwen3.5"). Adding rollback buffers means refactoring vLLM's KV-cache manager. |
| Modal's fused-KV helper / sync removal | **Medium-easy.** | These are DFlash-specific kernel/scheduler tweaks. Portable in principle to PR #40898's draft path. Worth reading the diff once #23000 lands. |
| Verifier on trtllm_mha + draft on fa4 | **Impossible on SM121.** | Hardware-blocked, not engine-blocked. Both backends require SM100 instructions (`tcgen05.mma`, TMEM) that GB10 does not have. This is a fixed-cost gap regardless of which engine we run. |

## 6. Recommended next concrete action

**Before committing to building anything, run a 1-day spike:**

1. (2–3 hours) Pull `scitrera/dgx-spark-sglang:0.5.10`, launch `Qwen/Qwen3.5-35B-A3B` (smaller, faster iteration than 122B) with `--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.5-35B-A3B-DFlash --attention-backend flashinfer --speculative-draft-attention-backend fa3 --mamba-scheduler-strategy extra_buffer --speculative-num-draft-tokens 16`. Confirm the path the AEON-7 setup proved works on smaller hybrid Qwen3.5 actually launches.
2. (1 hour) Run `dflash.benchmark` against gsm8k. The expectation: 35B-A3B on SM121 SGLang will land somewhere between 50 and 90 tok/s based on the comparable 30B-Coder result and our master-branch 35B numbers (112 tok/s with the full vLLM v2 stack).
3. (2–3 hours) If (2) works: swap to Qwen3.5-122B-A10B + z-lab/Qwen3.5-122B-A10B-DFlash. Either succeeds (skip to step 4) or hits a memory / DeltaNet / mamba-scheduler bug. Either outcome is informative.
4. (a single decision point at end of day 1): if 122B-on-SGLang lands inside ±15% of our master's 51 tok/s, **not worth pursuing further** — DFlash on SGLang is not winning vs MTP-2 on this hardware once `trtllm_mha` and `fa4` are off the table. If it lands at 60+ tok/s, **yes, build the production image** (additional 5–7 days for hybrid INT4+FP8 patch port, INT8 LM Head v2 port, Dockerfile, install.sh, sweep harness, README — most of master's plumbing translates).

**Concretely, the spike command for step 1** (no edit, no build, just `docker run`):

```bash
docker run --rm -it --gpus all --net=host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  scitrera/dgx-spark-sglang:0.5.10 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3.5-35B-A3B-DFlash \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend flashinfer \
    --speculative-draft-attention-backend fa3 \
    --mem-fraction-static 0.75 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code
```

If `flashinfer` rejects DFlash topk (it shouldn't, per docs, but the plan-stream sync warning is real), fall back to `--attention-backend triton --disable-cuda-graph` and re-measure. If `fa3` rejects on SM121 (rare; fa3 generally works on Blackwell), fall back to `--speculative-draft-attention-backend triton`.

---

## Summary message to manager

**Recommend a one-day spike, not a multi-week build.** The narrative that "SGLang is the production framework" is true on B200 (where Modal Labs runs `trtllm_mha + fa4`) but functionally meaningless on DGX Spark — both of those backends are SM121-blocked at the hardware-instruction level (`tcgen05.mma`, TMEM), not at the SGLang level. The SGLang stack we'd actually run on a Spark is `flashinfer` verifier + `fa3` (or `triton`) drafter, which is structurally similar to what vLLM 0.20 + #40898 already does for us — minus the Spec V2 overlap scheduler, which is the one Modal-specific advantage that *might* survive the SM121 cliff.

Concrete prior art: someone already got SGLang DFlash running on a DGX Spark with Qwen3-Coder-30B-A3B (50.1 tok/s, lmsysorg/sglang:spark + flashinfer + DFlash PR #16818). That's the ground truth to validate against. If 35B-A3B on the scitrera image hits ~60–80 tok/s, the day-2 cost-to-build is justified. If it lands at ~45 tok/s (i.e. the FlashInfer/no-overlap-scheduler ceiling on SM121 is the binding constraint, not the engine), we close the experiment, document it as the negative result it is, and our `dflash-integration` branch ships as the final word on DFlash-for-Spark — fanservice variant of master, not a winner.

Total proposed budget: 1 day spike + decision; up to 7 days build if green-lit.
