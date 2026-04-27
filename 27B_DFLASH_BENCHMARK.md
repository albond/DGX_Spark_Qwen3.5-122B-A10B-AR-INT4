# 27B vLLM-DFlash vs No-Spec Baseline (Single DGX Spark)

Date: 2026-04-26

## Goal

Determine whether vLLM-DFlash gives any speedup over vLLM with no speculative
decoding on the smaller Qwen3.5-27B INT4 target. The 122B-A10B comparison
(community data, dual-Spark FP8) showed DFlash and MTP-2 both landing ~46
tok/s — i.e. no DFlash win there. The 27B test is a clean two-way comparison
on a single Spark.

## Hardware / Software

| | |
|---|---|
| GPU | 1× NVIDIA GB10 (Blackwell SM121, 128 GB unified) |
| OS | Linux 6.17.0-1014-nvidia |
| Driver | CUDA 13.0 host / 13.2 in containers |
| Target | `Intel/Qwen3.5-27B-int4-AutoRound` (INT4 GPTQ-Marlin, 17.6 GiB resident) |
| Drafter | `z-lab/Qwen3.5-27B-DFlash` (5-layer BF16, 2.1 GB) |
| Bench harness | `bench_qwen35.sh` (5 prompt classes × 2 runs, T=0) |

| Run | Image | vLLM | Notes |
|---|---|---|---|
| Baseline (no spec) | `vllm-sm121:latest` | 0.19.1.dev0+g2a69949bd | FlashInfer, gpu-mem 0.85 |
| DFlash k=15 | `ghcr.io/aeon-7/vllm-dflash:latest` | 0.19.1rc1.dev110+gb55d830ec | FlashAttention-2, gpu-mem 0.85, max-num-seqs 4 |
| DFlash k=5 | same image | same | only `DFLASH_NUM_SPEC_TOKENS=5` differs |

Both vLLMs report the same `quantization=inc` autodetection on the Intel
AutoRound checkpoint and use `MarlinLinearKernel` for INT4 GEMM.
Drafter loads via the EAGLE pathway, sharing the target's `embed_tokens` /
`lm_head`, with auxiliary hidden states from layers `(1, 16, 31, 46, 61)`.

## Results: tok/s by prompt class

Two runs back-to-back. Run 1 of every spec-decode config has a cold-start
penalty on the first prompt (Q&A): the drafter's torch.compile/graph cache
warms up on the very first decode step. Run 2 numbers are the steady-state.

### Baseline (no spec)

| Prompt | run 1 | run 2 |
|---|---:|---:|
| Q&A | 13.0 | 13.3 |
| Code | 13.3 | 13.3 |
| JSON | 13.3 | 13.3 |
| Math | 13.3 | 13.3 |
| LongCode | 13.3 | 13.3 |
| **median** | **13.3** | **13.3** |

### DFlash k=5

| Prompt | run 1 | run 2 |
|---|---:|---:|
| Q&A | 6.7 (cold) | 39.3 |
| Code | 45.4 | 45.6 |
| JSON | 38.3 | 39.0 |
| Math | 36.1 | 36.5 |
| LongCode | 49.9 | 49.8 |
| **median (run 2)** | | **39.3** |

### DFlash k=15

| Prompt | run 1 | run 2 |
|---|---:|---:|
| Q&A | 6.5 (cold) | 45.8 |
| Code | 68.2 | 68.3 |
| JSON | 47.6 | 47.8 |
| Math | 43.5 | 43.5 |
| LongCode | 68.4 | 68.4 |
| **median (run 2)** | | **47.8** |

## Side-by-side (run-2 / steady-state)

| Prompt | baseline | k=5 | k=15 | k=15 vs baseline |
|---|---:|---:|---:|---|
| Q&A | 13.3 | 39.3 | 45.8 | **3.4× faster** |
| Code | 13.3 | 45.6 | 68.3 | **5.1× faster** |
| JSON | 13.3 | 39.0 | 47.8 | **3.6× faster** |
| Math | 13.3 | 36.5 | 43.5 | **3.3× faster** |
| LongCode | 13.3 | 68.4 | 68.4 | **5.1× faster** |
| **median** | **13.3** | **39.3** | **47.8** | **3.6× faster** |

`k=15 > k=5` on every class. Code and LongCode show the largest gains
(matching upstream block-diffusion expectations: repetitive structure means
the drafter accepts more tokens per verify step).

## Anomalies

1. **Run-1 Q&A cold start.** Both DFlash configs printed 6.5 / 6.7 tok/s on
   the very first prompt. Subsequent prompts in run 1 (and all of run 2) are
   at full speed. This is drafter graph warmup, not a real regression.
2. **Image-pull no surprises.** `ghcr.io/aeon-7/vllm-dflash:latest`
   (~18 GB) pulled clean, included MAX_JOBS=8 SM121 build, FlashInfer +
   FlashAttention-2 prebuilt, drafter via mounted HF cache.
3. **Entrypoint quirk.** The image's
   `/usr/local/bin/dflash-entrypoint.sh` re-downloads `MODEL_PATH` if it
   contains a `/` and isn't a local directory — i.e. passing
   `Intel/Qwen3.5-27B-int4-AutoRound` triggers a 20 GB
   `snapshot_download(local_dir=...)` even when the model is already in the
   mounted HF cache. Workaround: pass the absolute snapshot path
   (`/root/.cache/huggingface/hub/models--…/snapshots/<rev>`) in
   `MODEL_PATH` and `DFLASH_DRAFTER`.
4. **Snapshot-mount symlink trap.** Mounting only the snapshot directory
   (`-v <snapshots/abc>:/models/target`) breaks because the snapshot is
   symlinks pointing at `../../blobs/`. vLLM's loader follows the symlinks,
   gets ENOENT, and rejects the path. Fix: mount the entire
   `~/.cache/huggingface` and use absolute snapshot paths inside.
5. **No OOM, no crashes** on either backend. Target took 17.6 GiB (INT4) +
   3.2 GiB (drafter), KV cache got 71.5 GiB free with k=15 → 249,696
   tokens at max-model-len 32768 (12.06× concurrency).
6. **No FlashInfer with DFlash.** Per the upstream README, `--attention-backend
   flash_attn` is required because the DFlash drafter uses non-causal
   attention masks the FlashInfer prefill kernel cannot express. Confirmed:
   both k=5 and k=15 run on `FLASH_ATTN` (FlashAttention-2).

## Verdict

**vLLM-DFlash gives a real, large speedup on 27B**: a 3.6× median and up to
5.1× on code/long-code — clearly *not* a wash like the 122B-A10B case.
Recommended config on 27B is **k=15** (the upstream default), beating k=5
on every prompt class.

The 122B-A10B "DFlash and MTP both ~46 tok/s" result is therefore a
verifier-load specific phenomenon: at the 122B target, the per-step
verifier cost on a single (or even dual) Spark dominates and the ~5× theoretical
speedup collapses to parity with MTP-2. At the 27B target the verifier is
much cheaper relative to drafting, so the speedup survives.

This matches the z-lab paper's framing of DFlash as a drafting-bandwidth
optimization: drafting more tokens per pass only helps when verification
is not the bottleneck. On 27B-INT4 with a 5-layer BF16 drafter, that ratio
is favorable. On 122B-A10B (active 10B + MoE routing) it isn't.

---

## Manager-visible bottom line

Numbers (run-2 median): baseline **13.3 tok/s**, DFlash k=5 **39.3 tok/s**,
DFlash k=15 **47.8 tok/s**. Code/LongCode peak at **68.4 tok/s** with k=15
(5.1× over baseline). All on a single DGX Spark, INT4 target, BF16 5-layer
drafter, FlashAttention-2.

**Verdict ①: vLLM-DFlash works at small scale, the 122B problem is
verifier-load-specific, SGLang likely required.** DFlash on 27B is a clean
3–5× win and matches the z-lab paper's claims qualitatively (their B200
numbers are higher but the *speedup ratio* on Code/HumanEval lines up).
The 122B "wash" result is therefore not a vLLM-DFlash bug; it's the
verifier dominating at that scale on Spark hardware. If you want DFlash on
122B you probably need SGLang's spec-v2 + overlap-plan-stream path
(per upstream README) or a dual-Spark Ray setup like AoE's 56 tok/s run.
For a 27B production stack, vLLM-DFlash with k=15 on this same Spark is
already a viable config.
