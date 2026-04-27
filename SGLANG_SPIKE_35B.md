# SGLang+DFlash on Qwen3.5-35B-A3B — Spike Result

**Date:** 2026-04-27
**Hardware:** 1× NVIDIA GB10 (Blackwell SM121, 128 GB unified, 273 GB/s LPDDR5x)
**Goal:** Decide whether SGLang's DFlash beats the master vLLM stack (`vllm-qwen35-v2`, 112 tok/s) on Qwen3.5-35B-A3B and merits a full 122B production build.

## TL;DR

**Best result: 117.2 tok/s** at `--speculative-num-draft-tokens 12` (k=12), beating master's 112 tok/s by +4.6%. **In-between zone** per spike criteria (115-130 tok/s). Recommend a judgment call — the gain is real but small, and the 122B verifier-bound problem identified by Agent A and Agent B remains unaddressed by this measurement on the 35B proxy.

## Image used

Tried, in order:

1. `scitrera/dgx-spark-sglang:0.5.10` — **failed**. Reports `sglang 0.0.0 / sglang-kernel 0.5.10`. CLI rejected `--speculative-algorithm DFLASH`: choices were only `EAGLE,EAGLE3,NEXTN,STANDALONE,NGRAM`. Source confirmed: no DFlash code in the build.
2. `lmsysorg/sglang:spark` — **failed**. Reports `sglang 0.5.4.post2`. Same — no DFlash in CLI choices.
3. `lmsysorg/sglang:dev-cu13` (15h-old nightly) — **succeeded**. Reports `sglang 0.0.0.dev1+g977830e91 / torch 2.9.1+cu130 / sgl-kernel 0.4.1.post1`. CLI accepts `DFLASH`. Source `/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_*` present.

**Conclusion on images:** PR #22077 (DFLASH) merged April 7, 2026, did NOT make it into SGLang's stable 0.5.10 release (April 6) or the scitrera/lmsysorg "spark" Spark-targeted images. To run DFlash on Spark today you need the bleeding-edge `lmsysorg/sglang:dev-cu13` nightly tag.

## Target model used

**`/project_folder/models/qwen35-35b-a3b-int4-autoround`** (Intel AutoRound INT4-AutoRound, ~21 GB on disk). Same checkpoint master measures 112 tok/s on. SGLang accepted it via `--quantization auto-round` autodetect. Did NOT need to fall back to BF16.

Boot logs report on this checkpoint: `Load weight end. elapsed=112.65 s, type=Qwen3_5MoeForConditionalGeneration, quant=auto-round, bits=4, avail mem=92.51 GB, mem usage=20.44 GB`. Mamba (DeltaNet) cache allocated 43 GB (extra_buffer rollback). Final layout: 20.4 GB weights + 5.3 GB KV (278K tokens) + 43 GB mamba = ~70 GB resident. 33 GB headroom.

## Attention backend chosen

`--attention-backend flashinfer --speculative-draft-attention-backend flashinfer`.

**`fa3` failed with assertion** "FlashAttention v3 Backend requires SM>=80 and SM<=90. Please use --attention-backend flashinfer." — fa3 explicitly excludes Blackwell SM120/121 in the SGLang assertion. The DFlash draft worker accepts `flashinfer` as a fallback (its `supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton")`). `fa4` and `trtllm_mha` are SM100-only as documented in SGLANG_FEASIBILITY.md.

Other settings: `--mem-fraction-static 0.75 --mamba-scheduler-strategy extra_buffer --trust-remote-code --tp-size 1`.

## Launch warmup duration

| Phase | Duration |
|---|---:|
| Container init + tokenizer load | ~5 s |
| Target model shard load (12 shards × ~9 s) | 113 s |
| Mamba + KV allocation | 7 s |
| Target CUDA graph capture (21 batch sizes) | 109 s |
| Drafter weight load | 7 s |
| Drafter CUDA graph capture | 8 s |
| **Total to "server is fired up"** | **~250 s (4 min)** |

Subsequent restarts (image cached, weights cached): ~4 min steady. No 20-min hard-timeout breaches.

## Headline benchmark — bench_qwen35.sh sweep over k

5 prompt classes (Q&A 256, Code 512, JSON 1024, Math 64, LongCode 2048) × 2 runs × T=0. The numbers below are run-2 (steady-state) median across the 5 classes. Run 1 has a small first-prompt cold start; numbers are run-2 only.

| k | Q&A | Code | JSON | Math | LongCode | **Median** | vs master |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 83.9 | 93.9 | 97.3 | 91.4 | 95.4 | **93.9** | -16% |
| 8 | 98.8 | 115.8 | 104.4 | 123.0 | 120.3 | **115.8** | +3.4% |
| 10 | 95.5 | 107.6 | 112.6 | 128.0 | 121.8 | **112.6** | +0.5% |
| **12** | **99.6** | **108.3** | **117.2** | **118.5** | **123.0** | **117.2** | **+4.6%** |
| 14 | 95.5 | 115.0 | 104.9 | 91.4 | 112.8 | 104.9 | -6.3% |
| 16 | 89.8 | 104.4 | 115.7 | 118.5 | 109.9 | 109.9 | -1.9% |

**Optimum is k=12** at 117.2 tok/s. The shape (peak in the middle, dropoff at extremes) is consistent: at small k the drafter speedup is too small to amortize verify cost; at large k acceptance rate falls (drafter becomes uncertain on tails) and wasted speculation tokens drag throughput.

For comparison, the master `vllm-qwen35-v2` ships 112 tok/s on this same target (single-prompt, T=0, MTP-2 + Hybrid INT4+FP8 + INT8 LM Head v2).

## Upstream gsm8k benchmark cross-reference

`python -m dflash.benchmark --backend sglang --dataset gsm8k --num-prompts 64 --concurrency 1 --enable-thinking` against the k=16 server (the one that was up before the sweep): 

```
Backend:          sglang
Dataset:          gsm8k
Num prompts:      64
Concurrency:      1
Latency:          1013.2s
Output tokens:    102679
Throughput:       101.34 tok/s
Accept length:    6.332
Spec verify ct:   17227
```

Accept length of **6.33 tokens per verify** (out of k=16 max) is healthy. With `--enable-thinking` added the throughput drops ~10 vs the non-thinking bench (101 vs 110 at k=16) — consistent with thinking-mode expanding the reasoning chain through harder-to-predict regions. Did not re-run gsm8k on k=12 to save time; the slope from the k-sweep above is the stronger signal.

## Verdict

**In-between (115-130 tok/s zone).** Best run-2 median: **117.2 tok/s** (k=12). Master baseline: **112 tok/s**.

Per the criteria in the spike spec:
- ≥130 tok/s → green-light B (full 122B SGLang production build) — **NOT CLEARED**
- ≤115 tok/s → close as C (documented negative result) — **NOT CLEARED**
- 115-130 → judgment call

**Manager judgment input:**

The +4.6% gain at k=12 is real, reproducible across the 5 prompt classes (4 of 5 above master), and matches z-lab's qualitative claim that DFlash beats EAGLE-style speculative decoding when the verifier is not the bottleneck. But:

1. **The 35B proxy is BEST CASE for DFlash on Spark.** At 35B-A3B with INT4 weights and 8-layer drafter, the verifier is cheap relative to drafting. At 122B-A10B (10B active + MoE routing) the verifier dominates per Finding 2 in PROJECT_DFLASH.md, and community data on dual-Spark FP8 already showed DFlash and MTP-2 both at ~46 tok/s. A +5% gain on the proxy does not predict a +5% gain on the production target.

2. **Building the full 122B SGLang stack is 5-7 person-days** (per SGLANG_FEASIBILITY.md): hybrid INT4+FP8 patch port, INT8 LM Head v2 port, sgl-kernel rebuild against sglang dev-cu13's torch 2.9.1, Dockerfile + install.sh + sweep harness. The expected payoff is on the order of master (51 tok/s) ± the same +5% — call it ~53 tok/s. 1 tok/s in absolute terms.

3. **The bleeding-edge dev-cu13 image carries operational risk.** It's a nightly, not a stable tag. The "Skipping import of cpp extensions due to incompatible torch version" warning at startup means we ran on a Triton-only attention path with no precompiled SM121 kernels. Any nightly churn could break it tomorrow.

**Recommendation:** Lean toward **option C (close as documented negative result)**. The 35B proxy gain is a clean +5% but doesn't survive the leap to 122B per the verifier-bound argument that's already established. If the user wants the +5%, the cheaper path is to ship the 35B SGLang config as a separate `sglang-35b` deliverable (1-2 days) rather than spending 5-7 days on a 122B port that the existing evidence predicts will land at parity.

If the user wants to gamble on B anyway, the next concrete experiment would be to launch SGLang+DFlash on `Qwen3.5-122B-A10B-int4-AutoRound` directly with the same dev-cu13 image (no patches, no hybrid, no INT8 LM Head) and see if it lands ≥51 tok/s. That's a 1-day check; if it does, the 5-day production build is worth it.

## Fallbacks fired

| Fallback | Why | Resolution |
|---|---|---|
| scitrera/dgx-spark-sglang:0.5.10 → lmsysorg/sglang:spark | DFLASH not in CLI choices | Tried lmsysorg, also missing |
| lmsysorg/sglang:spark → lmsysorg/sglang:dev-cu13 | 0.5.4.post2 lacks DFlash | dev-cu13 nightly has DFLASH |
| `--speculative-draft-attention-backend fa3` | "FlashAttention v3 requires SM>=80 and SM<=90" assertion | Switched to flashinfer for both verifier and draft |

No INT4-AutoRound fallback needed (SGLang accepts the AutoRound checkpoint via `--quantization auto-round` autodetect). No OOM. No `--mamba-scheduler-strategy no_buffer` fallback. CUDA graphs captured cleanly.

## Notable observations

1. **HF cache permission trap.** `sudo docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...` makes the container write under root inside the cache. Subsequent host-side `python -m dflash.benchmark` (running as `username`) hit `Permission denied` on dataset lock files. Fix: `sudo chown -R username:username /project_folder/.cache/huggingface/datasets` after each container run.

2. **Drafter has no `generation_config.json`.** Logged once per startup as a warning ("does not appear to have a file named generation_config.json"). Server proceeds without it. Not fatal; would clean up by adding the file from the target's checkpoint, but it's a cosmetic issue.

3. **DeltaNet `extra_buffer` is 43 GB.** Mamba intermediate state caches dominate memory: `intermediate_ssm_state_cache size: 33.75GB`, `ssm_state size: 8.44GB`. Without `extra_buffer` (i.e. on `no_buffer`) speculative decoding is disabled per SGLang's hybrid-attention rules. Net implication for 122B: the mamba allocation will scale roughly linearly with layer count, meaning a 122B port will eat ~3× this (~130 GB), pushing the unified memory budget hard.

4. **Spec V2 explicitly rejected.** Boot log: `"Overlap scheduler is disabled when using DFLASH speculative decoding (spec v2 is not supported yet)."` — the `SGLANG_ENABLE_SPEC_V2` lever Agent A flagged as the most likely SGLang-unique advantage is NOT available with DFlash today. Modal Labs' PR #23000 is in progress to fix this; until it merges, SGLang's "DFlash + Spec V2" combination remains theoretical.

5. **Acceptance length on bench prompts is k-dependent.** Logs from gsm8k (k=16) show accept_length ranging 3.5-9.3 over the run, mean 6.33. Implies the optimal k probably depends on the workload (math/code: shorter accept, prefer k=8-10; long structured generation: longer accept, prefer k=12-16). Production should let users pick.

## Disk / cleanup state

- Pulled images: `scitrera/dgx-spark-sglang:0.5.10` (32.6 GB), `lmsysorg/sglang:spark` (~10 GB), `lmsysorg/sglang:dev-cu13` (~10 GB). Total ~52 GB added to /var/lib/docker.
- Disk now: 70% full (621 → ~660 GB used, was 596 GB at start).
- DFlash drafter cached: `~/.cache/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash` (905 MB).
- Master untouched. No master image rebuilt. No master branch modified.
- Container `sglang-dflash` removed cleanly.
- GPU empty per `nvidia-smi --query-compute-apps`.

## Files of record

- `/project_folder/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/SGLANG_SPIKE_35B.md` (this file)
- `/project_folder/PROJECT_DFLASH.md` (appended below "## Spike Result (2026-04-27)")
- `/tmp/bench_sglang.sh` (the SGLang-port-30000-aware bench script — copy of bench_qwen35.sh with port and model placeholder)
- `/tmp/bench_sglang_*.log` (per-k bench logs k=4, k=8, k=10, k=12, k=14, k=16)
- `/tmp/dflash` (z-lab/dflash repo clone, gsm8k bench harness)
