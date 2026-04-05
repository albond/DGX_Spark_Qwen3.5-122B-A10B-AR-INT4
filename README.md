# Qwen3.5-122B-A10B on DGX Spark: 28.3 to 38.4 tok/s

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Performance](https://img.shields.io/badge/tok%2Fs-38.4-brightgreen?style=flat&logo=speedtest&logoColor=white)](.)
[![Speedup](https://img.shields.io/badge/speedup-%2B36%25-orange?style=flat)](.)
[![Hardware](https://img.shields.io/badge/NVIDIA-DGX_Spark-76B900?style=flat&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-Qwen3.5--122B--A10B-yellow)](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
[![Quantization](https://img.shields.io/badge/Quant-INT4%2BFP8_Hybrid-purple)](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound)
[![vLLM](https://img.shields.io/badge/vLLM-0.19.1-red?style=flat)](https://github.com/vllm-project/vllm)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green?style=flat&logo=nvidia)](.)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](docker/Dockerfile.hybrid)
[![MTP](https://img.shields.io/badge/MTP_Acceptance-~95%25-ff69b4)](.)

Three optimizations that push Qwen3.5-122B-A10B inference on a single NVIDIA DGX Spark from **28.3 to 38.4 tok/s** (+36%), with no quality degradation.

> "We didn't make the GPU faster. We just stopped wasting 36% of it." -- the authors, probably

## Results

| Configuration | tok/s | Improvement | Cumulative |
|---|---|---|---|
| Baseline (vLLM 0.19 + AutoRound INT4 + FlashInfer) | **28.3** | -- | -- |
| + Hybrid INT4+FP8 Dense Layers | **30.8** | +8.8% | +8.8% |
| + MTP-1 Speculative Decoding | **38.4** | +24.7% | +35.7% |

### Detailed Benchmark (Run 2, warm cache)

| Test | Baseline | Hybrid | Hybrid+MTP |
|---|---|---|---|
| Q&A (256 tok) | 28.3 | 30.8 | 37.8 |
| Code (512 tok) | 28.3 | 30.8 | 39.1 |
| JSON (1024 tok) | 28.4 | 30.9 | 39.0 |
| Math (64 tok) | 27.3 | 29.7 | 36.3 |
| Long Code (2048 tok) | 28.3 | 31.0 | 39.9 |

---

## Quick Start (I Just Want 38.4 tok/s)

For the impatient. You need a DGX Spark, Docker, and about 30 minutes of your life you won't get back (mostly waiting for model loading).

### Step 0: Get the models

```bash
# Download Intel AutoRound INT4 (~65 GB, go grab a coffee)
huggingface-cli download Intel/Qwen3.5-122B-A10B-int4-AutoRound

# Download Qwen FP8 (only need a few shards, ~8 GB)
# The build script handles partial download automatically
```

### Step 1: Build the hybrid checkpoint

```bash
# Find your Intel AutoRound snapshot path
INTEL_DIR=$(find ~/.cache/huggingface/hub/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/snapshots -maxdepth 1 -mindepth 1 -type d)

python patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py \
    --gptq-dir "$INTEL_DIR" \
    --fp8-model Qwen/Qwen3.5-122B-A10B-FP8 \
    --output-dir ~/models/qwen35-122b-hybrid-int4fp8
```

This takes ~20 minutes. It replaces 874 BF16 tensors with FP8 and adds 153 scale tensors. Output: ~71 GB.

### Step 2: Add MTP weights

Intel included MTP weights in the checkpoint but forgot to tell the model index about them. Classic.

```bash
python patches/02-mtp-speculative/add-mtp-weights.py \
    --source "$INTEL_DIR" \
    --target ~/models/qwen35-122b-hybrid-int4fp8
```

This copies `model_extra_tensors.safetensors` (4.8 GB, BF16) and updates the index with 785 MTP tensor mappings.

### Step 3: Build patched Docker image

```bash
# You need a vLLM 0.19.x image compiled for SM121 as the base.
# Replace "vllm-qwen35-v019:latest" in docker/Dockerfile.hybrid if yours has a different name.

docker build -t vllm-qwen35-hybrid -f docker/Dockerfile.hybrid .
```

### Step 4: Launch

```bash
sudo docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/models:/models \
  vllm-qwen35-hybrid \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Wait ~10 minutes for model loading + torch.compile + warmup. Then:

```bash
# Health check
curl localhost:8000/health

# Talk to it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
```

### Step 5: Benchmark

```bash
./bench_qwen35.sh "my-run"
```

You should see ~38 tok/s on Run 2 (Run 1 is cold cache, expect ~31-35).

### Troubleshooting

| Symptom | Fix |
|---|---|
| `health` returns nothing | Wait. Seriously. It takes 10 minutes. |
| Garbage output | Make sure you used the patched image, not vanilla vLLM |
| `weight_scale_inv not found` warnings | Patch not applied. Rebuild image from Step 3 |
| OOM at startup | Lower `--gpu-memory-utilization` to 0.85 |
| `content: null` in response | That's normal for thinking models -- the response is in `reasoning` field. Increase `max_tokens` |
| Only ~30 tok/s, not ~38 | Check you have `--speculative-config` in the launch command |
| Model loads but crashes on first request | New image, stale Triton cache. Run: `docker exec vllm-qwen35 rm -rf /root/.cache/triton` and restart |

---

## Hardware

- **System:** NVIDIA DGX Spark (ASUS Ascent GX10)
- **GPU:** NVIDIA GB10 (Blackwell, SM121)
- **Memory:** 128 GB unified CPU-GPU (LPDDR5x, 273 GB/s)
- **CUDA:** 13.0

## Prerequisites

- vLLM 0.19.1 Docker image compiled for SM121 (`TORCH_CUDA_ARCH_LIST=12.1a`)
- [Intel/Qwen3.5-122B-A10B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) (INT4 base)
- [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) (FP8 source for dense layers)

---

## Optimization Details

### Optimization 1: FlashInfer Attention Backend

**Effect:** 24.0 &rarr; 28.3 tok/s (+16%)

vLLM 0.19.1 defaults to `FLASH_ATTN` on SM121. Explicitly selecting FlashInfer gives a significant speedup. One flag, no code changes, +16% free performance.

```bash
--attention-backend FLASHINFER
```

FlashInfer has optimized attention kernels for SM121 that better utilize the memory subsystem. The default FLASH_ATTN backend was designed for Ampere/Hopper and doesn't fully leverage the Blackwell memory hierarchy.

### Optimization 2: Hybrid INT4+FP8 Dense Layers

**Effect:** 28.3 &rarr; 30.8 tok/s (+8.8%)

Replace BF16 shared expert MLP weights with FP8 (float8_e4m3fn) from the official Qwen FP8 checkpoint. MoE expert weights remain in INT4 (Marlin kernels). The FP8 dense layers use CUTLASS FP8 block-128 kernels which are native SM121.

```
MoE experts (256 per layer, top-8)  -> INT4 Marlin kernels (SM80 PTX)
Shared expert (gate_up, down x 48)  -> FP8 CUTLASS block-128 (native SM121)
Norms, gates, embeddings            -> BF16 (unchanged)
```

The patch (`patches/01-hybrid-int4-fp8/inc.py`) modifies vLLM's INC quantization config to:

1. **Detect FP8 layers** in the checkpoint by scanning safetensors metadata for float8_e4m3fn weights with weight_scale_inv tensors
2. **Dispatch FP8** for those layers instead of defaulting to `UnquantizedLinearMethod`

**Key bug fixed:** The original `get_quant_method()` had an early return for layers with `bits >= 16` that bypassed all FP8 dispatch. Shared expert layers (marked as 16-bit by AutoRound) loaded FP8 weights without scale tensors = garbage output. We've all been there.

### Optimization 3: MTP Speculative Decoding

**Effect:** 30.8 &rarr; 38.4 tok/s (+25%)

Qwen3.5-122B-A10B ships with a native MTP head (1 layer, 785 tensors, 4.8 GB in BF16). It predicts 1 additional token per step. The MTP weights live in `model_extra_tensors.safetensors` in the Intel AutoRound checkpoint, but Intel didn't add them to the model index. So vLLM never knew they existed.

**MTP acceptance rate: ~95%.** Out of every 2 tokens proposed (1 regular + 1 speculative), 1.95 are accepted on average. That's basically free tokens.

Multiple vLLM issues ([#36331](https://github.com/vllm-project/vllm/issues/36331), [#36872](https://github.com/vllm-project/vllm/issues/36872)) report MTP failures on Qwen3.5. The root cause is **corrupted MTP weights** from aggressive quantization (NVFP4), not DeltaNet rollback. Our MTP weights are in original BF16 precision, which is why they work.

---

## What Didn't Work

We tested 7 optimization approaches. 3 worked (above), 4 didn't. Here's the graveyard so you don't have to repeat our mistakes.

### FP8 KV Cache (+0.2 tok/s -- basically nothing)

`--kv-cache-dtype fp8` adds +0.2 tok/s. SM121 lacks native FP8 attention kernels; the dtype conversion overhead eats any memory bandwidth savings. Not worth the risk of subtle accuracy issues.

### Prefix Caching (broken)

DeltaNet layers maintain recurrent state that conflicts with KV prefix caching. Enabling it produces incorrect outputs. vLLM correctly disables it automatically for Qwen3.5 hybrid attention architectures.

### NVFP4 Quantization (16.6 tok/s -- 42% slower)

[RedHatAI/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4) sounds great on paper. In practice: SM121 doesn't have working FP4 CUTLASS kernels in vLLM yet, so it falls back to Marlin SM80 which handles FP4 poorly. Result: **16.6 tok/s**, less than half our baseline. Waiting for vLLM PRs [#38957](https://github.com/vllm-project/vllm/pull/38957) and [#31607](https://github.com/vllm-project/vllm/pull/31607).

### Triton Native SM121 MoE Kernels (0% improvement)

We forced vLLM to use Triton-compiled native SM121 kernels instead of Marlin SM80 PTX for MoE expert GEMM. Result: **exactly the same speed**. The bottleneck is LPDDR5x memory bandwidth, not compute. Both kernel implementations sit around waiting for memory equally well.

### vLLM PR Cherry-picks (0% improvement)

- **PR #38990** (shared expert overlap): Not applicable to v0.19.1 -- the bug was introduced in a later refactor.
- **PR #37700** (FLA/TMA SM12x fix): Applied cleanly, 0% speedup. DeltaNet layers aren't the bottleneck here.

### Native SM121 FP4 CUTLASS Kernels (impossible)

SM121 (consumer Blackwell) does **not** have WGMMA or `tcgen05.mma` tensor core instructions. Those are datacenter-only (SM100/SM103). SM121 uses the same `mma.sync` as SM80 Ampere. That "3.65x speedup" on NVIDIA forums? Datacenter Blackwell. Not us. Not today.

### Abliterated Model (OOM)

The uncensored variant needs 244 GB in BF16. DGX Spark has 128 GB + 56 GB swap = 184 GB. Math doesn't work out. The model is not uncensored, and neither is our RAM budget.

---

## SM121 Architecture Notes

Things we learned the hard way about SM121 (DGX Spark GB10):

- **ISA:** Same `mma.sync` as SM80 (Ampere). No fancy new tensor core instructions. Sorry.
- **No native FP4:** Despite being "Blackwell", FP4 tensor core ops are datacenter-only (SM100/SM103).
- **Memory-bound at batch=1:** 273 GB/s LPDDR5x is the ceiling. Faster kernels don't help when the GPU is waiting for data.
- **Native SM121 cubins exist:** vLLM built with `TORCH_CUDA_ARCH_LIST=12.1a` has 43 native cubins. But Marlin INT4 stays SM80 PTX by design (hand-written assembly).
- **FlashInfer wins:** +16% over FlashAttention2. Always use `--attention-backend FLASHINFER`.

---

## Competitive Landscape (April 2026)

| Setup | tok/s | vs Ours |
|---|---|---|
| **This work (Hybrid+MTP)** | **38.4** | -- |
| Intel AutoRound INT4 (vLLM, FlashInfer) | 28.3 | -26% |
| Community Hybrid GPTQ+FP8 (vLLM 0.17) | 21.5 | -44% |
| llama.cpp GGUF Q5_K | 23.0 | -40% |
| Ollama Q4_K_M | 18.9 | -51% |
| NVFP4 RedHatAI (vLLM) | 16.6 | -57% |
| Official Qwen GPTQ-Int4 (vLLM) | 14.0 | -64% |

Based on publicly available benchmarks as of April 2026, **38.4 tok/s is the fastest reported single-user generation speed for Qwen3.5-122B-A10B on a single DGX Spark.** If you've beaten this, we'd love to hear about it.

---

## File Structure

```
.
├── README.md                              # This file (you are here)
├── bench_qwen35.sh                        # Benchmark script (5 tests x 2 runs)
├── patches/
│   ├── 01-hybrid-int4-fp8/
│   │   ├── inc.py                         # Pre-patched vLLM INC quantization module
│   │   ├── inc.py.patch                   # Diff for the curious
│   │   └── build-hybrid-checkpoint.py     # Hybrid checkpoint builder
│   └── 02-mtp-speculative/
│       └── add-mtp-weights.py             # Add MTP weights to checkpoint
├── docker/
│   └── Dockerfile.hybrid                  # Build patched vLLM image
└── configs/
    ├── launch-baseline.sh                 # 28.3 tok/s
    ├── launch-hybrid.sh                   # 30.8 tok/s
    └── launch-hybrid-mtp.sh              # 38.4 tok/s
```

## Acknowledgments

- [rmstxrx/vllm-hybrid-quant](https://github.com/rmstxrx/vllm-hybrid-quant) for the hybrid quantization concept and initial build script
- [Intel/Qwen3.5-122B-A10B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) for the optimized INT4 quantization (and the MTP weights they hid in there)
- [Qwen](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) for the official FP8 checkpoint with calibrated scales
- [vLLM](https://github.com/vllm-project/vllm) for the inference engine that made this all possible

## License

This project is licensed under **Apache 2.0**, following the license of the original model.

- **[Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)** — Apache 2.0
- **[Intel/Qwen3.5-122B-A10B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound)** — INT4 quantization by [intel/auto-round](https://github.com/intel/auto-round); follows the license of the original model (Apache 2.0)

The patches, scripts, and documentation in this repository are provided under the same Apache 2.0 license.
