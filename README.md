# Qwen3.5-122B-A10B on DGX Spark: 28.3 → 51 tok/s (+80%)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Performance](https://img.shields.io/badge/tok%2Fs-51-brightgreen?style=flat&logo=speedtest&logoColor=white)](.)
[![Qwen3.5-35B](https://img.shields.io/badge/Qwen3.5--35B-112_tok%2Fs-00cc44?style=flat&logo=speedtest&logoColor=white)](.)
[![Speedup](https://img.shields.io/badge/speedup-%2B80%25-orange?style=flat)](.)
[![Hardware](https://img.shields.io/badge/NVIDIA-DGX_Spark-76B900?style=flat&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-Qwen3.5--122B--A10B-yellow)](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
[![Quantization](https://img.shields.io/badge/Quant-INT4%2BFP8_Hybrid-purple)](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound)
[![INT8 LM Head](https://img.shields.io/badge/LM_Head-INT8_Triton-blueviolet?style=flat)](.)
[![MTP-2](https://img.shields.io/badge/MTP--2-~80%25_accept-ff69b4?style=flat)](.)
[![Context](https://img.shields.io/badge/Context-256K-blue?style=flat)](.)
[![TurboQuant](https://img.shields.io/badge/TQ-4x_KV_cache-cyan?style=flat)](.)
[![vLLM](https://img.shields.io/badge/vLLM-0.19.1-red?style=flat)](https://github.com/vllm-project/vllm)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green?style=flat&logo=nvidia)](.)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](docker/Dockerfile.v2)

Optimizations for Qwen3.5-122B-A10B inference on a single NVIDIA DGX Spark from **28.3 to 51 tok/s** (+80%), with 256K context support, no quality degradation.

## Results

| Configuration | tok/s | Improvement | Build |
|---|---|---|---|
| Baseline (vLLM 0.19 + AutoRound INT4 + FlashInfer) | **28.3** | -- | -- |
| + Hybrid INT4+FP8 Dense Layers | **30.8** | +8.8% | step 1 |
| + MTP-2 Speculative Decoding | **38.4** | +35.7% | step 2 |
| **v2** (+ INT8 LM Head v2) | **51** | **+80%** | **`Dockerfile.v2`** |
| v2-tq (+ TurboQuant KV Cache) | 39 | +38% | `Dockerfile.v2-tq` |

The same optimizations also work with Qwen3.5-35B-A3B (same architecture, smaller): **112 tok/s**.

### 256K Context Support

v2 supports 256K context out of the box (355K token KV cache). No TurboQuant needed for single-user 256K.

| Config | KV Cache | Concurrent Users @ 256K |
|---|---|---|
| v2 (standard) | 355K tokens | 1 |
| v2-tq (TurboQuant) | 1.4M tokens | 5 |

---

## Quick Start

All optimizations are independent — pick what you need:

| Path | Steps | tok/s | What you get |
|---|---|---|---|
| **MTP only** (easiest) | 0 → 2 → 3 → 4 → 5 | ~44 | MTP-2 + INT8 LM Head, no hybrid |
| **Full v2** (recommended) | 0 → 1 → 2 → 3 → 4 → 5 | **51** | All optimizations |

### Automated install (TL;DR)

If you want everything done for you, just run:

```bash
./install.sh
```

This walks through Steps 0-4 automatically with progress bars, elapsed time, and a final prompt to launch the container. It is idempotent — re-running skips steps whose outputs already exist.

The script never invokes `sudo` itself: if a prerequisite is missing (`python3-venv`, docker daemon access, etc.) it prints the exact `sudo` command you should run, then exits non-zero so you can fix it and re-run.

Useful flags:

```bash
./install.sh --launch       # build, then auto-launch container (no prompt)
./install.sh --no-launch    # build only, never prompt for launch
./install.sh --no-cache     # nuke existing images + BuildKit cache and rebuild
                            # from scratch (use after a previous failed build)
./install.sh --help         # full flag reference
```

If you prefer to do it step-by-step manually (or want to understand what the script does), follow Steps 0-4 below — the script runs exactly the same commands.

Out of scope for `install.sh`: the TurboQuant variant (see "Optional: TurboQuant KV Cache Compression" later) and the runtime Step 6 benchmark.

---

### Host-side Python environment

Steps 0-2 run on the host (not inside Docker) and need a small set of Python packages. Pick whichever install style you prefer — both produce the same result.

**Option A — virtualenv (recommended, clean):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch numpy safetensors huggingface_hub
```

Keep this venv active for the rest of Steps 0-2. Step 3 onward runs inside Docker and does not use the host venv.

**Option B — system-wide (quick & dirty):**

```bash
pip install --break-system-packages torch numpy safetensors huggingface_hub
```

On recent Ubuntu (24.04+), a plain `pip install` is blocked by PEP 668, hence the `--break-system-packages` flag. This works fine for a one-shot model prep but pollutes your global Python — if you ever use this machine for anything else, prefer Option A.

> Note: `torch` and `numpy` are only needed by the Step 1 hybrid-checkpoint script (safetensors falls back to numpy when saving non-contiguous tensors). Steps 0 and 2 use just `huggingface_hub` and the stdlib. If you're taking the MTP-only path and skipping Step 1, you can drop `torch` and `numpy` from the install list.

### Step 0: Download the model

```bash
hf download Intel/Qwen3.5-122B-A10B-int4-AutoRound
INTEL_DIR=$(find ~/.cache/huggingface/hub/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/snapshots -maxdepth 1 -mindepth 1 -type d)
```

> **Note:** `huggingface_hub` 1.x renamed the CLI from `huggingface-cli` to `hf`. If you installed an older version (0.x), use `huggingface-cli download ...` instead.

### Step 1: Build hybrid checkpoint *(optional, +9%)*

Replaces BF16 shared expert weights with FP8 from the official Qwen checkpoint. Skip this if you just want MTP — the Docker image works with both hybrid and non-hybrid checkpoints (the FP8 dispatch simply doesn't activate if no FP8 weights are present).

```bash
python patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py \
    --gptq-dir "$INTEL_DIR" \
    --fp8-repo Qwen/Qwen3.5-122B-A10B-FP8 \
    --output ~/models/qwen35-122b-hybrid-int4fp8 \
    --force
```

Takes ~20 minutes. Output: ~71 GB. If you skip this step, use `$INTEL_DIR` as your model path in step 2 and 4.

### Step 2: Add MTP weights

Intel AutoRound ships `model_extra_tensors.safetensors` (4.8 GB, 785 MTP tensors) but **does not list them** in `model.safetensors.index.json`. The file is physically present, but vLLM reads the index to discover weights — so it never sees the MTP head. This script copies the file (if needed) and **adds the 785 tensor mappings to the index**, so vLLM loads them for speculative decoding.

```bash
# Target = hybrid checkpoint (step 1) or original Intel dir (if skipping step 1)
MODEL_DIR=~/models/qwen35-122b-hybrid-int4fp8  # or $INTEL_DIR

python patches/02-mtp-speculative/add-mtp-weights.py \
    --source "$INTEL_DIR" \
    --target "$MODEL_DIR"
```

### Step 3: Build base vLLM image for SM121

DGX Spark requires vLLM compiled for SM121 (Blackwell). Pre-built wheels from PyPI don't support this architecture. Use [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) at the exact commit we tested against:

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
git checkout 49d6d9fefd7cd05e63af8b28e4b514e9d30d249f

# Remove two "TEMPORARY PATCH" RUN blocks that curl-apply vLLM PRs 35568
# and 38919. Both target main-branch bugs that don't affect v0.19.0, and
# both PRs were force-pushed after 2026-04-04, so their current .diff no
# longer applies to v0.19.0 at all. Our reference image was built without
# them (verified by MD5 comparison against pristine v0.19.0).
sed -i '/# TEMPORARY PATCH for broken FP8 kernels/,/&& rm pr35568.diff/d' Dockerfile
sed -i '/# TEMPORARY PATCH for broken compilation/,/&& rm pr38919.diff/d' Dockerfile

./build-and-copy.sh -t vllm-sm121 --vllm-ref v0.19.0 --tf5
cd ..
```

This takes 30-60 minutes (compiles PyTorch + FlashInfer + Triton for SM121).

> **Why `build-and-copy.sh` and not `docker build` directly:** the upstream Dockerfile does `COPY build-metadata.yaml`, and that file is generated *at build time* by `build-and-copy.sh` (and removed after). A plain `docker build` will fail with `"/build-metadata.yaml": not found`.
>
> **Why `--tf5` and `--vllm-ref v0.19.0`:** our image was built with `transformers_5: true` and `vllm_ref: v0.19.0` (read from `/workspace/build-metadata.yaml` inside the image). The script's defaults are different, and an image built with defaults will not be binary-compatible with our patches.
>
> **Why the two `sed` commands:** the upstream Dockerfile at `49d6d9f` has two hardcoded `RUN curl ... .diff | git apply` blocks that pull vLLM PRs 35568 and 38919 live from GitHub. Both PRs target bugs in `main`, not in `v0.19.0`, and both were force-pushed after our original 2026-04-04 build, so their current diffs no longer apply to `v0.19.0` (they reference files moved under `csrc/libtorch_stable/` which didn't exist yet at that path in `v0.19.0`). We verified via MD5 comparison that `marlin_utils.py` inside our reference image is byte-identical to pristine `v0.19.0` — i.e. PR 35568 was never actually applied to our image in the first place. Removing the two blocks reproduces the original build.

> **Version pinning:** Tested and verified with **vLLM 0.19.1** (exact build: `0.19.1.dev0+g2a69949bd.d20260404`, commit `2a69949bd`, eugr/spark-vllm-docker commit `49d6d9f`). Patches are version-specific — **do not use with other vLLM versions** without re-testing. vLLM releases frequently and internal APIs change between minor versions.

### Step 4: Build v2 image

```bash
docker build -t vllm-qwen35-v2 -f docker/Dockerfile.v2 .
```

### Step 5: Launch

```bash
docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v ~/models:/models \
  vllm-qwen35-v2 \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

> If you skipped step 1, replace the model path with your Intel AutoRound directory.

Wait ~10 minutes for loading + warmup. Then:

```bash
curl localhost:8000/health
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
```

### Step 6: Benchmark

```bash
./bench_qwen35.sh "v2"
```

Expected: ~51 tok/s with hybrid, ~44 tok/s without hybrid (Run 2; Run 1 is JIT warmup).

### Running in Production

The launch command in Step 5 (and `install.sh`) is a minimal smoke test. For daily use, you'll likely want a custom launch script tailored to your setup. Here's a community example from [@whpthomas](https://github.com/whpthomas) that demonstrates common production flags:

**`run-qwen.sh`** — start the server:

```bash
#!/bin/bash
docker rm vllm-qwen35
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
docker run -it --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v ~/models:/models \
  vllm-qwen35-v2 \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen/qwen3.5 \
  --max-model-len 196608 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.88 \
  --port 8000 \
  --host 0.0.0.0 \
  --load-format fastsafetensors \
  --attention-backend FLASHINFER \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config auto \
  --override-generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0, "repetition_penalty": 1.0}'
```

**`stop-qwen.sh`** — stop the server:

```bash
#!/bin/bash
docker stop $(docker ps -q --filter "name=vllm")
```

Notable flags in this example:

| Flag | Purpose |
|---|---|
| `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` | Flush page cache before launch for consistent memory |
| `--host 0.0.0.0` | Listen on all interfaces (LAN access) |
| `--load-format fastsafetensors` | Faster checkpoint loading |
| `--enable-chunked-prefill` | Better TTFT on long prompts |
| `--enable-auto-tool-choice --tool-call-parser ...` | Agent/tool-calling support (see note below) |
| `--override-generation-config '{...}'` | Server-side sampling defaults |

> **Adapt to your needs.** These are suggestions, not requirements. The only flags critical for this project's optimizations are `--attention-backend FLASHINFER` and `--speculative-config`. Everything else depends on your use case. See [vLLM documentation](https://docs.vllm.ai/) for the full flag reference.
>
> **Note:** `--enable-prefix-caching` is intentionally omitted — it crashes on Qwen3.5 due to DeltaNet hybrid attention (see Troubleshooting).

**Tool-call parser options:** vLLM ships two parsers for Qwen models. Pick the one that matches your model:

| Parser | Format | Models |
|---|---|---|
| `qwen3_xml` | `<tool_call>{"name": "fn", "arguments": {...}}</tool_call>` (JSON in XML tags) | **Qwen3.5-\***, Qwen3-\*-Instruct |
| `qwen3_coder` | `<tool_call><function=fn><parameter=x>val</parameter></function></tool_call>` (custom XML) | Qwen3-Coder-\* |

For Qwen3.5-122B (this project), use `--tool-call-parser qwen3_xml`. The example above uses `qwen3_coder` which also works but is designed for Coder-series models.

> **Known issue (vLLM 0.19):** When `--reasoning-parser qwen3` and `--tool-call-parser` are both active, tool calls emitted inside `<think>` blocks may be silently dropped in non-streaming mode ([vllm#39056](https://github.com/vllm-project/vllm/issues/39056)).

### Troubleshooting

| Symptom | Fix |
|---|---|
| `health` returns nothing | Wait. It takes 10 minutes to load. |
| Garbage output | Ensure patched image, not vanilla vLLM |
| OOM at startup | Lower `--gpu-memory-utilization` to 0.85 |
| `content: null` | Normal for thinking models. Response is in `reasoning` field. |
| Only ~38 tok/s | Check `--speculative-config` has `num_speculative_tokens:2` |
| Stale Triton cache after rebuild | `docker exec <name> rm -rf /root/.cache/triton` and restart |
| MTP doesn't work with PyTorch backend | MTP requires FlashInfer backend (`--attention-backend FLASHINFER`). PyTorch backend is not supported. Also check vLLM version — MTP had bugs in pre-0.19 versions ([#36843](https://github.com/vllm-project/vllm/issues/36843), [#36917](https://github.com/vllm-project/vllm/issues/36917)). |
| Multi-node / Ray cluster issues | This project is tested on a **single DGX Spark only**. Multi-node setups (Ray, 2x Spark) have different requirements and are not covered here. Community reports suggest up to 56 tok/s on 2x Spark with Ray, but cluster configuration is outside our scope. |
| 408 "unexpected unmatched FP8 tensor" warnings during checkpoint build | **Normal.** These are DeltaNet linear_attn projections (36 of 48 layers) plus some attention norms/gates. They exist in the Qwen FP8 checkpoint but have no matching counterparts in Intel AutoRound INT4 (different naming conventions). The script only replaces shared_expert dense layers (144 tensors) with FP8 — everything else stays in its original format. Use `--force` to proceed. |

---

## Hardware

- **System:** NVIDIA DGX Spark (ASUS Ascent GX10)
- **GPU:** NVIDIA GB10 (Blackwell, SM121)
- **Memory:** 128 GB unified CPU-GPU (LPDDR5x, 273 GB/s)
- **CUDA:** 13.0
- **Architecture:** aarch64 (ARM Grace CPU)

## Tested Environment

These exact versions were used for all benchmarks. Mismatched versions may cause errors.

| Component | Version |
|---|---|
| **vLLM** | `0.19.1.dev0+g2a69949bd.d20260404` (commit `2a69949bd`) |
| **PyTorch** | `2.12.0.dev20260403+cu130` |
| **CUDA Toolkit** | 13.2 (V13.2.51) |
| **CUDA (torch)** | 13.0 |
| **FlashInfer** | 0.6.7 |
| **Triton** | 3.7.0 |
| **Python** | 3.12.3 |
| **OS** | Ubuntu 24.04.4 LTS (aarch64) |
| **Build flags** | `TORCH_CUDA_ARCH_LIST=12.1a` `FLASHINFER_CUDA_ARCH_LIST=12.1a` |

## Prerequisites

- vLLM 0.19.1 Docker image compiled for SM121 (see versions above)
- [Intel/Qwen3.5-122B-A10B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound)
- [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) (FP8 source for dense layers, optional if skipping hybrid)

### Building vLLM 0.19 for SM121 (DGX Spark)

DGX Spark uses the GB10 GPU (SM121 / Blackwell). vLLM doesn't ship pre-built images for this architecture, so you need to compile from source. The recommended approach is [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) which handles all SM121 specifics. Use the exact commit we tested against — upstream Dockerfiles change frequently:

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
git checkout 49d6d9fefd7cd05e63af8b28e4b514e9d30d249f
sed -i '/# TEMPORARY PATCH for broken FP8 kernels/,/&& rm pr35568.diff/d' Dockerfile
sed -i '/# TEMPORARY PATCH for broken compilation/,/&& rm pr38919.diff/d' Dockerfile
./build-and-copy.sh -t vllm-sm121 --vllm-ref v0.19.0 --tf5
```

Do not run `docker build` directly — the upstream Dockerfile `COPY`s a `build-metadata.yaml` file that only exists transiently during `build-and-copy.sh`. The two `sed` commands strip out upstream "TEMPORARY PATCH" RUN blocks that curl-fetch vLLM PRs 35568 and 38919 from live GitHub URLs; those PRs target main-branch bugs that don't apply to `v0.19.0`, and both were force-pushed after 2026-04-04. The flags `--vllm-ref v0.19.0` and `--tf5` match the `build_args` recorded inside our reference image (`vllm_ref: v0.19.0`, `transformers_5: true`).

If building manually, the critical environment variables are:

```bash
TORCH_CUDA_ARCH_LIST="12.1a"        # SM121 (Blackwell consumer)
FLASHINFER_CUDA_ARCH_LIST="12.1a"   # FlashInfer kernels for SM121
CUDA_HOME=/usr/local/cuda-13.0      # or 13.2
```

The base CUDA image should be `nvidia/cuda:13.2.0-devel-ubuntu24.04` (aarch64). Build takes 30-60 minutes on DGX Spark.

> **Note:** Pre-built vLLM wheels from PyPI do not support SM121. You must compile from source.

**For [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) users:** The hybrid patch is also available as a community mod (`enable-hybrid-int4fp8`). Apply it with `--apply-mod` in the launch script. Note that the `inc.py` patch is tied to a specific vLLM version — if you update the community Docker, the patch may need adjusting (the internal `inc.py` API changes frequently).

---

## Optimization Details

### Optimization 1: FlashInfer Attention Backend

**Effect:** 24.0 → 28.3 tok/s (+16%)

vLLM defaults to `FLASH_ATTN` on SM121. FlashInfer has optimized kernels that better utilize the Blackwell memory hierarchy. One flag, +16% free.

```bash
--attention-backend FLASHINFER
```

### Optimization 2: Hybrid INT4+FP8 Dense Layers

**Effect:** 28.3 → 30.8 tok/s (+8.8%)

MoE expert weights stay in INT4 (Marlin). Shared expert MLP weights replaced with FP8 from the official Qwen FP8 checkpoint, using native SM121 CUTLASS FP8 block-128 kernels.

The patch (`patches/01-hybrid-int4-fp8/inc.py`) fixes a bug where shared expert layers (marked as 16-bit by AutoRound) loaded FP8 weights without scale tensors.

### Optimization 3: MTP-2 Speculative Decoding

**Effect:** 30.8 → 38.4 tok/s (+25%)

Qwen3.5-122B ships with a native MTP head (785 tensors, 4.8 GB BF16). MTP-2 (`num_speculative_tokens:2`) predicts 2 additional tokens per step with ~80% acceptance rate on position 2.

**Why MTP-2, not MTP-1:** MTP-1 was the initial v1 configuration (38.4 tok/s). MTP-2 provides an additional +10% at no quality cost, with ~80% acceptance rate on position 2.

The MTP weights live in `model_extra_tensors.safetensors` in the Intel AutoRound checkpoint but are missing from the model index. The script `add-mtp-weights.py` registers all 785 tensors.

> **FAQ:** *Can I use MTP without the hybrid checkpoint?* Yes. `add-mtp-weights.py` only copies `model_extra_tensors.safetensors` and updates the index — it works on any Qwen3.5 checkpoint (INT4, FP8, or hybrid). The hybrid patch (step 1) and MTP (step 2) are independent optimizations.
>
> *What happens if I add `--speculative-config` without running `add-mtp-weights.py`?* vLLM won't find the MTP tensors and will either error out or silently disable speculative decoding (no speedup, no error — just no effect).

### Optimization 4: INT8 LM Head v2

**Effect:** 38.4 → 51 tok/s (+33%)

The LM Head (248K × 3072 = 729 MB) is the single largest BF16 bottleneck in the decode step. The v2 shared-weight Triton GEMV kernel:

1. **Quantizes BF16 → INT8 at runtime** (per-channel, no calibration needed)
2. **Single kernel launch** reads the 729 MB weight matrix ONCE per batch, regardless of batch size (the v1 kernel launched N times for N tokens)
3. **Triton INT8 GEMV** achieves 84% bandwidth utilization vs 24% for BF16 matmul

No quality degradation: INT8 per-channel quantization of the output layer preserves top-k token rankings.

---

## Optional: TurboQuant KV Cache Compression

[TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (Google, ICLR 2026) compresses the KV cache from bf16 to ~3.5 bits per element, giving **4x more KV cache capacity** (1.4M tokens vs 355K) at the cost of **-22% generation speed** (39 vs 51 tok/s).

### How it works

Each KV vector (head_size=256, bf16 = 512 bytes) is compressed to **120 bytes** using two techniques:

1. **MSE quantization**: Vectors are rotated via structured Hadamard transform, then each coordinate is quantized to the nearest Lloyd-Max centroid (3-bit for outlier dims, 2-bit for regular).
2. **QJL residual**: The quantization residual is encoded as 1-bit sign projections through a random matrix, preserving inner product estimates.

```
Packed format per KV position (120 bytes total):
  Group 0 (128 outlier dims): 48B MSE indices + 16B QJL signs + 4B norms = 68 bytes
  Group 1 (128 regular dims): 32B MSE indices + 16B QJL signs + 4B norms = 52 bytes

Compression: 512 bytes (bf16) → 120 bytes = 4.27x
```

The attention score is computed **directly on compressed data** without decompression:

```
score(q, k) = Σ_group [ vector_norm × (
    dot(codebook[mse_indices], F @ q) +           // MSE term
    residual_norm × dot(qjl_signs, G @ q × scale) // QJL term
)]
```

where `F` and `G` are structured Hadamard rotation matrices, and the codebook has only 8 entries (3-bit) or 4 entries (2-bit) — fits entirely in GPU registers.

### Why -22% speed is unavoidable

The performance penalty comes from two architectural changes:

1. **Triton replaces FlashInfer** (-15%): TQ requires a custom attention kernel that reads packed data. FlashInfer doesn't support custom KV formats, so TQ uses a Triton-based attention backend. Triton generates less optimized GPU code than FlashInfer's hand-tuned CUDA kernels.

2. **PIECEWISE instead of FULL CUDA graphs** (-7%): The Triton TQ kernel has JIT compilation behavior that's incompatible with FULL CUDA graph capture. PIECEWISE graphs add CPU launch overhead between kernel segments.

We also developed and benchmarked a **custom CUDA fused kernel** that computes attention directly on TQ packed data (in `patches/04-turboquant/cuda_tq_fused/`). After 8 optimization iterations (238x speedup from v1 to v8), the CUDA kernel achieves 38 GB/s effective bandwidth — but FlashInfer achieves 221 GB/s on the same hardware. The gap is fundamental: bit unpacking + codebook lookup requires ~1.5x more integer ALU operations per KV position than a simple bf16 dot product. With 4.3x less data but 1.5x more compute per byte, the net result is always slower than bf16 FlashInfer on bandwidth-bound SM121 hardware.

**Bottom line**: TQ is a memory-for-speed trade-off. The -22% penalty is the cost of 4x compression. There is no free lunch.

### When to use TQ

- **High-throughput serving**: 4x more concurrent users at 256K context (5 vs 1)
- **Not needed for 256K context on single user**: the standard v2 image already fits 256K with 355K token cache

### Important notes

- Qwen3.5 was tested by its developers on context lengths up to **256K only**. Longer contexts are not validated.
- The current INT4 quantization (Intel AutoRound) was calibrated for standard context lengths. For deeper contexts (>256K), a custom AutoRound calibration would be needed.
- The 1.4M token KV cache theoretically supports ~1M context, but model quality beyond 256K is unverified.

### TurboQuant Recipes

The `generate_tq_metadata.py` script supports multiple recipes. Choose based on your trade-off between memory, speed, and quality:

| Recipe | K Storage | V Storage | Memory vs fp16 | Outlier Dims | Best For |
|--------|-----------|-----------|----------------|--------------|----------|
| `turboquant35` | TQ35 | TQ35 | 4× (128B/head) | 50% K+V same | Default, balanced |
| `turboquant25` | TQ25 | TQ25 | 5× (108B/head) | 25% K+V same | Maximum compression |
| `turboquant_asym` | TQ35 | TQ35 | 4× (128B/head) | 50% K≠V disjoint | Better needle retrieval |
| `turboquant_q8k_tq35v` | int8 | TQ35 | ~2× (258B/head) | K=full, V=50% | Best quality |
| `turboquant_q8k_tq25v` | int8 | TQ25 | ~2× (258B/head) | K=full, V=25% | Quality + V compression |

**Recipe details:**

- **`turboquant35`** (default): Symmetric 3.5-bit encoding. First 50% of dimensions are high-precision outliers for both K and V. Best balance of compression and quality.

- **`turboquant25`**: More aggressive 2.5-bit encoding. Only 25% of dimensions are high-precision outliers. Higher compression ratio but more reconstruction error.

- **`turboquant_asym`** (Asymmetric): Same storage as TQ35 (128B/head), but K uses the first 50% of dimensions while V uses the last 50%. This disjoint selection decorrelates K and V index sets, improving reconstruction quality for long-context needle-in-haystack tasks without extra memory cost.

- **`turboquant_q8k_tq35v`** (True Asymmetric): K stored as int8 + fp16 scale (258 bytes/head), V stored as TQ35. ~2× memory overhead vs fp16 (vs 4× for symmetric TQ). K's full int8 precision preserves attention score quality, while V's TQ compression reduces the overall footprint. Recovers needle-in-haystack quality to 3/3 at 256K context (vs 0/3 with symmetric TQ35).

- **`turboquant_q8k_tq25v`**: Same as above but V uses TQ25 (25% outlier dims). Same memory as q8k-tq35v, more aggressive V compression.

### Build & Run (TQ variant)

```bash
# 1. Generate TQ metadata (one-time, places turboquant_kv.json in model dir)
#    Default: symmetric TQ35
python patches/04-turboquant/generate_tq_metadata.py \
    --model-dir ~/models/qwen35-122b-hybrid-int4fp8

#    Asymmetric (disjoint K/V outlier dims):
python patches/04-turboquant/generate_tq_metadata.py \
    --model-dir ~/models/qwen35-122b-hybrid-int4fp8 \
    --recipe turboquant_asym

#    Q8K (K=int8, V=TQ35 - best quality):
python patches/04-turboquant/generate_tq_metadata.py \
    --model-dir ~/models/qwen35-122b-hybrid-int4fp8 \
    --recipe turboquant_q8k_tq35v

# 2. Build TQ image
docker build -t vllm-qwen35-v2-tq -f docker/Dockerfile.v2-tq .

# 3. Run (match --kv-cache-dtype to the recipe used for metadata generation)
docker run -d --name vllm-qwen35-tq \
  --gpus all --net=host -v ~/models:/models \
  vllm-qwen35-v2-tq \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen --port 8000 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --kv-cache-dtype turboquant35 --enable-turboquant \
  --turboquant-metadata-path /models/qwen35-122b-hybrid-int4fp8/turboquant_kv.json \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

> **First launch is slow (~15-20 min vs ~10 min for v2).** TQ patches modify vLLM internals at startup, and the Triton decode kernels are JIT-compiled on first run. Subsequent launches with cached Triton kernels are faster.

> **Important:** The `--kv-cache-dtype` must match the recipe used when generating metadata. Mismatched recipes will cause runtime errors or degraded quality.

### TQ Benchmarks

| Config | tok/s | KV Cache | Concurrent @ 256K | Memory/head | Notes |
|---|---|---|---|---|---|
| v2 (standard) | **51** | 355K | 1 | 512B (fp16) | Baseline |
| v2-tq (TQ35) | 39 (-22%) | 1.4M | 5 | 128B | Best memory efficiency |
| v2-tq-asym | ~39 | 1.4M | 5 | 128B | Better long-context retrieval |
| v2-tq-q8k-tq35v | ~35 (-31%) | ~700K | ~3 | 258B | Best quality, 2× memory |

---

## What Didn't Work

We tested 20+ optimization approaches across speculative decoding, quantization, sparsity, kernel optimization, and system tuning. 5 worked (above), the rest didn't. Here's the full graveyard so you don't repeat our experiments.

### Speculative Decoding

**EAGLE-3** — *+10%, but loses to MTP-2.* EAGLE-3 requires downloading and storing separate draft model weights (~5 GB). MTP-2 uses the built-in MTP head (already in the checkpoint) and is both simpler and faster. Two patches were created for EAGLE-3 integration but ultimately abandoned since MTP-2 wins on every metric.

**KnapSpec Self-Speculative** — *Skipped after analysis.* KnapSpec uses the model itself as a draft by skipping layers. Math showed it's bandwidth-inferior to MTP: the draft forward pass reads ~75% of full model weights, while the MTP head reads only 4.4%. On a bandwidth-bound system like DGX Spark, this makes KnapSpec slower than MTP by design.

**MTP-1 → MTP-2** — *MTP-1 replaced, not failed.* MTP-1 (`num_speculative_tokens:1`) gave 38.4 tok/s. MTP-2 (`num_speculative_tokens:2`) gives 51 tok/s with ~80% acceptance rate on position 2. Strictly better — more tokens per step with no quality cost.

### Expert-Level Optimizations

**SERE Expert Re-routing** — *0% improvement.* SERE re-routes tokens from underutilized experts to similar popular ones. But Qwen3.5's 256 experts are extremely specialized — pairwise similarity is only ~0.02 (max 0.08). SERE needs coarser MoE architectures (8-16 experts) where redundancy exists.

**MoE Expert Pruning (MoE-Spec)** — *Skipped.* Since SERE proved all 256 experts are unique, pruning any of them would degrade quality. No intra-expert redundancy to exploit.

**2:4 Structured Sparsity** — *Skipped.* Expert FFN hidden size is only 512. Too small for structured pruning patterns (2:4 needs meaningful redundancy within weight matrices). Same conclusion as SERE/MoE-Spec: quality is sacred.

### LM Head / Output Layer

**AdaptiveSoftmax** — *0% improvement.* Mathematically correct idea (only compute logits for likely tokens), but the scattered memory reads for frequent-token subsets achieve only 2.5% bandwidth utilization. The INT8 GEMV approach (full matrix, 84% BW utilization) wins decisively by reading contiguously.

**INT8 Shared Expert** — *GIBBERISH output.* The shared expert weights are already in FP8 (from hybrid patch) with carefully calibrated per-tensor scales. Naive FP8→INT8 re-quantization with per-channel scales destroys those calibrations — produced garbage output. Never re-quantize calibrated FP8 weights to INT8.

### Quantization

**NVFP4 (RedHatAI)** — *-42% slower (16.6 tok/s).* SM121 doesn't have working FP4 CUTLASS kernels in vLLM yet, so it falls back to Marlin SM80 PTX which handles FP4 poorly. Waiting for vLLM PRs [#38957](https://github.com/vllm-project/vllm/pull/38957) and [#31607](https://github.com/vllm-project/vllm/pull/31607).

**FP8 KV Cache** — *+0.2 tok/s (negligible).* `--kv-cache-dtype fp8` adds almost nothing. SM121 lacks native FP8 attention kernels; the dtype conversion overhead eats any bandwidth savings. Not worth the risk of subtle accuracy issues.

**Abliterated (Uncensored) Model** — *OOM.* The abliterated variant needs 244 GB in BF16. DGX Spark has 128 GB + 56 GB swap = 184 GB. Doesn't fit.

### Kernel Optimizations

**Triton Native SM121 MoE Kernels** — *0% improvement.* We forced vLLM to use Triton-compiled native SM121 kernels instead of Marlin SM80 PTX for MoE expert GEMM. Exactly the same speed. The bottleneck is LPDDR5x memory bandwidth (273 GB/s), not compute. Both kernel implementations achieve the same memory throughput.

**Native SM121 FP4 CUTLASS** — *Not possible.* SM121 (consumer Blackwell) does NOT have WGMMA or `tcgen05.mma` tensor core instructions — those are datacenter-only (SM100/SM103). SM121 uses the same `mma.sync` as SM80 Ampere. The "3.65x speedup" reported on NVIDIA forums was on datacenter Blackwell, not DGX Spark.

**MARLIN_USE_ATOMIC_ADD** — *0% on single GPU.* `VLLM_MARLIN_USE_ATOMIC_ADD=1` is designed for multi-GPU tensor parallel. No effect on single-GPU SM121.

### System / Runtime

**CPU/JIT Overhead Reduction** — *Noise (+0.4%).* torch.profiler showed 183% CPU utilization (2 of 20 cores), GIL-bound. Pinning to big ARM cores gave +0.4% — within noise. The OS already schedules correctly on Grace CPU.

**Prefix Caching** — *Broken on Qwen3.5.* DeltaNet layers maintain recurrent state that conflicts with KV prefix caching. Enabling `--enable-prefix-caching` produces incorrect outputs. vLLM correctly disables it automatically for hybrid attention architectures. Experimental 'align' mode gave -2% (slightly slower).

**vLLM PR Cherry-picks** — *0% improvement.* PR #38990 (shared expert overlap): not applicable to v0.19.1, the bug was introduced in a later refactor. PR #37700 (FLA/TMA SM12x fix): applied cleanly, 0% speedup — DeltaNet layers aren't the decode bottleneck.

---

## SM121 Architecture Notes

- **ISA:** Same `mma.sync` as SM80 (Ampere). No datacenter-only tensor core instructions.
- **No native FP4:** FP4 tensor core ops are SM100/SM103 only (datacenter Blackwell).
- **Memory-bound at batch=1:** 273 GB/s LPDDR5x is the ceiling.
- **FlashInfer wins:** +16% over FlashAttention2. Always use `--attention-backend FLASHINFER`.

---

## Competitive Landscape (April 2026)

| Setup | tok/s | vs Ours |
|---|---|---|
| **This work (v2)** | **51** | -- |
| Previous best (v1, MTP-1 + Hybrid) | 38.4 | -25% |
| Intel AutoRound INT4 (vLLM, FlashInfer) | 28.3 | -45% |
| llama.cpp GGUF Q5_K | 23.0 | -55% |
| NVFP4 RedHatAI (vLLM) | 16.6 | -67% |
| Official Qwen GPTQ-Int4 (vLLM) | 15.0 | -71% |

---

## File Structure

```
.
├── README.md
├── bench_qwen35.sh                          # Benchmark script (5 tests × 2 runs)
├── LICENSE                                  # Apache 2.0
├── patches/
│   ├── 01-hybrid-int4-fp8/
│   │   ├── inc.py                           # Pre-patched vLLM INC module
│   │   ├── inc.py.patch                     # Diff for reference
│   │   └── build-hybrid-checkpoint.py       # Hybrid checkpoint builder (~20 min)
│   ├── 02-mtp-speculative/
│   │   └── add-mtp-weights.py               # Register MTP weights in model index
│   ├── 03-int8-lm-head/
│   │   └── patch_int8_lmhead.py             # INT8 LM Head v2 (runtime patch)
│   └── 04-turboquant/                       # Optional: TurboQuant KV cache
│       ├── generate_tq_metadata.py           # Generate turboquant_kv.json
│       ├── kv_cache_interface.py            # TQ-aware KV cache interface
│       ├── patch_turboquant_v2.py           # Main TQ patch script
│       ├── turboquant_kv_cache.py           # TQ layout, pack/unpack, codebooks
│       ├── turboquant_metadata.py           # Per-layer TQ config + JSON
│       ├── triton_turboquant_decode.py      # Triton decode attention kernel
│       ├── triton_turboquant_kv_update.py   # Triton encode (compress) kernel
│       ├── selector.py                      # TQ-aware attention backend selector
│       ├── triton_attn.py                   # Modified Triton attention backend
│       └── cuda_tq_fused/                   # Experimental: CUDA fused kernel
│           ├── tq_fused_decode.cu           # CUDA kernel (v8, SM121)
│           ├── tq_fused_decode.py           # Python wrapper + Hadamard transforms
│           ├── setup.py                     # CUDAExtension build
│           ├── lab_2_debug.py               # Correctness verification
│           ├── lab_2_perf.py                # Performance benchmark
│           └── lab_2_tq_fused_bench.py      # Full benchmark suite
├── docker/
│   ├── Dockerfile.v2                        # Main: vLLM + hybrid + INT8 LM Head
│   ├── Dockerfile.v2-tq                     # Optional: + TurboQuant
│   ├── entrypoint-v2.sh                     # Applies INT8 LM Head patch
│   └── entrypoint-v2-tq.sh                 # Applies INT8 LM Head + TQ patches
└── configs/
    ├── launch-baseline.sh                   # 28.3 tok/s (reference)
    ├── launch-hybrid.sh                     # 30.8 tok/s (hybrid only)
    ├── launch-v2.sh                         # 51 tok/s (production)
    └── launch-v2-tq.sh                     # 39 tok/s (TQ variant)
```

## Acknowledgments

- [rmstxrx/vllm-hybrid-quant](https://github.com/rmstxrx/vllm-hybrid-quant) for the hybrid quantization concept
- [Intel/Qwen3.5-122B-A10B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) for the optimized INT4 quantization
- [Qwen](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) for the official FP8 checkpoint
- [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) for the TurboQuant vLLM 0.19 integration
- [bjk110/spark_vllm_docker](https://github.com/bjk110/spark_vllm_docker/tree/feat/turboquant) for the original TurboQuant SM121 adaptation and CUDA WPH kernel
- [Google Research — TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) for the KV cache compression algorithm (ICLR 2026)
- [0xSero/turboquant](https://github.com/0xSero/turboquant) for Triton kernel reference implementation
- [vLLM](https://github.com/vllm-project/vllm) for the inference engine
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/365639) DGX Spark community for testing and feedback

## License

Apache 2.0, following the license of the original model.
