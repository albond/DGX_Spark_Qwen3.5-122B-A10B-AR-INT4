#!/bin/bash
# v2-tq-asym: v2 + Asymmetric TurboQuant KV Cache
#
# Asymmetric TurboQuant (turboquant_asym):
#   K cache: first 128 dims as high-precision outliers  (head_size=256, TQ35 layout)
#   V cache: last  128 dims as high-precision outliers  (disjoint from K)
#
# Why asymmetric?
#   Symmetric TQ applies the same outlier-dim selection to both K and V.
#   K is used to compute attention scores (dot-product), so covering the first
#   half of dimensions (often where most variance lives) gives good score quality.
#   V is used to compute the output weighted sum.  Its important dimensions may
#   differ from K's — using the *last* half decorrelates the two selections and
#   improves reconstruction quality for V without any extra storage cost.
#
#   This matches the asymmetric recommendation from TurboQuant+ research:
#   keep K at higher effective quality, compress V more aggressively.
#   Both K and V are stored with the TQ35 format (128 bytes/head, 3.5 bpd).
#   No kernel changes are required; only different metadata indices.
#
# Result: same memory as v2-tq (1.4M token KV cache), better retrieval quality
#   vs symmetric TQ (especially for long-context needle-in-haystack tasks).
#
# Prerequisites:
#   1. Generate asymmetric metadata:
#        python patches/04-turboquant/generate_tq_metadata.py \
#            --model-dir /path/to/models/qwen35-122b-hybrid-int4fp8 \
#            --recipe turboquant_asym
#   2. Build the v2-tq Docker image (Dockerfile.v2-tq).

docker run -d --name vllm-qwen35-tq-asym \
  --gpus all --net=host --ipc=host \
  -v /path/to/models:/models \
  vllm-qwen35-v019-v2-tq \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --kv-cache-dtype turboquant_asym --enable-turboquant \
  --turboquant-metadata-path /models/qwen35-122b-hybrid-int4fp8/turboquant_kv.json \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
