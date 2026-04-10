#!/bin/bash
# v2-tq-q8k-tq35v: True Asymmetric TurboQuant — K=Q8 (int8), V=TQ35
#
# Storage format per KV head (head_size=256):
#   K slot: [scale_lo, scale_hi, val[0]..val[255]]  (258 bytes, int8 + fp16 scale)
#   V slot: TQ35 payload (128 bytes) + 130 zero-pad bytes
#   packed_dim = 258 bytes (K dominates; V zero-pads)
#
# Memory comparison (Qwen3.5-122B, head_size=256):
#   fp16 baseline:       512 bytes/head  (1×)
#   Symmetric TQ35:      128 bytes/head  (4× — best memory)
#   K=Q8 + V=TQ35:       258 bytes/head  (~2× — best quality)
#   K=Q8 + V=TQ25:       258 bytes/head  (~2× — same memory, more V compression)
#
# Why K=Q8, V=TQ?
#   From TurboQuant+ research: K is used for attention score computation (dot
#   product with Q), so high quantization error in K directly corrupts scores.
#   V is used for the weighted output sum, which averages over many heads and
#   is more robust to noise.  Keeping K at int8 quality while compressing V
#   with TurboQuant recovers needle-in-haystack quality to 3/3 at all context
#   lengths (vs 0/3 with symmetric TQ35 at 256k context).
#
# Decode path:
#   Uses Python fallback (_fallback_turboquant_attention): dequantize K from
#   int8 and V from TQ35, then run eager attention.  A custom Triton decode
#   kernel can be added as a follow-on for higher throughput.
#
# Prerequisites:
#   1. Generate Q8K metadata:
#        python patches/04-turboquant/generate_tq_metadata.py \
#            --model-dir /path/to/models/qwen35-122b-hybrid-int4fp8 \
#            --recipe turboquant_q8k_tq35v
#   2. Build the v2-tq Docker image (Dockerfile.v2-tq).

docker run -d --name vllm-qwen35-tq-q8k-tq35v \
  --gpus all --net=host --ipc=host \
  -v /path/to/models:/models \
  vllm-qwen35-v019-v2-tq \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --kv-cache-dtype turboquant_q8k_tq35v --enable-turboquant \
  --turboquant-metadata-path /models/qwen35-122b-hybrid-int4fp8/turboquant_kv.json \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
