#!/bin/bash
# v2-tq-q8k-tq25v: True Asymmetric TurboQuant — K=Q8 (int8), V=TQ25
#
# Like v2-tq-q8k-tq35v but with TQ25 for V (2.5 bpd instead of 3.5 bpd).
# V uses 25% outlier dims (64 high-precision + 192 low-precision per head).
# The cache packed_dim is still 258 bytes (K=int8 dominates); V's 80-byte
# TQ25 payload zero-pads to 258.
#
# Storage format per KV head (head_size=256):
#   K slot: [scale_lo, scale_hi, val[0]..val[255]]  (258 bytes, int8 + fp16 scale)
#   V slot: TQ25 payload (80 bytes) + 178 zero-pad bytes
#   packed_dim = 258 bytes
#
# Compared to v2-tq-q8k-tq35v:
#   Same memory usage (packed_dim unchanged — K dominates).
#   More V compression → potentially lower V reconstruction quality at very
#   long contexts.  Use this only if TQ35 V quality is acceptable to trade for
#   faster V decode (fewer bits to unpack per V head in future Triton kernel).
#
# Prerequisites:
#   1. Generate Q8K/TQ25 metadata:
#        python patches/04-turboquant/generate_tq_metadata.py \
#            --model-dir /path/to/models/qwen35-122b-hybrid-int4fp8 \
#            --recipe turboquant_q8k_tq25v
#   2. Build the v2-tq Docker image (Dockerfile.v2-tq).

docker run -d --name vllm-qwen35-tq-q8k-tq25v \
  --gpus all --net=host --ipc=host \
  -v /path/to/models:/models \
  vllm-qwen35-v019-v2-tq \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --kv-cache-dtype turboquant_q8k_tq25v --enable-turboquant \
  --turboquant-metadata-path /models/qwen35-122b-hybrid-int4fp8/turboquant_kv.json \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
