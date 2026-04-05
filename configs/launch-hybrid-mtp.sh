#!/bin/bash
# Hybrid INT4+FP8 + MTP-1 Speculative Decoding
# Result: 38.4 tok/s on single DGX Spark (+25% vs hybrid, +36% vs baseline)
#
# Prerequisites:
#   - Hybrid checkpoint with MTP weights (add-mtp-weights.py)
#   - Patched vLLM Docker image (Dockerfile.hybrid)
#   - Checkpoint at /path/to/models/qwen35-122b-hybrid-int4fp8

sudo docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/models:/models \
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
