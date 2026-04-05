#!/bin/bash
# Hybrid INT4+FP8: MoE experts in INT4 (Marlin), shared expert in FP8 (CUTLASS)
# Result: 30.8 tok/s on single DGX Spark (+8.8% vs baseline)
#
# Prerequisites:
#   - Hybrid checkpoint built with build-hybrid-checkpoint.py
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
  --attention-backend FLASHINFER
