#!/bin/bash
# Baseline: vLLM 0.19 + Intel AutoRound INT4 + FlashInfer
# Result: 28.3 tok/s on single DGX Spark
#
# Prerequisites:
#   - vLLM Docker image compiled for SM121
#   - Intel/Qwen3.5-122B-A10B-int4-AutoRound downloaded to HuggingFace cache

sudo docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-qwen35-v019 \
  serve Intel/Qwen3.5-122B-A10B-int4-AutoRound \
  --served-model-name qwen \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER
