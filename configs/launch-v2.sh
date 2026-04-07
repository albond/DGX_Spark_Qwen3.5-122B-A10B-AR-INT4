#!/bin/bash
# v2: Hybrid INT4+FP8 + INT8 LM Head v2 + MTP-2 + FlashInfer
# Result: 51 tok/s on DGX Spark (+80% vs baseline)
# Context: 256K (355K token KV cache)

docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v /path/to/models:/models \
  vllm-qwen35-v019-v2 \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
