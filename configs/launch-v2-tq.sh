#!/bin/bash
# v2-tq: v2 + TurboQuant KV Cache Compression
# Result: 39 tok/s (-22% vs v2), but 4x more KV cache (1.4M tokens)
# Context: 256K with 5x concurrent capacity

docker run -d --name vllm-qwen35-tq \
  --gpus all --net=host --ipc=host \
  -v /path/to/models:/models \
  vllm-qwen35-v019-v2-tq \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --kv-cache-dtype turboquant35 --enable-turboquant \
  --turboquant-metadata-path /models/qwen35-122b-hybrid-int4fp8/turboquant_kv.json \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
