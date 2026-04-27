#!/bin/bash
# v2-dflash: Hybrid INT4+FP8 + INT8 LM Head v2 + DFlash speculative + FlashAttention
# Replaces MTP with DFlash (block-diffusion drafter) — see README "DFlash benchmarks".
#
# Notes:
#   - num_speculative_tokens: 5 is the recommended value on this hardware. The
#     upstream default is 15, but at 122B-A10B on a single Spark the verifier
#     becomes the bottleneck and k=15 lands at 27.2 tok/s vs k=5's 43.4 tok/s.
#     See the README "DFlash Sweep Results" section for measured numbers.
#   - --attention-backend MUST be FLASH_ATTN (not FLASHINFER): the DFlash drafter
#     uses non-causal attention masks the FlashInfer prefill kernel cannot express.
#   - --gpu-memory-utilization 0.85 (vs 0.90 for MTP): the DFlash drafter is a
#     separate ~1 GB BF16 model that needs its own KV cache + activations.
#   - --max-num-batched-tokens 4096: required at k=5 to give the scheduler enough
#     headroom for draft-token slots. Without it vLLM 0.20 dies at startup with
#     "max_num_scheduled_tokens is set to -1536".
#   - DFlash drafter weights resolve from the HF hub on first launch unless you
#     pre-download to a local dir and pass that path in "model". The drafter
#     ships only the 5 transformer layers it needs; embed_tokens and lm_head are
#     shared with the target at runtime (vLLM logs this on startup).

docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/models:/models \
  vllm-qwen35-v2-dflash \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --attention-backend FLASH_ATTN \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-122B-A10B-DFlash","num_speculative_tokens":5}'
