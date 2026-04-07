#!/bin/bash
set -e
echo "[v2] Applying INT8 LM Head patch..."
python3 /opt/patches/patch_int8_lmhead.py
echo "[v2] Starting vLLM..."
exec vllm "$@"
