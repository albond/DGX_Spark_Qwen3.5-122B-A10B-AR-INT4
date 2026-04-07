#!/bin/bash
set -e
echo "[v2-tq] Applying INT8 LM Head patch..."
python3 /opt/patches/patch_int8_lmhead.py
echo "[v2-tq] Applying TurboQuant patches..."
python3 /opt/patches/patch_turboquant.py
echo "[v2-tq] Starting vLLM..."
exec vllm "$@"
