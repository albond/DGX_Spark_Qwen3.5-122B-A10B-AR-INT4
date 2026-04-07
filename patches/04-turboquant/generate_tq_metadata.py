#!/usr/bin/env python3
"""Generate default TurboQuant metadata (turboquant_kv.json) for Qwen3.5-122B.

Places the file in the model directory so vLLM finds it automatically.

Usage:
    python patches/04-turboquant/generate_tq_metadata.py \
        --model-dir ~/models/qwen35-122b-hybrid-int4fp8
"""
import argparse
import json
from pathlib import Path


# Qwen3.5-122B-A10B: 48 layers, but only every 4th uses standard attention
# (layers 11,15,19,23,27,31,35,39,43,47 + some others).
# The rest use DeltaNet linear attention (no KV cache).
# Plus 3 MTP layers.
QWEN35_122B_ATTENTION_LAYERS = [
    "model.layers.11.self_attn.attn",
    "model.layers.15.self_attn.attn",
    "model.layers.19.self_attn.attn",
    "model.layers.23.self_attn.attn",
    "model.layers.27.self_attn.attn",
    "model.layers.31.self_attn.attn",
    "model.layers.35.self_attn.attn",
    "model.layers.39.self_attn.attn",
    "model.layers.43.self_attn.attn",
    "model.layers.47.self_attn.attn",
    # Additional attention layers
    "model.layers.3.self_attn.attn",
    "model.layers.7.self_attn.attn",
    # MTP layers
    "mtp.layers.0.self_attn.attn",
    "mtp.layers.1.self_attn.attn",
    "mtp.layers.2.self_attn.attn",
]

HEAD_SIZE = 256
NUM_KV_HEADS = 2
RECIPE = "turboquant35"


def main():
    parser = argparse.ArgumentParser(description="Generate TurboQuant metadata")
    parser.add_argument("--model-dir", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Build default metadata: first half of head dims are outliers
    outlier_count = HEAD_SIZE // 2  # turboquant35 = 50% outlier ratio
    default_indices = [list(range(outlier_count)) for _ in range(NUM_KV_HEADS)]

    layers = {}
    for layer_name in QWEN35_122B_ATTENTION_LAYERS:
        layers[layer_name] = {
            "key_high_precision_indices": default_indices,
            "value_high_precision_indices": default_indices,
        }

    metadata = {
        "version": 1,
        "recipe": RECIPE,
        "head_size": HEAD_SIZE,
        "model_name": "Qwen3.5-122B-A10B",
        "transform_version": "structured_hadamard_v1",
        "codebook_version": "lloyd_beta_v1",
        "layers": layers,
    }

    output_path = model_dir / "turboquant_kv.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {output_path}")
    print(f"  Recipe: {RECIPE}")
    print(f"  Head size: {HEAD_SIZE}")
    print(f"  Layers: {len(layers)}")
    print(f"  Outlier dims per head: {outlier_count}/{HEAD_SIZE}")


if __name__ == "__main__":
    main()
