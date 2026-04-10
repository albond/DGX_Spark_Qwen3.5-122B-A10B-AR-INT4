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

VALID_RECIPES = (
    "turboquant35",
    "turboquant25",
    "turboquant_asym",
    "turboquant_q8k_tq35v",
    "turboquant_q8k_tq25v",
)

# Outlier ratios per recipe (fraction of head dims treated as high-precision).
OUTLIER_RATIOS = {
    "turboquant35": 0.50,
    "turboquant25": 0.25,
    "turboquant_asym": 0.50,   # Same storage layout as TQ35
    # Q8K: K is int8 (no outlier selection needed); V uses the named TQ recipe.
    # Outlier ratio here drives the V-side index generation only.
    "turboquant_q8k_tq35v": 0.50,  # V uses TQ35 (50% outlier dims)
    "turboquant_q8k_tq25v": 0.25,  # V uses TQ25 (25% outlier dims)
}
# For asym recipes the *storage* recipe drives dim calculations.
ASYM_BASE = {
    "turboquant_asym": "turboquant35",
    # Q8K: resolves to the V-side storage recipe.
    "turboquant_q8k_tq35v": "turboquant35",
    "turboquant_q8k_tq25v": "turboquant25",
}

# Q8K recipes: K metadata is unused at runtime (K is int8, not TQ-indexed).
# It is included for schema completeness and set equal to V indices.
Q8K_RECIPES = {"turboquant_q8k_tq35v", "turboquant_q8k_tq25v"}


def _outlier_count(head_size: int, recipe: str) -> int:
    storage_recipe = ASYM_BASE.get(recipe, recipe)
    ratio = OUTLIER_RATIOS[storage_recipe]
    group_alignment = 16
    aligned = int(round(head_size * ratio / group_alignment) * group_alignment)
    if aligned <= 0 or aligned >= head_size:
        raise ValueError(
            f"Cannot compute valid outlier count for head_size={head_size}, recipe={recipe}"
        )
    return aligned


def main():
    parser = argparse.ArgumentParser(
        description="Generate TurboQuant metadata (turboquant_kv.json) for Qwen3.5-122B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recipes
-------
turboquant35         Symmetric TQ35 — same first-half dims for both K and V (default).
turboquant25         Symmetric TQ25 — same first-quarter dims for both K and V.
turboquant_asym      Asymmetric TQ  — K uses first-half dims, V uses last-half dims.
                     Both K and V are stored with the TQ35 format, but V's outlier
                     dimensions are shifted to maximise decorrelation from K's set.
                     This improves reconstruction quality (especially needle-in-haystack)
                     without changing memory usage or requiring kernel modifications.
turboquant_q8k_tq35v True asymmetric — K stored as int8 (Q8 + fp16 scale), V as TQ35.
                     Memory ~2× over fp16 baseline (vs 4× for symmetric TQ35).
                     K metadata is included for schema completeness but unused at
                     runtime (K is stored as int8, not TQ-indexed).
turboquant_q8k_tq25v Like turboquant_q8k_tq35v but with TQ25 for V (more compression).
""",
    )
    parser.add_argument("--model-dir", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--recipe",
        default="turboquant35",
        choices=VALID_RECIPES,
        help="TurboQuant recipe (default: turboquant35)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    recipe = args.recipe
    is_asym = recipe in ASYM_BASE
    is_q8k = recipe in Q8K_RECIPES
    outlier_count = _outlier_count(HEAD_SIZE, recipe)

    if is_q8k:
        # Q8K: V uses the standard TQ outlier dims (first-half or first-quarter).
        # K indices are set equal to V indices for schema completeness;
        # they are NOT used at runtime (K is stored as int8, not TQ-indexed).
        v_start_q8k = 0  # V uses first-half dims (same as symmetric TQ default)
        value_indices = [
            list(range(v_start_q8k, v_start_q8k + outlier_count))
            for _ in range(NUM_KV_HEADS)
        ]
        key_indices = value_indices  # K indices are unused; mirror V for schema
    elif is_asym:
        # Soft asymmetric: K first-half, V last-half (disjoint).
        # K: first <outlier_count> dims
        key_indices = [list(range(outlier_count)) for _ in range(NUM_KV_HEADS)]
        # V: last <outlier_count> dims — disjoint from K to maximise coverage.
        # With head_size=256 and outlier_count=128 this gives K→[0..127], V→[128..255].
        v_start = HEAD_SIZE - outlier_count
        value_indices = [list(range(v_start, HEAD_SIZE)) for _ in range(NUM_KV_HEADS)]
    else:
        # Symmetric: same first-half dims for both K and V.
        key_indices = [list(range(outlier_count)) for _ in range(NUM_KV_HEADS)]
        value_indices = key_indices

    layers = {}
    for layer_name in QWEN35_122B_ATTENTION_LAYERS:
        layers[layer_name] = {
            "key_high_precision_indices": key_indices,
            "value_high_precision_indices": value_indices,
        }

    metadata = {
        "version": 1,
        "recipe": recipe,
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
    print(f"  Recipe:          {recipe}")
    print(f"  Head size:       {HEAD_SIZE}")
    print(f"  Layers:          {len(layers)}")
    if is_q8k:
        print(f"  V outlier dims:  {value_indices[0][0]}..{value_indices[0][-1]}  ({outlier_count}/{HEAD_SIZE})")
        print(f"  K outlier dims:  same as V (unused — K stored as int8)")
        print()
        print("  Q8K mode: K is stored as int8 + fp16 scale (~2× vs fp16 baseline).")
        print("  V is stored with TQ format. K metadata is for schema only.")
    elif is_asym:
        print(f"  K outlier dims:  {key_indices[0][0]}..{key_indices[0][-1]}  ({outlier_count}/{HEAD_SIZE})")
        print(f"  V outlier dims:  {value_indices[0][0]}..{value_indices[0][-1]}  ({outlier_count}/{HEAD_SIZE})  ← shifted")
        print()
        print("  Asymmetric mode: K and V use disjoint outlier-dimension sets.")
        print("  Both are stored with TQ35 format — no extra memory or kernel changes.")
    else:
        print(f"  K outlier dims:  {key_indices[0][0]}..{key_indices[0][-1]}  ({outlier_count}/{HEAD_SIZE})")
        print(f"  V outlier dims:  same as K (symmetric)")


if __name__ == "__main__":
    main()
