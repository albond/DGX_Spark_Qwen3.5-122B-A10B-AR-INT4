#!/usr/bin/env python3
"""Add MTP (Multi-Token Prediction) weights to a hybrid INT4+FP8 checkpoint.

Intel AutoRound INT4 includes MTP weights in model_extra_tensors.safetensors
but does not reference them in the model index. This script copies the file
and updates model.safetensors.index.json so vLLM can load MTP heads for
speculative decoding.

Usage:
    python add-mtp-weights.py \
        --source ~/.cache/huggingface/hub/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/snapshots/<hash> \
        --target /path/to/hybrid-checkpoint
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Add MTP weights to hybrid checkpoint"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to Intel AutoRound checkpoint (contains model_extra_tensors.safetensors)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to hybrid checkpoint to update",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)

    # Verify source has MTP weights
    mtp_file = source / "model_extra_tensors.safetensors"
    if not mtp_file.exists():
        raise FileNotFoundError(f"MTP weights not found: {mtp_file}")

    # Read source index to find MTP tensor names
    source_index = source / "model.safetensors.index.json"
    with open(source_index) as f:
        src_idx = json.load(f)

    mtp_keys = {
        k: v for k, v in src_idx["weight_map"].items() if "mtp" in k.lower()
    }
    print(f"Found {len(mtp_keys)} MTP tensors in source index")

    if not mtp_keys:
        raise ValueError("No MTP tensors found in source checkpoint")

    # Copy MTP weights file
    target_mtp = target / "model_extra_tensors.safetensors"
    print(f"Copying {mtp_file} -> {target_mtp}")
    shutil.copy2(mtp_file, target_mtp)
    print(f"  Size: {target_mtp.stat().st_size / 1e9:.1f} GB")

    # Update target index
    target_index = target / "model.safetensors.index.json"
    with open(target_index) as f:
        tgt_idx = json.load(f)

    existing = len(tgt_idx["weight_map"])
    for key in mtp_keys:
        tgt_idx["weight_map"][key] = "model_extra_tensors.safetensors"

    with open(target_index, "w") as f:
        json.dump(tgt_idx, f, indent=2)

    added = len(tgt_idx["weight_map"]) - existing
    print(f"Added {added} MTP tensor mappings to index")
    print(f"Total tensors: {len(tgt_idx['weight_map'])}")
    print("Done. MTP speculative decoding is now available.")


if __name__ == "__main__":
    main()
