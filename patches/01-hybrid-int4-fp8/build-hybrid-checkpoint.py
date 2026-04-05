#!/usr/bin/env python3
"""
Build a hybrid GPTQ-INT4 + FP8 checkpoint for Qwen3.5-122B-A10B.

Takes MoE expert weights from the GPTQ-INT4 checkpoint (0.5 bytes/param),
and dense layers (attention, shared experts, embeddings) from the official
FP8 checkpoint (1 byte/param + calibrated block scales).

Result: a checkpoint that is ~9 GB smaller than the GPTQ-INT4 original
while using properly calibrated FP8 scales (not naive cast), yielding
better decode throughput on bandwidth-limited hardware.

NOTE: Requires the hybrid FP8 dispatch patch from https://github.com/rmstxrx/vllm/tree/v0.17.1-hybrid-fp8

Usage:
    python build-hybrid-checkpoint.py \
        --gptq-dir  ~/inference/models/hf/qwen3.5-122b-a10b-gptq-int4 \
        --fp8-repo  Qwen/Qwen3.5-122B-A10B-FP8 \
        --output    ~/inference/models/hf/qwen3.5-122b-a10b-fp8hybrid
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file


logger = logging.getLogger(__name__)


def get_fp8_non_expert_manifest(fp8_repo: str) -> dict[str, str]:
    """Get the weight map for non-expert tensors from the FP8 checkpoint.

    Args:
        fp8_repo: Hugging Face repo ID for the FP8 checkpoint.

    Returns:
        Mapping of tensor names to shard filenames for non-expert tensors.
    """
    idx_path = hf_hub_download(fp8_repo, "model.safetensors.index.json")
    with open(idx_path, encoding="utf-8") as f:
        idx = json.load(f)

    wm = idx["weight_map"]
    return {k: v for k, v in wm.items() if ".experts." not in k}


def download_fp8_shards(fp8_repo: str, shards: set[str], cache_dir: Path) -> dict[str, Path]:
    """Download only the needed FP8 shards.

    Args:
        fp8_repo: Hugging Face repo ID for the FP8 checkpoint.
        shards: Shard filenames that contain non-expert tensors.
        cache_dir: Local cache directory for downloaded shards.

    Returns:
        Mapping of shard filename to downloaded local path.
    """
    shard_paths: dict[str, Path] = {}
    for shard in sorted(shards):
        logger.info("  Downloading %s...", shard)
        path = hf_hub_download(fp8_repo, shard, local_dir=str(cache_dir))
        shard_paths[shard] = Path(path)
        logger.info("    -> %s", path)
    return shard_paths


def extract_fp8_tensors(shard_paths: dict[str, Path], wanted: dict[str, str]) -> dict[str, torch.Tensor]:
    """Extract the requested FP8 tensors from downloaded shards.

    Args:
        shard_paths: Mapping of shard filename to local path.
        wanted: Mapping of tensor name to shard filename.

    Returns:
        Mapping of tensor name to loaded FP8 tensor.
    """
    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_path in shard_paths.items():
        keys_in_shard = [k for k, v in wanted.items() if v == shard_name]
        if not keys_in_shard:
            continue

        logger.info("  Extracting %d tensors from %s...", len(keys_in_shard), shard_name)
        with safe_open(str(shard_path), framework="pt") as f:
            for k in keys_in_shard:
                tensors[k] = f.get_tensor(k)

    return tensors


def find_model_safetensors_files(directory: Path) -> list[Path]:
    """Find model safetensors files in a GPTQ checkpoint directory.

    Args:
        directory: Directory containing GPTQ checkpoint files.

    Returns:
        Sorted list of shard paths, or a single-element list for single-file checkpoints.

    Raises:
        FileNotFoundError: If no supported model safetensors files are found.
    """
    gptq_shards = sorted(directory.glob("model.safetensors-*"))
    if gptq_shards:
        return gptq_shards

    # AutoRound / HF naming: model-NNNNN-of-NNNNN.safetensors
    gptq_shards = sorted(directory.glob("model-*-of-*.safetensors"))
    if gptq_shards:
        return gptq_shards

    single_file = directory / "model.safetensors"
    if single_file.is_file():
        return [single_file]

    raise FileNotFoundError(
        f"No model.safetensors files found in {directory}. Expected a sharded "
        "checkpoint (model.safetensors-NNNNN-of-NNNNN)."
    )


def validate_gptq_input(gptq_dir: Path) -> None:
    """Validate the GPTQ checkpoint directory before any downloads.

    Args:
        gptq_dir: Path to the local GPTQ checkpoint directory.

    Raises:
        FileNotFoundError: If the directory or required files do not exist.
        NotADirectoryError: If the path exists but is not a directory.
    """
    if not gptq_dir.exists():
        raise FileNotFoundError(f"GPTQ directory does not exist: {gptq_dir}")
    if not gptq_dir.is_dir():
        raise NotADirectoryError(f"GPTQ path is not a directory: {gptq_dir}")
    if not any(path.is_file() for path in gptq_dir.glob("*.safetensors*")):
        raise FileNotFoundError(f"No .safetensors files found in {gptq_dir}")
    if not (gptq_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in {gptq_dir}")


def validate_output_dir(output_dir: Path, force: bool) -> None:
    """Validate the output directory before building.

    Args:
        output_dir: Path where the hybrid checkpoint will be written.
        force: Whether destructive cleanup is allowed.

    Raises:
        FileExistsError: If the output directory is non-empty without `force`.
        NotADirectoryError: If the output path exists but is not a directory.
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory {output_dir} exists and is not empty. Use --force "
            "to remove existing model.safetensors* and config.json files before building."
        )


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    """Create or clean the output directory before writing files.

    Args:
        output_dir: Path where the hybrid checkpoint will be written.
        force: Whether destructive cleanup is allowed.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if not force:
        return

    delete_targets = sorted(
        {
            path
            for pattern in ("model.safetensors*", "config.json")
            for path in output_dir.glob(pattern)
            if path.is_file()
        }
    )
    for path in delete_targets:
        path.unlink()
        logger.info("  Deleted %s", path)


def build_hybrid_checkpoint(
    gptq_dir: Path,
    fp8_tensors: dict[str, torch.Tensor],
    output_dir: Path,
    force: bool,
) -> tuple[int, int, int]:
    """Build the hybrid checkpoint from GPTQ and FP8 tensors.

    Args:
        gptq_dir: Path to the local GPTQ checkpoint directory.
        fp8_tensors: FP8 tensors keyed by tensor name.
        output_dir: Output directory for rewritten checkpoint shards.
        force: Whether to continue if many unexpected FP8 tensors are unmatched.

    Returns:
        Tuple of replaced tensor count, added scale tensor count, and bytes saved.

    Raises:
        FileNotFoundError: If no supported GPTQ model files are found.
        RuntimeError: If too many FP8 tensors cannot be matched and `force` is not set.
        ValueError: If a matched FP8 tensor has an incompatible shape.
    """
    # 1. Copy non-safetensors files
    for f in gptq_dir.iterdir():
        if f.suffix != ".safetensors" and not f.name.startswith("."):
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)

    # 2. Process each GPTQ shard
    gptq_shards = find_model_safetensors_files(gptq_dir)
    total_replaced = 0
    total_added = 0
    total_saved_bytes = 0

    # Track which FP8 tensors have been placed
    placed_fp8: set[str] = set()

    for i, shard_path in enumerate(gptq_shards):
        shard_name = shard_path.name
        logger.info("  [%d/%d] %s...", i + 1, len(gptq_shards), shard_name)

        gptq_tensors = load_file(str(shard_path))
        output_tensors: dict[str, torch.Tensor] = {}
        replaced = 0

        for name, tensor in gptq_tensors.items():
            if name in fp8_tensors:
                # Replace BF16 tensor with FP8 version
                fp8_tensor = fp8_tensors[name]
                if tensor.shape != fp8_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: GPTQ={tensor.shape}, FP8={fp8_tensor.shape}"
                    )
                old_bytes = tensor.numel() * tensor.element_size()
                new_bytes = fp8_tensor.numel() * fp8_tensor.element_size()
                total_saved_bytes += old_bytes - new_bytes
                output_tensors[name] = fp8_tensor
                placed_fp8.add(name)
                replaced += 1

                # Also add the scale tensor if it exists
                scale_name = name.replace(".weight", ".weight_scale_inv")
                if scale_name in fp8_tensors and scale_name != name:
                    output_tensors[scale_name] = fp8_tensors[scale_name]
                    placed_fp8.add(scale_name)
                    total_added += 1
            else:
                output_tensors[name] = tensor

        total_replaced += replaced

        save_file(output_tensors, str(output_dir / shard_name))
        logger.info("    replaced=%d, tensors=%d", replaced, len(output_tensors))

    # 3. Check for any FP8 tensors not yet placed (e.g. scale tensors
    #    for weights that exist in shards we already processed)
    unplaced = set(fp8_tensors.keys()) - placed_fp8
    if unplaced:
        expected_unplaced = {
            name
            for name in unplaced
            if name.endswith(".weight_scale_inv")
            and f"{name.removesuffix('.weight_scale_inv')}.weight" in placed_fp8
        }
        unexpected_unplaced = sorted(unplaced - expected_unplaced)

        for name in unexpected_unplaced:
            logger.warning("WARNING: unexpected unmatched FP8 tensor %s", name)

        if len(unexpected_unplaced) > 5 and not force:
            message = (
                f"{len(unexpected_unplaced)} FP8 tensors could not be matched to GPTQ tensors. "
                "This likely indicates a naming mismatch between the GPTQ and FP8 checkpoints. "
                "Use --force to proceed anyway."
            )
            logger.error("ERROR: %s", message)
            raise RuntimeError(message)
        if len(unexpected_unplaced) > 5 and force:
            logger.warning(
                "WARNING: proceeding despite %d unexpected unmatched FP8 tensors because --force was provided",
                len(unexpected_unplaced),
            )

        if expected_unplaced:
            logger.info("  Adding %d expected unplaced FP8 scale tensors to final shard...", len(expected_unplaced))

        if expected_unplaced:
            # Load last shard, add expected scale tensors, re-save
            last_shard = output_dir / gptq_shards[-1].name
            existing = load_file(str(last_shard))
            for name in sorted(expected_unplaced):
                existing[name] = fp8_tensors[name]
                total_added += 1
            save_file(existing, str(last_shard))

    return total_replaced, total_added, total_saved_bytes


def update_safetensors_index(output_dir: Path) -> None:
    """Rebuild `model.safetensors.index.json` from actual shard contents.

    Args:
        output_dir: Directory containing rewritten model shards.
    """
    weight_map: dict[str, str] = {}
    total_size = 0

    for shard_path in find_model_safetensors_files(output_dir):
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard_path.name
                tensor = f.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }

    with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    logger.info("  Index rebuilt: %d tensors, %.2f GB", len(weight_map), total_size / 1e9)


def update_config(output_dir: Path) -> None:
    """Update `config.json` with hybrid quantization metadata.

    Args:
        output_dir: Directory containing the hybrid checkpoint.
    """
    config_path = output_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    config["_hybrid_quant_info"] = {
        "description": "Hybrid GPTQ-INT4 + FP8 checkpoint for single-GPU deployment",
        "moe_experts": "GPTQ INT4 (group_size=128, sym=True, Marlin kernels)",
        "dense_layers": "FP8 E4M3 block-128 (from official Qwen/Qwen3.5-122B-A10B-FP8, calibrated scales)",
        "norms_gates_embeddings": "Preserved at source dtype (BF16 for norms/gates, FP8 for others)",
        "source_gptq": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        "source_fp8": "Qwen/Qwen3.5-122B-A10B-FP8",
        "vllm_patch": "https://github.com/rmstxrx/vllm/tree/v0.17.1-hybrid-fp8",
        "target_hardware": "NVIDIA DGX Spark (GB10, 128GB unified, 273 GB/s)",
        "converter": "build-hybrid-checkpoint.py"
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> int:
    """Build a hybrid GPTQ-INT4 + FP8 checkpoint.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Build hybrid GPTQ-INT4 + FP8 checkpoint")
    parser.add_argument("--gptq-dir", required=True, help="Path to GPTQ-INT4 model")
    parser.add_argument("--fp8-repo", default="Qwen/Qwen3.5-122B-A10B-FP8", help="HF repo for FP8 model")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow output cleanup and continue despite many unmatched FP8 tensors",
    )
    args = parser.parse_args()

    gptq_dir = Path(args.gptq_dir)
    output_dir = Path(args.output)

    validate_gptq_input(gptq_dir)
    find_model_safetensors_files(gptq_dir)
    validate_output_dir(output_dir, args.force)

    logger.info("=== Hybrid GPTQ-INT4 + FP8 Checkpoint Builder ===")
    logger.info("  GPTQ source: %s", gptq_dir)
    logger.info("  FP8 source:  %s", args.fp8_repo)
    logger.info("  Output:      %s", output_dir)
    logger.info("")

    # Step 1: Get manifest of non-expert tensors from FP8 checkpoint
    logger.info("[1/5] Fetching FP8 tensor manifest...")
    fp8_manifest = get_fp8_non_expert_manifest(args.fp8_repo)
    shards_needed = set(fp8_manifest.values())
    logger.info("  Non-expert tensors: %d", len(fp8_manifest))
    logger.info("  Shards to download: %s", sorted(shards_needed))

    if args.dry_run:
        logger.info("")
        logger.info("[DRY RUN] Would download shards and build hybrid. Exiting.")
        return 0

    prepare_output_dir(output_dir, args.force)
    cache_dir = output_dir / ".fp8_cache"
    cache_dir.mkdir(exist_ok=True)

    # Step 2: Download needed FP8 shards
    logger.info("")
    logger.info("[2/5] Downloading %d FP8 shards...", len(shards_needed))
    shard_paths = download_fp8_shards(args.fp8_repo, shards_needed, cache_dir)

    # Step 3: Extract non-expert FP8 tensors
    logger.info("")
    logger.info("[3/5] Extracting FP8 tensors...")
    fp8_tensors = extract_fp8_tensors(shard_paths, fp8_manifest)
    logger.info("  Extracted: %d tensors", len(fp8_tensors))

    # Show dtype breakdown
    dtypes: dict[str, int] = {}
    for name, t in fp8_tensors.items():
        d = str(t.dtype)
        dtypes[d] = dtypes.get(d, 0) + 1
    logger.info("  Dtypes: %s", dtypes)

    # Step 4: Build hybrid checkpoint
    logger.info("")
    logger.info("[4/5] Building hybrid checkpoint...")
    replaced, added, saved = build_hybrid_checkpoint(
        gptq_dir,
        fp8_tensors,
        output_dir,
        args.force,
    )

    # Step 5: Update index and config
    logger.info("")
    logger.info("[5/5] Updating index and config...")
    update_safetensors_index(output_dir)
    update_config(output_dir)

    # Cleanup downloaded FP8 shards
    shutil.rmtree(cache_dir, ignore_errors=True)

    logger.info("")
    logger.info("=== Complete ===")
    logger.info("  Tensors replaced (BF16→FP8):  %d", replaced)
    logger.info("  Scale tensors added:           %d", added)
    logger.info("  Bytes saved:                   %.2f GB", saved / 1e9)
    logger.info("  Output: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
