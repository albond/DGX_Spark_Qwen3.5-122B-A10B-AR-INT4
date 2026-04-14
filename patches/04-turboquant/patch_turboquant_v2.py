"""Apply TurboQuant patches to base vLLM installation.

Based on mitkox/vllm-turboquant fork. Copies new TQ files, overwrites
heavily-modified files, and applies small edits to register TQ in vLLM.

TurboQuant patches are inert unless --kv-cache-dtype turboquantXX is used.
"""
import shutil
import os
import sys

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
SRC = "/opt/patches/tq_src"

_errors = []


def read(path):
    with open(path) as f:
        return f.read()


def write(path, content):
    with open(path, "w") as f:
        f.write(content)


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  + {os.path.basename(dst)}")


def patch(path, old, new, desc=""):
    content = read(path)
    if old not in content:
        msg = f"Pattern not found in {os.path.basename(path)}: {old[:60]!r}"
        print(f"  FAIL: {msg}", file=sys.stderr)
        _errors.append(msg)
        return
    content = content.replace(old, new, 1)
    write(path, content)
    print(f"  ~ {os.path.basename(path)} ({desc})")


# ── Step 1: Copy 4 new TurboQuant source files ──
print("=== Step 1: Copy new TurboQuant files ===")
new_files = [
    ("ops/turboquant_kv_cache.py", f"{VLLM}/v1/attention/ops/turboquant_kv_cache.py"),
    ("ops/turboquant_metadata.py", f"{VLLM}/v1/attention/ops/turboquant_metadata.py"),
    ("ops/triton_turboquant_decode.py", f"{VLLM}/v1/attention/ops/triton_turboquant_decode.py"),
    ("ops/triton_turboquant_kv_update.py", f"{VLLM}/v1/attention/ops/triton_turboquant_kv_update.py"),
]
for src_rel, dst in new_files:
    copy(f"{SRC}/{src_rel}", dst)

# ── Step 2: Overwrite heavily-modified files ──
# These are TQ-integrated versions of existing files. Overwriting is safer
# than patching because the changes are extensive (triton_attn.py: 645→1529 lines).
print("\n=== Step 2: Overwrite TQ-integrated files ===")
overwrites = [
    ("selector.py", f"{VLLM}/v1/attention/selector.py"),
    ("backends/triton_attn.py", f"{VLLM}/v1/attention/backends/triton_attn.py"),
    ("kv_cache_interface.py", f"{VLLM}/v1/kv_cache_interface.py"),
]
for src_rel, dst in overwrites:
    copy(f"{SRC}/{src_rel}", dst)

# ── Step 2b: Hybrid model compatibility — aligned packed_dim ──
# TQ packed_dim (e.g. 120 for 3.5-bit, head_size=256) may not be compatible
# with Mamba/DeltaNet page sizes in hybrid models like Qwen3.5.
# vLLM's page allocator requires all layer page sizes to be mutually divisible.
# Solution: pad packed_dim to next power-of-2 (120→128). This guarantees
# compatibility with ANY page size while adding only 6.25% overhead.
# Triton kernels use TurboQuantLayout.packed_dim (unaligned) internally,
# so padding bytes are never read/written — zero performance cost.
print("\n=== Step 2b: Hybrid model alignment ===")
tq_kv_path = f"{VLLM}/v1/attention/ops/turboquant_kv_cache.py"
content = read(tq_kv_path)
if "_next_pow2" not in content:
    content = content.replace(
        "def get_turboquant_packed_dim(\n"
        "    head_size: int,\n"
        "    bits_or_dtype: float | int | str,\n"
        ") -> int:\n"
        "    kv_cache_dtype = _canonical_turboquant_dtype(bits_or_dtype)\n"
        "    return get_turboquant_layout(kv_cache_dtype, head_size).packed_dim",
        "def _next_pow2(n: int) -> int:\n"
        "    \"\"\"Round up to next power of 2 for hybrid page-size compatibility.\"\"\"\n"
        "    p = 1\n"
        "    while p < n:\n"
        "        p <<= 1\n"
        "    return p\n"
        "\n"
        "\n"
        "def get_turboquant_packed_dim(\n"
        "    head_size: int,\n"
        "    bits_or_dtype: float | int | str,\n"
        ") -> int:\n"
        "    kv_cache_dtype = _canonical_turboquant_dtype(bits_or_dtype)\n"
        "    raw_dim = get_turboquant_layout(kv_cache_dtype, head_size).packed_dim\n"
        "    # Align to power-of-2 for hybrid model (attention+Mamba) page unification.\n"
        "    # Triton kernels use raw_dim via TurboQuantLayout; padding bytes are inert.\n"
        "    return _next_pow2(raw_dim)",
        1,
    )
    write(tq_kv_path, content)
    print("  ~ turboquant_kv_cache.py (aligned packed_dim for hybrid models)")
else:
    print("  SKIP: turboquant_kv_cache.py already aligned")

# ── Step 2c: Force fallback prefill for SM121 (GB10) ──
# SM121 has 99 KB shared memory, Triton prefill kernel needs 160 KB for head_size=256.
# Set max head size to 0 to always use the fallback attention path for prefill.
# Decode (token generation) is unaffected — uses TQ decode kernel at full speed.
print("\n=== Step 2c: SM121 prefill fallback ===")
triton_attn_path = f"{VLLM}/v1/attention/backends/triton_attn.py"
content = read(triton_attn_path)
if "TURBOQUANT_TRITON_PREFILL_MAX_HEAD_SIZE = 256" in content:
    content = content.replace(
        "TURBOQUANT_TRITON_PREFILL_MAX_HEAD_SIZE = 256",
        "# SM121 (GB10): 99 KB shared memory < 160 KB needed for head_size=256.\n"
        "# Use fallback attention for prefill; decode is unaffected.\n"
        "TURBOQUANT_TRITON_PREFILL_MAX_HEAD_SIZE = 0",
    )
    write(triton_attn_path, content)
    print("  ~ triton_attn.py (SM121: prefill fallback for all head sizes)")
else:
    print("  SKIP: triton_attn.py already patched")

# ── Step 2d: CUDA graph compatibility (WIP) ──
# Lab verified: encode kernel is graph-safe, .item() and repeat_interleave fixed.
# FULL graph capture works, but runtime fails with device-side assert on SM121.
# Root cause: Triton JIT recompiles at inference with shapes not seen during capture.
# TODO: needs lab to find which kernel shape triggers SM121 assert.
# For now: keep NEVER (PIECEWISE mode) — proven stable at 40 tok/s.
print("\n=== Step 2d: CUDA graphs (PIECEWISE — FULL WIP) ===")
content = read(triton_attn_path)
_2d_changed = False

# Fix 1: Guard .item() GPU→CPU sync — skip during graph capture
if '(slot_mapping >= 0).any().item()' in content:
    content = content.replace(
        '            if not (slot_mapping >= 0).any().item():\n'
        '                return',
        '            if not torch.cuda.is_current_stream_capturing():\n'
        '                if not (slot_mapping >= 0).any().item():\n'
        '                    return',
    )
    _2d_changed = True

# Fix 2: Keep NEVER for now — FULL graphs capture OK but runtime asserts on SM121.
# TODO: lab_autotune to find which Triton kernel shape fails at runtime.
print("  SKIP: keeping AttentionCGSupport.NEVER (FULL graphs runtime assert on SM121)")

# Fix 3: Guard repeat_interleave in _build_turboquant_token_metadata
# repeat_interleave with data-dependent output size is graph-incompatible.
# During decode (graph capture), query_lens are all 1s, so repeat_interleave
# is identity. Replace with simple arange during capture.
if 'torch.repeat_interleave(' in content and '_build_turboquant_token_metadata' in content:
    content = content.replace(
        '        if attn_metadata.turboquant_seq_ids is None or kv_lens is not None:\n'
        '            query_lens = (\n'
        '                attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]\n'
        '            )\n'
        '            seq_ids = torch.repeat_interleave(\n'
        '                torch.arange(\n'
        '                    attn_metadata.seq_lens.shape[0],\n'
        '                    device=attn_metadata.seq_lens.device,\n'
        '                    dtype=torch.int32,\n'
        '                ),\n'
        '                query_lens,\n'
        '            )',
        '        if attn_metadata.turboquant_seq_ids is None or kv_lens is not None:\n'
        '            query_lens = (\n'
        '                attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]\n'
        '            )\n'
        '            # CUDA graph fix: repeat_interleave has data-dependent output\n'
        '            # size (graph-incompatible). During decode, query_lens are all 1s\n'
        '            # so repeat_interleave = identity = simple arange.\n'
        '            if torch.cuda.is_current_stream_capturing():\n'
        '                seq_ids = torch.arange(\n'
        '                    attn_metadata.num_actual_tokens,\n'
        '                    device=attn_metadata.seq_lens.device,\n'
        '                    dtype=torch.int32,\n'
        '                )\n'
        '            else:\n'
        '                seq_ids = torch.repeat_interleave(\n'
        '                    torch.arange(\n'
        '                        attn_metadata.seq_lens.shape[0],\n'
        '                        device=attn_metadata.seq_lens.device,\n'
        '                        dtype=torch.int32,\n'
        '                    ),\n'
        '                    query_lens,\n'
        '                )',
    )
    _2d_changed = True

if _2d_changed:
    write(triton_attn_path, content)
    print("  ~ triton_attn.py (.item() guard + CUDA graphs + repeat_interleave fix)")
else:
    print("  SKIP: triton_attn.py already fixed")

# ── Step 2e: Relax packed_dim validation in encode kernel ──
# The encode kernel validates cache.shape[3] == layout.packed_dim, but with
# our alignment the cache slot is larger (128) than the logical packed_dim (120).
# The extra bytes are inert padding. Relax to >= check.
print("\n=== Step 2e: Relax encode kernel validation ===")
kv_update_path = f"{VLLM}/v1/attention/ops/triton_turboquant_kv_update.py"
content = read(kv_update_path)
if 'cache.shape[3] != layout.packed_dim' in content:
    content = content.replace(
        '    if cache.shape[3] != layout.packed_dim:\n'
        '        raise ValueError("TurboQuant cache packed_dim does not match the layout.")',
        '    if cache.shape[3] < layout.packed_dim:\n'
        '        raise ValueError(\n'
        '            f"TurboQuant cache packed_dim {cache.shape[3]} is smaller than "\n'
        '            f"layout packed_dim {layout.packed_dim}."\n'
        '        )',
    )
    write(kv_update_path, content)
    print("  ~ triton_turboquant_kv_update.py (allow aligned packed_dim >= layout)")
else:
    print("  SKIP: kv_update.py already patched")

# ── Step 3: Patch config/cache.py ──
print("\n=== Step 3: Patch config/cache.py ===")
cache_path = f"{VLLM}/config/cache.py"
content = read(cache_path)

if "turboquant" not in content:
    # Add turboquant dtypes to CacheDType Literal
    if '    "fp8_ds_mla",\n]' in content:
        content = content.replace(
            '    "fp8_ds_mla",\n]',
            '    "fp8_ds_mla",\n'
            '    "turboquant25",\n    "turboquant35",\n    "turboquant_asym",\n'
            '    "turboquant_q8k_tq35v",\n    "turboquant_q8k_tq25v",\n]',
        )
    elif '    "fp8_per_token_head",\n]' in content:
        content = content.replace(
            '    "fp8_per_token_head",\n]',
            '    "fp8_per_token_head",\n'
            '    "turboquant25",\n    "turboquant35",\n    "turboquant_asym",\n'
            '    "turboquant_q8k_tq35v",\n    "turboquant_q8k_tq25v",\n]',
        )
    else:
        _errors.append("Could not find CacheDType closing pattern in cache.py")

    # Add enable_turboquant field
    if "cpu_kvcache_space_bytes" in content:
        content = content.replace(
            "    cpu_kvcache_space_bytes",
            '    enable_turboquant: bool = False\n'
            '    """Enable TurboQuant KV cache (requires NVIDIA GB10 / SM121)."""\n'
            '    turboquant_metadata_path: str | None = None\n'
            '    """Path to TurboQuant per-layer metadata JSON."""\n'
            "    cpu_kvcache_space_bytes",
        )

    # Add turboquant validation in _validate_cache_dtype
    if 'return cache_dtype\n\n    @model_validator' in content:
        content = content.replace(
            'return cache_dtype\n\n    @model_validator',
            '        elif cache_dtype.startswith("turboquant"):\n'
            '            _tq_msg = (\n'
            '                "Using TurboQuant KV cache with the Triton attention backend "\n'
            '                "(requires NVIDIA GB10 / SM121). "\n'
            '                "Asymmetric mode (turboquant_asym): K and V use disjoint "\n'
            '                "outlier-dim selections for improved quality."\n'
            '                if cache_dtype == "turboquant_asym"\n'
            '                else "Using TurboQuant KV cache (K=Q8 int8, V=TQ35) with the Triton "\n'
            '                "attention backend (requires NVIDIA GB10 / SM121)."\n'
            '                if cache_dtype == "turboquant_q8k_tq35v"\n'
            '                else "Using TurboQuant KV cache (K=Q8 int8, V=TQ25) with the Triton "\n'
            '                "attention backend (requires NVIDIA GB10 / SM121)."\n'
            '                if cache_dtype == "turboquant_q8k_tq25v"\n'
            '                else "Using TurboQuant KV cache with the Triton attention backend "\n'
            '                "(requires NVIDIA GB10 / SM121)."\n'
            '            )\n'
            '            logger.warning(_tq_msg)\n'
            '        return cache_dtype\n\n'
            '    @model_validator(mode="after")\n'
            '    def _validate_turboquant(self) -> "CacheConfig":\n'
            '        if not self.cache_dtype.startswith("turboquant"):\n'
            '            return self\n'
            '        if not self.enable_turboquant:\n'
            '            raise ValueError(\n'
            '                "TurboQuant KV cache requires --enable-turboquant flag."\n'
            '            )\n'
            '        return self\n\n    @model_validator',
            1,
        )

    write(cache_path, content)
    print("  ~ cache.py (add turboquant25/35 + enable_turboquant)")
else:
    print("  SKIP: cache.py already patched")

# ── Step 4: Patch engine/arg_utils.py ──
print("\n=== Step 4: Patch engine/arg_utils.py ===")
arg_path = f"{VLLM}/engine/arg_utils.py"
content = read(arg_path)

if "enable_turboquant" not in content:
    # Add turboquant fields
    content = content.replace(
        "    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype\n",
        "    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype\n"
        "    enable_turboquant: bool = CacheConfig.enable_turboquant\n"
        "    turboquant_metadata_path: str | None = CacheConfig.turboquant_metadata_path\n",
    )

    # Add CLI arguments
    old_cli = (
        '        cache_group.add_argument(\n'
        '            "--calculate-kv-scales", **cache_kwargs["calculate_kv_scales"]\n'
        '        )\n'
        '        cache_group.add_argument('
    )
    new_cli = (
        '        cache_group.add_argument(\n'
        '            "--calculate-kv-scales", **cache_kwargs["calculate_kv_scales"]\n'
        '        )\n'
        '        cache_group.add_argument(\n'
        '            "--enable-turboquant", **cache_kwargs["enable_turboquant"]\n'
        '        )\n'
        '        cache_group.add_argument(\n'
        '            "--turboquant-metadata-path",\n'
        '            **cache_kwargs["turboquant_metadata_path"],\n'
        '        )\n'
        '        cache_group.add_argument('
    )
    content = content.replace(old_cli, new_cli, 1)

    # Add to create_cache_config kwargs
    content = content.replace(
        "            calculate_kv_scales=self.calculate_kv_scales,\n",
        "            calculate_kv_scales=self.calculate_kv_scales,\n"
        "            enable_turboquant=self.enable_turboquant,\n"
        "            turboquant_metadata_path=self.turboquant_metadata_path,\n",
    )

    write(arg_path, content)
    print("  ~ arg_utils.py (add TQ CLI flags)")
else:
    print("  SKIP: arg_utils.py already patched")

# ── Step 5: Patch utils/torch_utils.py ──
print("\n=== Step 5: Patch utils/torch_utils.py ===")
torch_path = f"{VLLM}/utils/torch_utils.py"
content = read(torch_path)

if "turboquant" not in content:
    content = content.replace(
        "}\n\nTORCH_DTYPE_TO_NUMPY_DTYPE",
        '    "turboquant25": torch.uint8,\n'
        '    "turboquant35": torch.uint8,\n'
        '    "turboquant_asym": torch.uint8,\n'
        '    "turboquant_q8k_tq35v": torch.uint8,\n'
        '    "turboquant_q8k_tq25v": torch.uint8,\n'
        "}\n\nTORCH_DTYPE_TO_NUMPY_DTYPE",
    )
    write(torch_path, content)
    print("  ~ torch_utils.py (add turboquant dtype mappings)")
else:
    print("  SKIP: torch_utils.py already patched")

# ── Step 6: Patch attention.py ──
print("\n=== Step 6: Patch attention.py ===")
attn_path = f"{VLLM}/model_executor/layers/attention/attention.py"
content = read(attn_path)

if "turboquant_layer_name" not in content:
    # Add import
    content = content.replace(
        "from vllm.platforms import current_platform",
        "from vllm.platforms import current_platform\n"
        "from vllm.v1.attention.ops.turboquant_kv_cache import is_turboquant_kv_cache",
    )

    # Add TQ metadata args to extra_impl_args
    content = content.replace(
        "        self.layer_name = prefix",
        '        self.layer_name = prefix\n'
        "\n"
        "        # TurboQuant: pass layer metadata args to attention backend\n"
        "        if is_turboquant_kv_cache(kv_cache_dtype):\n"
        '            extra_impl_args["turboquant_layer_name"] = prefix\n'
        '            extra_impl_args["turboquant_model_name"] = (\n'
        "                None if vllm_config.model_config is None\n"
        "                else vllm_config.model_config.model\n"
        "            )\n"
        '            extra_impl_args["turboquant_metadata_path"] = (\n'
        "                None if cache_config is None\n"
        '                else getattr(cache_config, "turboquant_metadata_path", None)\n'
        "            )",
        1,
    )

    # Patch get_kv_cache_spec to pass cache_dtype_str to FullAttentionSpec
    # Without this, kv_cache_interface can't compute TQ packed dimensions
    content = content.replace(
        "            return FullAttentionSpec(\n"
        "                block_size=block_size,\n"
        "                num_kv_heads=self.num_kv_heads,\n"
        "                head_size=self.head_size,\n"
        "                head_size_v=self.head_size_v,\n"
        "                dtype=self.kv_cache_torch_dtype,\n"
        "            )",
        "            return FullAttentionSpec(\n"
        "                block_size=block_size,\n"
        "                num_kv_heads=self.num_kv_heads,\n"
        "                head_size=self.head_size,\n"
        "                head_size_v=self.head_size_v,\n"
        "                dtype=self.kv_cache_torch_dtype,\n"
        "                cache_dtype_str=self.kv_cache_dtype,\n"
        "            )",
        1,
    )

    # Also patch SlidingWindowSpec if it has cache_dtype_str support
    content = content.replace(
        "            return SlidingWindowSpec(\n"
        "                block_size=block_size,\n"
        "                num_kv_heads=self.num_kv_heads,\n"
        "                head_size=self.head_size,\n"
        "                dtype=self.kv_cache_torch_dtype,\n"
        "                sliding_window=self.sliding_window,\n"
        "            )",
        "            return SlidingWindowSpec(\n"
        "                block_size=block_size,\n"
        "                num_kv_heads=self.num_kv_heads,\n"
        "                head_size=self.head_size,\n"
        "                dtype=self.kv_cache_torch_dtype,\n"
        "                sliding_window=self.sliding_window,\n"
        "                cache_dtype_str=self.kv_cache_dtype,\n"
        "            )",
        1,
    )

    write(attn_path, content)
    print("  ~ attention.py (TQ metadata + cache_dtype_str in specs)")
else:
    print("  SKIP: attention.py already patched")

# ── Step 7: SM121 Triton autotuning for TQ decode kernel ──
# Default: num_warps=4, num_stages=2 (conservative). SM121 benefits from
# more warps (hide memory latency) and deeper pipeline (more stages).
# Also try larger BLOCK_N for better cache locality.
print("\n=== Step 7: SM121 Triton decode autotuning ===")
decode_path = f"{VLLM}/v1/attention/ops/triton_turboquant_decode.py"
content = read(decode_path)
if "num_warps=4,\n        num_stages=2," in content:
    # Add env var support for runtime autotuning without rebuild.
    # TQ_DECODE_WARPS (default 4), TQ_DECODE_STAGES (default 2), TQ_DECODE_BLOCK_N
    content = content.replace(
        "num_warps=4,\n        num_stages=2,",
        "num_warps=int(__import__('os').environ.get('TQ_DECODE_WARPS', '4')),\n"
        "        num_stages=int(__import__('os').environ.get('TQ_DECODE_STAGES', '2')),",
        1,
    )
    content = content.replace(
        "block_n = 8 if query.shape[-1] >= 256 else 16",
        "block_n = int(__import__('os').environ.get('TQ_DECODE_BLOCK_N', '8' if query.shape[-1] >= 256 else '16'))",
    )
    write(decode_path, content)
    print("  ~ triton_turboquant_decode.py (env var config: TQ_DECODE_WARPS/STAGES/BLOCK_N)")
else:
    print("  SKIP: decode kernel already tuned")

# ── Final report ──
print()
if _errors:
    print(f"=== TurboQuant: {len(_errors)} ERRORS ===", file=sys.stderr)
    for e in _errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
else:
    print("=== TurboQuant (mitkox) patch applied successfully ===")
    print("  Usage: vllm serve <model> --kv-cache-dtype turboquant35 --enable-turboquant")
