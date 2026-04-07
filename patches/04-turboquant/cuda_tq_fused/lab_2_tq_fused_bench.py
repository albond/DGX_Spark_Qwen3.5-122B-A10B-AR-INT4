#!/usr/bin/env python3
"""
Lab 2: TQ Fused Decode Kernel — Correctness & Performance Test.

Tests the CUDA kernel that computes attention directly on TQ packed data.
Compares with reference implementation (dequant TQ → bf16 → standard attention).

Run inside a container with the compiled extension:
  cd /opt/cuda_tq_fused && python setup.py install
  python lab_2_tq_fused_bench.py
"""

import sys
import time
import math
import torch
import torch.nn.functional as F

# Add path for our extension
sys.path.insert(0, "/opt/cuda_tq_fused")

# ─── Test configuration ──────────────────────────────────────────────────────

HEAD_SIZE = 256
G0_DIM = 128  # outlier dims
G1_DIM = 128  # regular dims
G0_MSE_BITS = 3
G1_MSE_BITS = 2
G0_MSE_LEVELS = 8
G1_MSE_LEVELS = 4
NUM_QO_HEADS = 16
NUM_KV_HEADS = 2
PAGE_SIZE = 16
CACHE_DIM = 128  # padded from 120
PACKED_LOGICAL = 120

QJL_SCALE = 1.2533141373155003  # sqrt(pi/2)

DEVICE = torch.device("cuda:0")


# ─── TQ Pack/Unpack helpers (reference implementation) ────────────────────────

def pack_mse_indices(indices: torch.Tensor, dim: int, mse_bits: int) -> torch.Tensor:
    """Pack MSE indices into bytes. indices: [..., dim] with values in [0, 2^mse_bits)."""
    num_bytes = (dim * mse_bits + 7) // 8
    packed = torch.zeros(*indices.shape[:-1], num_bytes, dtype=torch.uint8, device=indices.device)
    for d in range(dim):
        idx = indices[..., d].int()
        bit_pos = d * mse_bits
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        packed[..., byte_idx] |= (idx << bit_offset).to(torch.uint8)
        if bit_offset + mse_bits > 8:
            packed[..., byte_idx + 1] |= (idx >> (8 - bit_offset)).to(torch.uint8)
    return packed


def pack_qjl_signs(signs: torch.Tensor, dim: int) -> torch.Tensor:
    """Pack sign bits (0 or 1) into bytes."""
    num_bytes = (dim + 7) // 8
    packed = torch.zeros(*signs.shape[:-1], num_bytes, dtype=torch.uint8, device=signs.device)
    for d in range(dim):
        bit = signs[..., d].to(torch.uint8)
        byte_idx = d // 8
        bit_offset = d % 8
        packed[..., byte_idx] |= (bit << bit_offset)
    return packed


def create_tq_packed(
    vectors: torch.Tensor,  # [..., head_size] bf16/float
    codebook_g0: torch.Tensor,
    codebook_g1: torch.Tensor,
    F0: torch.Tensor,
    G0: torch.Tensor,
    F1: torch.Tensor,
    G1: torch.Tensor,
) -> torch.Tensor:
    """Simulate TQ encoding: compress vectors into packed format.

    Returns: [..., CACHE_DIM] uint8 (padded)
    """
    vf = vectors.float()
    shape = vf.shape[:-1]

    # Split into groups (default: first half outlier, second half regular)
    g0_data = vf[..., :G0_DIM]
    g1_data = vf[..., G0_DIM:]

    packed = torch.zeros(*shape, CACHE_DIM, dtype=torch.uint8, device=vectors.device)
    cursor = 0

    for g_data, F_mat, G_mat, codebook, mse_bits, mse_bytes, qjl_bytes, dim in [
        (g0_data, F0, G0, codebook_g0, G0_MSE_BITS, 48, 16, G0_DIM),
        (g1_data, F1, G1, codebook_g1, G1_MSE_BITS, 32, 16, G1_DIM),
    ]:
        # Compute norms
        norms = g_data.norm(dim=-1, keepdim=True)
        unit = g_data / norms.clamp_min(1e-8)

        # MSE: rotate and quantize
        rotated = torch.matmul(unit, F_mat.T)
        # Find nearest centroid
        diffs = (rotated.unsqueeze(-1) - codebook.unsqueeze(0)).abs()
        indices = diffs.argmin(dim=-1)
        rotated_hat = codebook[indices]
        mse_hat = torch.matmul(rotated_hat, F_mat)

        # Residual
        residual = unit - mse_hat
        res_norms = residual.norm(dim=-1, keepdim=True)
        # QJL signs
        qjl_proj = torch.matmul(residual, G_mat.T)
        qjl_signs = (qjl_proj >= 0).int()

        # Pack MSE indices
        mse_packed = pack_mse_indices(indices, dim, mse_bits)
        # Pack QJL signs
        qjl_packed = pack_qjl_signs(qjl_signs, dim)

        # Pack norms as float16
        norm_f16 = norms.squeeze(-1).half()
        norm_bytes = torch.zeros(*shape, 2, dtype=torch.uint8, device=vectors.device)
        # Manual float16 → 2 bytes
        norm_raw = norm_f16.view(torch.uint8).reshape(*shape, 2)
        res_norm_f16 = res_norms.squeeze(-1).half()
        res_norm_raw = res_norm_f16.view(torch.uint8).reshape(*shape, 2)

        # Assemble packed group
        packed[..., cursor:cursor+mse_bytes] = mse_packed
        cursor += mse_bytes
        packed[..., cursor:cursor+qjl_bytes] = qjl_packed
        cursor += qjl_bytes
        packed[..., cursor:cursor+2] = norm_raw
        cursor += 2
        packed[..., cursor:cursor+2] = res_norm_raw
        cursor += 2

    return packed


# ─── Reference: standard attention on bf16 ────────────────────────────────────

def reference_attention(
    query: torch.Tensor,    # [batch, num_qo_heads, head_size]
    keys: torch.Tensor,     # [batch, seq_len, num_kv_heads, head_size]
    values: torch.Tensor,   # [batch, seq_len, num_kv_heads, head_size]
    sm_scale: float,
) -> torch.Tensor:
    """Standard GQA attention (no paging, for reference)."""
    batch, nq, hd = query.shape
    _, seq_len, nkv, _ = keys.shape
    gqa_ratio = nq // nkv

    output = torch.zeros_like(query)
    for h in range(nq):
        kv_h = h // gqa_ratio
        q = query[:, h, :]  # [batch, hd]
        k = keys[:, :, kv_h, :]  # [batch, seq_len, hd]
        v = values[:, :, kv_h, :]  # [batch, seq_len, hd]
        scores = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) * sm_scale  # [batch, seq_len]
        weights = torch.softmax(scores, dim=-1)
        output[:, h, :] = torch.matmul(weights.unsqueeze(1), v).squeeze(1)
    return output


# ─── Build Hadamard transforms ───────────────────────────────────────────────

def _hadamard_block_sizes(dim):
    sizes = []
    remaining = dim
    while remaining > 0:
        bs = 1
        while bs * 2 <= remaining:
            bs *= 2
        sizes.append(bs)
        remaining -= bs
    return sizes


def _build_structured_hadamard(dim, seed, device):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    signs = torch.where(
        torch.rand(dim, generator=gen) < 0.5,
        torch.ones(dim), -torch.ones(dim)
    ).to(device=device, dtype=torch.float32)

    H = torch.zeros(dim, dim, device=device, dtype=torch.float32)
    offset = 0
    for bs in _hadamard_block_sizes(dim):
        h = torch.ones(1, 1, device=device, dtype=torch.float32)
        size = 1
        while size < bs:
            h = torch.cat([
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ], dim=0)
            size *= 2
        h = h / (bs ** 0.5)
        H[offset:offset+bs, offset:offset+bs] = h
        offset += bs

    return torch.diag(signs) @ H


SEED = 42
MSE_OFFSET = 0
QJL_OFFSET = 1000000


def get_transforms(dim, device):
    F_mat = _build_structured_hadamard(dim, SEED + MSE_OFFSET + dim, device)
    G_mat = _build_structured_hadamard(dim, SEED + QJL_OFFSET + dim, device)
    return F_mat, G_mat


# ─── Main test ────────────────────────────────────────────────────────────────

def test_correctness(batch_size=1, seq_len=64):
    print(f"\n{'='*60}")
    print(f"Correctness test: batch={batch_size}, seq_len={seq_len}")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Generate random data
    query = torch.randn(batch_size, NUM_QO_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.bfloat16)
    keys = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.bfloat16)
    values = torch.randn(batch_size, seq_len, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(HEAD_SIZE)

    # Reference: standard attention
    ref_output = reference_attention(query, keys, values, sm_scale)
    print(f"Reference output: shape={ref_output.shape}, norm={ref_output.float().norm():.4f}")

    # Build transforms and codebooks
    F0, G0 = get_transforms(G0_DIM, DEVICE)
    F1, G1 = get_transforms(G1_DIM, DEVICE)

    # Lloyd-Max codebooks (simplified: uniform quantile approximation)
    cb_g0 = torch.linspace(-1, 1, G0_MSE_LEVELS, device=DEVICE, dtype=torch.float32)
    cb_g1 = torch.linspace(-1, 1, G1_MSE_LEVELS, device=DEVICE, dtype=torch.float32)

    # Pack KV into TQ format
    # For paged cache: [num_pages, 2, page_size, num_kv_heads, cache_dim]
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    kv_cache = torch.zeros(num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, CACHE_DIM,
                           dtype=torch.uint8, device=DEVICE)

    for s in range(seq_len):
        page_id = s // PAGE_SIZE
        pos_in_page = s % PAGE_SIZE
        for kv_idx, data in enumerate([keys, values]):
            for h in range(NUM_KV_HEADS):
                vec = data[:, s, h, :]  # [batch, head_size]
                packed = create_tq_packed(vec, cb_g0, cb_g1, F0, G0, F1, G1)
                kv_cache[page_id, kv_idx, pos_in_page, h, :] = packed[0]  # batch=0 for simplicity

    # Page table (simple: pages in order)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    last_page_len = seq_len - (num_pages - 1) * PAGE_SIZE
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=DEVICE)

    # ── Test our fused kernel ──
    from tq_fused_decode import tq_fused_attention

    tq_output = tq_fused_attention(
        query, kv_cache,
        kv_indptr, kv_indices, kv_last_page_len,
        cb_g0, cb_g1,
        PAGE_SIZE, NUM_QO_HEADS, NUM_KV_HEADS, sm_scale,
        HEAD_SIZE, CACHE_DIM, torch.bfloat16,
    )
    print(f"TQ fused output: shape={tq_output.shape}, norm={tq_output.float().norm():.4f}")

    # Compare
    diff = (ref_output.float() - tq_output.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos_sim = F.cosine_similarity(
        ref_output.float().reshape(-1).unsqueeze(0),
        tq_output.float().reshape(-1).unsqueeze(0),
    ).item()

    print(f"\nMax absolute error: {max_err:.6f}")
    print(f"Mean absolute error: {mean_err:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    # TQ introduces quantization error, so we expect non-zero error
    # but it should be bounded (typically < 0.1 for reasonable quantization)
    if cos_sim > 0.9:
        print("PASS: cosine similarity > 0.9 (accounting for TQ quantization)")
    else:
        print("WARN: cosine similarity low — check implementation")

    return cos_sim


def benchmark(batch_size=1, seq_len=4096, num_warmup=5, num_iters=20):
    print(f"\n{'='*60}")
    print(f"Performance test: batch={batch_size}, seq_len={seq_len}")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Generate data
    query = torch.randn(batch_size, NUM_QO_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(HEAD_SIZE)

    F0, G0 = get_transforms(G0_DIM, DEVICE)
    F1, G1 = get_transforms(G1_DIM, DEVICE)
    cb_g0 = torch.linspace(-1, 1, G0_MSE_LEVELS, device=DEVICE, dtype=torch.float32)
    cb_g1 = torch.linspace(-1, 1, G1_MSE_LEVELS, device=DEVICE, dtype=torch.float32)

    # Create packed cache
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    kv_cache = torch.randint(0, 256, (num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, CACHE_DIM),
                             dtype=torch.uint8, device=DEVICE)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    last_page_len = seq_len - (num_pages - 1) * PAGE_SIZE
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=DEVICE)

    from tq_fused_decode import tq_fused_attention

    # Warmup
    for _ in range(num_warmup):
        _ = tq_fused_attention(
            query, kv_cache, kv_indptr, kv_indices, kv_last_page_len,
            cb_g0, cb_g1, PAGE_SIZE, NUM_QO_HEADS, NUM_KV_HEADS, sm_scale,
            HEAD_SIZE, CACHE_DIM,
        )
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iters):
        _ = tq_fused_attention(
            query, kv_cache, kv_indptr, kv_indices, kv_last_page_len,
            cb_g0, cb_g1, PAGE_SIZE, NUM_QO_HEADS, NUM_KV_HEADS, sm_scale,
            HEAD_SIZE, CACHE_DIM,
        )
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
    print(f"Time per decode step: {elapsed_ms:.3f} ms")

    # Bandwidth analysis
    bytes_read_tq = seq_len * NUM_KV_HEADS * 2 * PACKED_LOGICAL  # K+V, packed
    bytes_read_bf16 = seq_len * NUM_KV_HEADS * 2 * HEAD_SIZE * 2  # K+V, bf16
    bw_tq = bytes_read_tq / (elapsed_ms / 1000) / 1e9
    print(f"TQ bandwidth: {bw_tq:.1f} GB/s (reading {bytes_read_tq/1e6:.1f} MB)")
    print(f"Equivalent bf16 bandwidth: {bytes_read_bf16/1e6:.1f} MB (4.27x more)")
    print(f"SM121 peak: 273 GB/s, utilization: {bw_tq/273*100:.0f}%")

    return elapsed_ms


if __name__ == "__main__":
    print("TQ Fused Decode Kernel — Lab 2")
    print(f"Config: head_size={HEAD_SIZE}, groups=128+128, "
          f"MSE bits=3+2, QO={NUM_QO_HEADS}, KV={NUM_KV_HEADS}")

    # Correctness
    cos_sim = test_correctness(batch_size=1, seq_len=64)

    if cos_sim > 0.8:
        # Performance at various seq lengths
        for sl in [256, 1024, 4096, 16384]:
            benchmark(batch_size=1, seq_len=sl)
    else:
        print("\nSkipping perf benchmark due to low correctness.")
