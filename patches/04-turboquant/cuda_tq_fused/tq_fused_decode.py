"""
TQ Fused Decode — Python wrapper (v2).

Handles:
1. Query pre-transform (Hadamard rotation → q_rot, q_qjl per group)
2. CUDA kernel call (attention on packed TQ data, outputs separate MSE/QJL accumulators)
3. Post-process (separate inverse Hadamard for MSE and QJL + scatter to full head_dim)
"""
import torch
from functools import lru_cache


# ─── Load CUDA extension ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_extension():
    """Load AOT-compiled CUDA extension."""
    import tq_fused_decode_ext as ext
    return ext


# ─── Hadamard helpers (ported from turboquant_kv_cache.py) ────────────────────

def _hadamard_block_sizes(dim: int) -> list[int]:
    """Decompose dim into power-of-2 block sizes for structured Hadamard."""
    sizes = []
    remaining = dim
    while remaining > 0:
        bs = 1
        while bs * 2 <= remaining:
            bs *= 2
        sizes.append(bs)
        remaining -= bs
    return sizes


def _build_structured_hadamard(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Build structured Hadamard matrix: H = D * H_block, where D = diag(signs)."""
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


# TurboQuant seeds (must match turboquant_kv_cache.py constants)
TURBOQUANT_SEED = 42
MSE_SEED_OFFSET = 0
QJL_SEED_OFFSET = 1000000
QJL_SCALE = 1.2533141373155003  # sqrt(pi/2)


@lru_cache(maxsize=4)
def _get_transforms(dim: int, seed_offset: int, device_str: str):
    """Get Hadamard transform matrix for a group dimension with given seed offset."""
    device = torch.device(device_str)
    return _build_structured_hadamard(dim, TURBOQUANT_SEED + seed_offset + dim, device)


@lru_cache(maxsize=4)
def _get_group_indices(head_size: int, device_str: str):
    """Get outlier/regular dimension indices (turboquant35: 50% split)."""
    device = torch.device(device_str)
    outlier_count = head_size // 2
    g0_idx = torch.arange(outlier_count, device=device)
    g1_idx = torch.arange(outlier_count, head_size, device=device)
    return g0_idx, g1_idx


def transform_queries(
    query: torch.Tensor,  # [..., head_size]
    head_size: int = 256,
    group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pre-transform queries for TQ fused attention.

    Returns: (q_rot_g0, q_qjl_g0, q_rot_g1, q_qjl_g1) each [..., group_dim] float32
    """
    device_str = str(query.device)
    g0_idx, g1_idx = group_indices or _get_group_indices(head_size, device_str)

    g0_dim = g0_idx.shape[0]
    g1_dim = g1_idx.shape[0]

    # Get transforms: F (MSE rotation), G (QJL rotation)
    F0 = _get_transforms(g0_dim, MSE_SEED_OFFSET, device_str)
    G0 = _get_transforms(g0_dim, QJL_SEED_OFFSET, device_str)
    F1 = _get_transforms(g1_dim, MSE_SEED_OFFSET, device_str)
    G1 = _get_transforms(g1_dim, QJL_SEED_OFFSET, device_str)

    # Gather query dimensions for each group
    q_g0 = query[..., g0_idx].float()
    q_g1 = query[..., g1_idx].float()

    # q_rot = F @ q (MSE rotation)
    q_rot_g0 = torch.matmul(q_g0, F0.T)
    q_rot_g1 = torch.matmul(q_g1, F1.T)

    # q_qjl = (G @ q) * (QJL_SCALE / dim)
    # QJL_SCALE/dim is baked into q_qjl so the kernel just does dot(signs, q_qjl)
    q_qjl_g0 = torch.matmul(q_g0, G0.T) * (QJL_SCALE / g0_dim)
    q_qjl_g1 = torch.matmul(q_g1, G1.T) * (QJL_SCALE / g1_dim)

    return q_rot_g0.contiguous(), q_qjl_g0.contiguous(), q_rot_g1.contiguous(), q_qjl_g1.contiguous()


def postprocess_output(
    out_mse_g0: torch.Tensor,   # [batch, num_qo_heads, g0_dim]
    out_qjl_g0: torch.Tensor,
    out_mse_g1: torch.Tensor,   # [batch, num_qo_heads, g1_dim]
    out_qjl_g1: torch.Tensor,
    head_size: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Post-process: separate inverse Hadamard for MSE (F^T) and QJL (G^T),
    then scatter to full head_dim.

    Returns: [batch, num_qo_heads, head_size]
    """
    device_str = str(out_mse_g0.device)
    g0_idx, g1_idx = group_indices or _get_group_indices(head_size, device_str)

    g0_dim = g0_idx.shape[0]
    g1_dim = g1_idx.shape[0]

    # Get inverse transforms
    F0 = _get_transforms(g0_dim, MSE_SEED_OFFSET, device_str)  # F is orthogonal: F^T = F^{-1}
    G0 = _get_transforms(g0_dim, QJL_SEED_OFFSET, device_str)
    F1 = _get_transforms(g1_dim, MSE_SEED_OFFSET, device_str)
    G1 = _get_transforms(g1_dim, QJL_SEED_OFFSET, device_str)

    # Inverse Hadamard: F^T @ acc_mse and G^T @ acc_qjl (separate transforms!)
    recon_g0 = torch.matmul(out_mse_g0.float(), F0) + torch.matmul(out_qjl_g0.float(), G0)
    recon_g1 = torch.matmul(out_mse_g1.float(), F1) + torch.matmul(out_qjl_g1.float(), G1)

    # Scatter to full head dimension
    batch_shape = out_mse_g0.shape[:-1]
    output = torch.zeros(*batch_shape, head_size, device=out_mse_g0.device, dtype=torch.float32)
    output[..., g0_idx] = recon_g0
    output[..., g1_idx] = recon_g1

    return output.to(dtype)


def tq_fused_attention(
    query: torch.Tensor,        # [batch, num_qo_heads, head_size]
    kv_data: torch.Tensor,      # [num_pages, 2, page_size, num_kv_heads, cache_dim] uint8
    kv_indptr: torch.Tensor,    # [batch+1] int32
    kv_indices: torch.Tensor,   # [total_pages] int32
    kv_last_page_len: torch.Tensor,  # [batch] int32
    codebook_g0: torch.Tensor,  # [8] float32
    codebook_g1: torch.Tensor,  # [4] float32
    page_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    sm_scale: float,
    head_size: int = 256,
    cache_dim: int = 128,       # actual cache last dim (padded from 120)
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Full TQ fused attention pipeline:
    1. Pre-transform queries (Hadamard rotations)
    2. CUDA kernel (attention on packed TQ data)
    3. Post-process (inverse Hadamard + scatter)

    Returns: [batch, num_qo_heads, head_size] in output_dtype
    """
    ext = _load_extension()

    # 1. Pre-transform queries
    q_rot_g0, q_qjl_g0, q_rot_g1, q_qjl_g1 = transform_queries(query, head_size)

    # 2. CUDA kernel — returns 4 separate accumulators
    out_mse_g0, out_qjl_g0, out_mse_g1, out_qjl_g1 = ext.decode(
        q_rot_g0, q_qjl_g0, q_rot_g1, q_qjl_g1,
        kv_data,
        kv_indptr, kv_indices, kv_last_page_len,
        codebook_g0, codebook_g1,
        page_size, num_qo_heads, num_kv_heads, cache_dim, sm_scale,
    )

    # 3. Post-process (separate F^T for MSE, G^T for QJL, then scatter)
    return postprocess_output(out_mse_g0, out_qjl_g0, out_mse_g1, out_qjl_g1,
                              head_size, output_dtype)
