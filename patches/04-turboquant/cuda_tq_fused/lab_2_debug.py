#!/usr/bin/env python3
"""Minimal debug test for TQ fused kernel — isolate the bug."""
import sys, math, torch
sys.path.insert(0, "/tmp/cuda_tq_fused")
import tq_fused_decode_ext as ext

DEVICE = torch.device("cuda:0")

# Simple test: 1 batch, 1 KV position, verify kernel output matches manual computation
print("=== Debug Test: 1 batch, 1 KV position ===")

# Config
G0_DIM = 128
G1_DIM = 128
HEAD_SIZE = 256
NUM_QO = 16
NUM_KV = 2
PAGE_SIZE = 16
CACHE_DIM = 128
QJL_SCALE = 1.2533141373155003
SM_SCALE = 1.0 / math.sqrt(HEAD_SIZE)

# Create codebooks
cb_g0 = torch.linspace(-1, 1, 8, device=DEVICE, dtype=torch.float32)
cb_g1 = torch.linspace(-1, 1, 4, device=DEVICE, dtype=torch.float32)

# Create packed KV data with KNOWN values
# 1 page, 2 (K+V), PAGE_SIZE slots, NUM_KV heads, CACHE_DIM bytes
kv_data = torch.zeros(1, 2, PAGE_SIZE, NUM_KV, CACHE_DIM, dtype=torch.uint8, device=DEVICE)

# Fill slot 0, head 0 with specific values:
# Group 0 MSE: all indices = 4 (codebook[4] = 0.14286 for 8-level linspace)
# Group 1 MSE: all indices = 2 (codebook[2] = 0.33333 for 4-level linspace)
# Group 0 QJL: all signs = 1 (all bits set)
# Group 1 QJL: all signs = 0 (all bits clear)
# Norms: vector_norm = 1.0, residual_norm = 0.5

# Pack for K (slot 0, kv=0) and V (slot 0, kv=1) for BOTH kv heads
for kv_idx in range(2):  # K and V
    for kv_head in range(NUM_KV):
        slot = kv_data[0, kv_idx, 0, kv_head, :]

        # Group 0: 3-bit MSE indices, all = 4 (binary: 100)
        # 128 dims * 3 bits = 384 bits = 48 bytes
        for d in range(G0_DIM):
            bit_pos = d * 3
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            val = 4  # centroid index
            slot[byte_idx] = slot[byte_idx] | torch.tensor(((val << bit_offset) & 0xFF), dtype=torch.uint8, device=DEVICE)
            if bit_offset + 3 > 8:
                slot[byte_idx + 1] = slot[byte_idx + 1] | torch.tensor(((val >> (8 - bit_offset)) & 0xFF), dtype=torch.uint8, device=DEVICE)

        # Group 0 QJL: all bits = 1 (bytes at offset 48, 16 bytes)
        slot[48:64] = 0xFF

        # Group 0 norms at offset 64-67: vnorm=1.0, rnorm=0.5
        vnorm_fp16 = torch.tensor([1.0], dtype=torch.float16)
        rnorm_fp16 = torch.tensor([0.5], dtype=torch.float16)
        vnorm_bytes = vnorm_fp16.view(torch.uint8)
        rnorm_bytes = rnorm_fp16.view(torch.uint8)
        slot[64] = vnorm_bytes[0]
        slot[65] = vnorm_bytes[1]
        slot[66] = rnorm_bytes[0]
        slot[67] = rnorm_bytes[1]

        # Group 1: 2-bit MSE indices, all = 2 (binary: 10)
        # 128 dims * 2 bits = 256 bits = 32 bytes, starting at offset 68
        for d in range(G1_DIM):
            bit_pos = d * 2
            byte_idx = 68 + bit_pos // 8
            bit_offset = bit_pos % 8
            val = 2
            slot[byte_idx] = slot[byte_idx] | torch.tensor(((val << bit_offset) & 0xFF), dtype=torch.uint8, device=DEVICE)

        # Group 1 QJL: all bits = 0 (already zero) at offset 100

        # Group 1 norms at offset 116-119
        slot[116] = vnorm_bytes[0]
        slot[117] = vnorm_bytes[1]
        slot[118] = rnorm_bytes[0]
        slot[119] = rnorm_bytes[1]

# Page table: 1 batch, 1 page
kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
kv_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
kv_last_page_len = torch.tensor([1], dtype=torch.int32, device=DEVICE)  # 1 position in last page

# Random queries (pre-transformed)
torch.manual_seed(42)
q_rot_g0 = torch.randn(1, NUM_QO, G0_DIM, device=DEVICE, dtype=torch.float32)
q_qjl_g0 = torch.randn(1, NUM_QO, G0_DIM, device=DEVICE, dtype=torch.float32)
q_rot_g1 = torch.randn(1, NUM_QO, G1_DIM, device=DEVICE, dtype=torch.float32)
q_qjl_g1 = torch.randn(1, NUM_QO, G1_DIM, device=DEVICE, dtype=torch.float32)

# Run kernel
results = ext.decode(
    q_rot_g0, q_qjl_g0, q_rot_g1, q_qjl_g1,
    kv_data,
    kv_indptr, kv_indices, kv_last_page_len,
    cb_g0, cb_g1,
    PAGE_SIZE, NUM_QO, NUM_KV, CACHE_DIM, SM_SCALE,
)
out_mse_g0, out_qjl_g0, out_mse_g1, out_qjl_g1 = results

print(f"Kernel output shapes: mse_g0={out_mse_g0.shape}, qjl_g0={out_qjl_g0.shape}")
print(f"out_mse_g0 stats: mean={out_mse_g0.mean():.6f}, std={out_mse_g0.std():.6f}, norm={out_mse_g0.norm():.4f}")
print(f"out_qjl_g0 stats: mean={out_qjl_g0.mean():.6f}, std={out_qjl_g0.std():.6f}, norm={out_qjl_g0.norm():.4f}")
print(f"out_mse_g1 stats: mean={out_mse_g1.mean():.6f}, std={out_mse_g1.std():.6f}, norm={out_mse_g1.norm():.4f}")
print(f"out_qjl_g1 stats: mean={out_qjl_g1.mean():.6f}, std={out_qjl_g1.std():.6f}, norm={out_qjl_g1.norm():.4f}")

# ── Manual computation (Python reference) ──
# With 1 KV position, softmax weight = 1.0 (exp(score)*1/exp(score) = 1)
# So output = vnorm * codebook[idx] for MSE, vnorm * rnorm * sign for QJL

# Group 0: vnorm=1.0, rnorm=0.5, idx=4 → cb_g0[4]=0.14286, signs=+1
expected_mse_g0 = 1.0 * cb_g0[4].item()  # = 0.14286 for all dims
expected_qjl_g0 = 1.0 * 0.5 * 1.0  # = 0.5 for all dims (signs=+1)

# Group 1: vnorm=1.0, rnorm=0.5, idx=2 → cb_g1[2]=0.33333, signs=-1
expected_mse_g1 = 1.0 * cb_g1[2].item()  # = 0.33333
expected_qjl_g1 = 1.0 * 0.5 * (-1.0)  # = -0.5 (signs=0 → -1)

print(f"\nExpected (all dims same):")
print(f"  mse_g0: {expected_mse_g0:.6f}")
print(f"  qjl_g0: {expected_qjl_g0:.6f}")
print(f"  mse_g1: {expected_mse_g1:.6f}")
print(f"  qjl_g1: {expected_qjl_g1:.6f}")

print(f"\nActual (first head, first 8 dims):")
print(f"  mse_g0: {out_mse_g0[0, 0, :8].tolist()}")
print(f"  qjl_g0: {out_qjl_g0[0, 0, :8].tolist()}")
print(f"  mse_g1: {out_mse_g1[0, 0, :8].tolist()}")
print(f"  qjl_g1: {out_qjl_g1[0, 0, :8].tolist()}")

# Check
mse_g0_ok = torch.allclose(out_mse_g0, torch.full_like(out_mse_g0, expected_mse_g0), atol=0.01)
qjl_g0_ok = torch.allclose(out_qjl_g0, torch.full_like(out_qjl_g0, expected_qjl_g0), atol=0.01)
mse_g1_ok = torch.allclose(out_mse_g1, torch.full_like(out_mse_g1, expected_mse_g1), atol=0.01)
qjl_g1_ok = torch.allclose(out_qjl_g1, torch.full_like(out_qjl_g1, expected_qjl_g1), atol=0.01)

print(f"\nmse_g0 correct: {mse_g0_ok}")
print(f"qjl_g0 correct: {qjl_g0_ok}")
print(f"mse_g1 correct: {mse_g1_ok}")
print(f"qjl_g1 correct: {qjl_g1_ok}")

if all([mse_g0_ok, qjl_g0_ok, mse_g1_ok, qjl_g1_ok]):
    print("\n=== KERNEL ARITHMETIC VERIFIED ===")
else:
    print("\n=== BUG DETECTED — kernel arithmetic wrong ===")
