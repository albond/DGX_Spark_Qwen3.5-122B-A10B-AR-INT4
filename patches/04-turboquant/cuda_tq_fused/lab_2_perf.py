#!/usr/bin/env python3
"""
Lab 2: TQ Fused Decode — Performance Benchmark.

Kernel arithmetic already verified (lab_2_debug.py).
Now benchmark the CUDA kernel at various sequence lengths and compare bandwidth.
Uses random packed data (correctness of encoding not needed for perf measurement).
"""
import sys, math, time, torch
sys.path.insert(0, "/tmp/cuda_tq_fused")
import tq_fused_decode_ext as ext

DEVICE = torch.device("cuda:0")

G0_DIM = 128
G1_DIM = 128
NUM_QO = 16
NUM_KV = 2
PAGE_SIZE = 16
CACHE_DIM = 128
PACKED_LOGICAL = 120
HEAD_SIZE = 256
SM_SCALE = 1.0 / math.sqrt(HEAD_SIZE)


def bench_kernel_only(seq_len: int, batch_size: int = 1, warmup: int = 10, iters: int = 50):
    """Benchmark just the CUDA kernel (no pre/post-processing)."""
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE

    # Random packed data
    kv_data = torch.randint(0, 256, (num_pages, 2, PAGE_SIZE, NUM_KV, CACHE_DIM),
                            dtype=torch.uint8, device=DEVICE)
    # But set valid norms (bytes 64-67, 116-119 for each slot need valid float16)
    # Just set norms to 1.0 for all slots
    one_f16 = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    for p in range(num_pages):
        for kv in range(2):
            for s in range(PAGE_SIZE):
                for h in range(NUM_KV):
                    slot = kv_data[p, kv, s, h]
                    # G0 norms
                    slot[64] = one_f16[0]; slot[65] = one_f16[1]
                    slot[66] = one_f16[0]; slot[67] = one_f16[1]
                    # G1 norms
                    slot[116] = one_f16[0]; slot[117] = one_f16[1]
                    slot[118] = one_f16[0]; slot[119] = one_f16[1]

    kv_indptr = torch.tensor([i * num_pages // batch_size for i in range(batch_size + 1)],
                             dtype=torch.int32, device=DEVICE)
    # For batch=1, just [0, num_pages]
    if batch_size == 1:
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    last_page_len = seq_len - (num_pages - 1) * PAGE_SIZE
    kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device=DEVICE)

    # Random pre-transformed queries
    q_rot_g0 = torch.randn(batch_size, NUM_QO, G0_DIM, device=DEVICE, dtype=torch.float32)
    q_qjl_g0 = torch.randn(batch_size, NUM_QO, G0_DIM, device=DEVICE, dtype=torch.float32)
    q_rot_g1 = torch.randn(batch_size, NUM_QO, G1_DIM, device=DEVICE, dtype=torch.float32)
    q_qjl_g1 = torch.randn(batch_size, NUM_QO, G1_DIM, device=DEVICE, dtype=torch.float32)

    cb_g0 = torch.linspace(-1, 1, 8, device=DEVICE, dtype=torch.float32)
    cb_g1 = torch.linspace(-1, 1, 4, device=DEVICE, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        ext.decode(q_rot_g0, q_qjl_g0, q_rot_g1, q_qjl_g1,
                   kv_data, kv_indptr, kv_indices, kv_last_page_len,
                   cb_g0, cb_g1, PAGE_SIZE, NUM_QO, NUM_KV, CACHE_DIM, SM_SCALE)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ext.decode(q_rot_g0, q_qjl_g0, q_rot_g1, q_qjl_g1,
                   kv_data, kv_indptr, kv_indices, kv_last_page_len,
                   cb_g0, cb_g1, PAGE_SIZE, NUM_QO, NUM_KV, CACHE_DIM, SM_SCALE)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters

    # Bandwidth analysis
    bytes_read_kv = seq_len * NUM_KV * 2 * PACKED_LOGICAL  # K+V
    bytes_read_q = batch_size * NUM_QO * (G0_DIM + G1_DIM) * 4 * 4  # 4 query tensors, float32
    bytes_written = batch_size * NUM_QO * (G0_DIM * 2 + G1_DIM * 2) * 4  # 4 output tensors
    total_bytes = bytes_read_kv + bytes_read_q + bytes_written
    bw = total_bytes / (ms / 1000) / 1e9

    # Equivalent bf16 bandwidth
    bytes_bf16 = seq_len * NUM_KV * 2 * HEAD_SIZE * 2
    compression_ratio = bytes_bf16 / bytes_read_kv

    return ms, bw, bytes_read_kv, bytes_bf16, compression_ratio


if __name__ == "__main__":
    print("TQ Fused Decode — Kernel Performance Benchmark")
    print(f"SM121 peak bandwidth: 273 GB/s")
    print(f"Config: QO={NUM_QO}, KV={NUM_KV}, head={HEAD_SIZE}, page={PAGE_SIZE}")
    print()
    print(f"{'seq_len':>8} {'ms':>8} {'GB/s':>8} {'TQ MB':>8} {'bf16 MB':>8} {'ratio':>6} {'util%':>6}")
    print("-" * 60)

    for seq_len in [256, 1024, 4096, 8192, 16384, 32768]:
        ms, bw, tq_bytes, bf16_bytes, ratio = bench_kernel_only(seq_len)
        print(f"{seq_len:>8} {ms:>8.3f} {bw:>8.1f} {tq_bytes/1e6:>8.2f} {bf16_bytes/1e6:>8.2f} {ratio:>6.1f}x {bw/273*100:>5.0f}%")

    print()
    print("Note: This benchmarks ONLY the CUDA kernel (no query pre-transform or post-process)")
    print("Pre/post-processing adds ~0.1-0.3ms overhead (independent of seq_len)")
