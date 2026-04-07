#!/usr/bin/env python3
"""EXP-19: INT8 LM Head v2 — Batched 2D Triton GEMV.

v1 problem: Python `for b in range(batch)` launched kernel once per token.
  228 launches/128 tokens = 5 per step. Each reads 485MB weights.
  Total: 11.34ms/step at 49% BW.

v2 fix: Single 2D kernel launch (row_blocks × batch). Reads weights ONCE.
  Autotuned BLOCK_M/BLOCK_K. No Python loop.
  Expected: ~3ms/step → save 8ms → 105+ tok/s.
"""

import os, sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py"


def apply():
    if not os.path.exists(TARGET):
        print(f"FAIL: {TARGET} not found"); sys.exit(1)

    with open(TARGET) as f:
        content = f.read()

    if "DGX_SPARK_INT8_LMHEAD_V2" in content:
        print("SKIP: INT8 LM Head v2 already applied"); return

    # Remove v1 marker if present (we're replacing it)
    if "DGX_SPARK_INT8_LMHEAD" in content and "DGX_SPARK_INT8_LMHEAD_V2" not in content:
        print("NOTE: Replacing v1 INT8 LM Head with v2")

    old = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)'''

    # Also handle v1 patch (replace the entire v1 block)
    old_v1_start = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # DGX_SPARK_INT8_LMHEAD: Fused INT8 GEMV via Triton'''

    new = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # DGX_SPARK_INT8_LMHEAD_V2: Batched 2D INT8 GEMV — single kernel launch
        if not hasattr(self, '_int8v2_initialized'):
            self._int8v2_initialized = True
            w = lm_head.weight.data
            if w.dtype in (torch.bfloat16, torch.float16) and w.shape[0] > 100000:
                scales = w.float().abs().amax(dim=1) / 127.0
                scales = scales.clamp(min=1e-12)
                w_int8 = (w.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
                lm_head._ww_int8 = w_int8
                lm_head._ww_scales = scales.to(torch.float16)
                orig_size = w.numel() * w.element_size()
                lm_head.weight.data = torch.empty(0, device=w.device, dtype=w.dtype)
                import sys as _sys
                print(f"DGX_SPARK_V2: LM Head -> INT8 Batched Triton ({list(w_int8.shape)}, saved {orig_size//1024//1024}MB)", file=_sys.stderr, flush=True)
                import triton
                import triton.language as tl
                @triton.jit
                def _k_v2(out_ptr, w_ptr, x_ptr, s_ptr, M, K,
                          stride_ob, stride_xb, NUM_BATCH: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
                    # 1D grid: each block processes ALL batch elements
                    # Weight tile loaded ONCE, reused for all batch inputs
                    pid_m = tl.program_id(0)
                    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    rmask = rows < M
                    # One accumulator per batch element (unrolled by compiler)
                    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    for ks in range(0, K, BLOCK_K):
                        co = ks + tl.arange(0, BLOCK_K)
                        km = co < K
                        # Load weight tile ONCE
                        w = tl.load(w_ptr + rows[:, None] * K + co[None, :],
                                    mask=rmask[:, None] & km[None, :], other=0).to(tl.float32)
                        # Reuse weight tile for each batch element
                        x0 = tl.load(x_ptr + 0 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                        acc0 += tl.sum(w * x0[None, :], axis=1)
                        if NUM_BATCH > 1:
                            x1 = tl.load(x_ptr + 1 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc1 += tl.sum(w * x1[None, :], axis=1)
                        if NUM_BATCH > 2:
                            x2 = tl.load(x_ptr + 2 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc2 += tl.sum(w * x2[None, :], axis=1)
                        if NUM_BATCH > 3:
                            x3 = tl.load(x_ptr + 3 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc3 += tl.sum(w * x3[None, :], axis=1)
                    # Scale and store
                    s = tl.load(s_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
                    tl.store(out_ptr + 0 * stride_ob + rows, (acc0 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 1:
                        tl.store(out_ptr + 1 * stride_ob + rows, (acc1 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 2:
                        tl.store(out_ptr + 2 * stride_ob + rows, (acc2 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 3:
                        tl.store(out_ptr + 3 * stride_ob + rows, (acc3 * s).to(tl.float16), mask=rmask)
                lm_head._ww_kernel_v2 = _k_v2

        if hasattr(lm_head, '_ww_int8'):
            M, K = lm_head._ww_int8.shape
            x = hidden_states.view(-1, K)
            batch = x.shape[0]
            out = torch.empty(batch, M, dtype=torch.float16, device=x.device)
            BLOCK_M = 128
            BLOCK_K = 256
            grid = ((M + BLOCK_M - 1) // BLOCK_M,)
            if batch <= 4:
                # Small batch (decode): shared-weight kernel reads weights ONCE
                nb = batch
                lm_head._ww_kernel_v2[grid](
                    out, lm_head._ww_int8, x.to(torch.float16),
                    lm_head._ww_scales, M, K,
                    out.stride(0), x.stride(0), NUM_BATCH=nb,
                    BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K)
            else:
                # Large batch (prefill/profile): fall back to per-row loop
                for b in range(batch):
                    lm_head._ww_kernel_v2[grid](
                        out[b:b+1], lm_head._ww_int8, x[b:b+1].to(torch.float16),
                        lm_head._ww_scales, M, K,
                        M, K, NUM_BATCH=1,
                        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K)
            logits = out.view(hidden_states.shape[:-1] + (M,))
            if embedding_bias is not None:
                logits = logits + embedding_bias
            return logits

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)'''

    # Try replacing v1 first
    if old_v1_start in content:
        # Find end of v1 block — it ends before the next method or at the fallback line
        idx_start = content.index(old_v1_start)
        # Find the fallback logits line that ends the v1 block
        fallback = "        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)"
        # Find the LAST occurrence of fallback after our start (v1 adds it at the end)
        remaining = content[idx_start:]
        last_fallback = remaining.rfind(fallback)
        if last_fallback >= 0:
            idx_end = idx_start + last_fallback + len(fallback)
            content = content[:idx_start] + new + content[idx_end:]
            with open(TARGET, "w") as f:
                f.write(content)
            print("OK: INT8 LM Head v2 patch applied (replaced v1)")
            return

    # Try clean apply (no v1)
    if old in content:
        content = content.replace(old, new)
        with open(TARGET, "w") as f:
            f.write(content)
        print("OK: INT8 LM Head v2 patch applied (clean)")
    else:
        print("FAIL: pattern not found (neither v1 nor clean)")
        sys.exit(1)


if __name__ == "__main__":
    apply()
