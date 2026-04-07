/*
 * TQ Fused Decode v8 — Best of all versions
 *
 * Combines: split-KV (v4) + batch unpacking (v5) + single reduction (v5b)
 *           + shared memory tiles (NOT warp shuffle — smem is faster)
 *           + register codebooks + distributed tile loading
 *
 * NCU-informed: launch_bounds(256, 2) for 0 spill, ~80 regs
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

constexpr int WARP_SIZE = 32;
constexpr int G0_DIM = 128, G1_DIM = 128;
constexpr int G0_MSE_BITS = 3, G1_MSE_BITS = 2;
constexpr int G0_MSE_LEVELS = 8, G1_MSE_LEVELS = 4;
constexpr int G0_PACKED = 68;
constexpr int G0_QJL_OFF = 48, G0_VNORM_OFF = 64, G0_RNORM_OFF = 66;
constexpr int G1_OFF = 68;
constexpr int G1_QJL_OFF = 100, G1_VNORM_OFF = 116, G1_RNORM_OFF = 118;
constexpr int EPT = 4;
constexpr int TILE_N = 64;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}

__device__ __forceinline__ float rd16(const uint8_t* p) {
    uint16_t b = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    __half h; memcpy(&h, &b, 2); return __half2float(h);
}

// PTX bit field extract: single instruction for (val >> offset) & ((1<<bits)-1)
__device__ __forceinline__ int bfe(uint32_t val, int offset, int bits) {
    int result;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(val), "r"(offset), "r"(bits));
    return result;
}

__device__ __forceinline__ float cb8(int i, float c0,float c1,float c2,float c3,
                                       float c4,float c5,float c6,float c7) {
    float a = (i&1)?c1:c0, b = (i&1)?c3:c2, c = (i&1)?c5:c4, d = (i&1)?c7:c6;
    float e = (i&2)?b:a, f = (i&2)?d:c;
    return (i&4)?f:e;
}

__device__ __forceinline__ float cb4(int i, float c0,float c1,float c2,float c3) {
    float a = (i&1)?c1:c0, b = (i&1)?c3:c2;
    return (i&2)?b:a;
}

// ─── Main kernel ─────────────────────────────────────────────────────────────

template <int CACHE_DIM>
__global__ __launch_bounds__(256, 2)
void tq_fused_v8(
    const float* __restrict__ q_rot_g0, const float* __restrict__ q_qjl_g0,
    const float* __restrict__ q_rot_g1, const float* __restrict__ q_qjl_g1,
    const uint8_t* __restrict__ kv_data,
    const int32_t* __restrict__ kv_indptr, const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_last_page_len,
    const float* __restrict__ codebook_g0, const float* __restrict__ codebook_g1,
    float* __restrict__ p_mse_g0, float* __restrict__ p_qjl_g0,
    float* __restrict__ p_mse_g1, float* __restrict__ p_qjl_g1,
    float* __restrict__ p_m, float* __restrict__ p_d,
    int page_size, int num_qo, int num_kv, int batch_size,
    int chunk_size, int num_chunks, float sm_scale
) {
    const int batch_idx = blockIdx.x / num_chunks;
    const int chunk_idx = blockIdx.x % num_chunks;
    const int kvh = blockIdx.y;
    const int nqpk = num_qo / num_kv;
    const int wid = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int qoh = kvh * nqpk + wid;
    const int nthreads = blockDim.x;

    if (batch_idx >= batch_size || wid >= nqpk) return;

    extern __shared__ uint8_t smem[];
    uint8_t* tile_k = smem;
    uint8_t* tile_v = smem + TILE_N * CACHE_DIM;

    // Codebooks in shared memory (saves 12 registers → better occupancy)
    __shared__ float s_cb0[G0_MSE_LEVELS];
    __shared__ float s_cb1[G1_MSE_LEVELS];
    if (threadIdx.x < G0_MSE_LEVELS) s_cb0[threadIdx.x] = codebook_g0[threadIdx.x];
    if (threadIdx.x < G1_MSE_LEVELS) s_cb1[threadIdx.x] = codebook_g1[threadIdx.x];

    // Query in registers
    float qr0[EPT], qq0[EPT], qr1[EPT], qq1[EPT];
    int qb = batch_idx * num_qo + qoh;
    #pragma unroll
    for (int e = 0; e < EPT; e++) {
        int d = lane * EPT + e;
        qr0[e]=q_rot_g0[qb*G0_DIM+d]; qq0[e]=q_qjl_g0[qb*G0_DIM+d];
        qr1[e]=q_rot_g1[qb*G1_DIM+d]; qq1[e]=q_qjl_g1[qb*G1_DIM+d];
    }

    int pgs = kv_indptr[batch_idx], pge = kv_indptr[batch_idx+1];
    int seq_len = pge > pgs ? (pge-pgs-1)*page_size + kv_last_page_len[batch_idx] : 0;
    int cs = chunk_idx * chunk_size, ce = min(cs + chunk_size, seq_len);
    if (cs >= seq_len) {
        int oi = (chunk_idx * batch_size + batch_idx) * num_qo + qoh;
        if (lane == 0) { p_m[oi] = -FLT_MAX; p_d[oi] = 0.f; }
        return;
    }

    int s_page = 2 * page_size * num_kv * CACHE_DIM;
    int s_kv = page_size * num_kv * CACHE_DIM;
    int s_pos = num_kv * CACHE_DIM;

    float m_val = -FLT_MAX, d_val = 0.f;
    float am0[EPT]={}, aq0[EPT]={}, am1[EPT]={}, aq1[EPT]={};

    // ── Tiled loop ──
    for (int ts = cs; ts < ce; ts += TILE_N) {
        int tl = min(TILE_N, ce - ts);

        // Cooperative tile load
        __syncthreads();
        int total_words = tl * (CACHE_DIM / 4);
        for (int w = threadIdx.x; w < total_words; w += nthreads) {
            int pos = w / (CACHE_DIM / 4);
            int word = w % (CACHE_DIM / 4);
            int kv_pos = ts + pos;
            int pid = kv_indices[pgs + kv_pos / page_size];
            int base = pid * s_page + (kv_pos % page_size) * s_pos + kvh * CACHE_DIM;
            ((int32_t*)(tile_k + pos * CACHE_DIM))[word] = ((const int32_t*)(kv_data + base))[word];
            ((int32_t*)(tile_v + pos * CACHE_DIM))[word] = ((const int32_t*)(kv_data + base + s_kv))[word];
        }
        __syncthreads();

        // Process each position from shared memory
        for (int t = 0; t < tl; t++) {
            const uint8_t* k = tile_k + t * CACHE_DIM;
            const uint8_t* v = tile_v + t * CACHE_DIM;

            // Norms (broadcast — all threads read same addr, no bank conflict)
            float vn0 = rd16(k + G0_VNORM_OFF), rn0 = rd16(k + G0_RNORM_OFF);
            float vn1 = rd16(k + G1_VNORM_OFF), rn1 = rd16(k + G1_RNORM_OFF);

            // Batch unpack G0 MSE (3-bit: uint16 → 4 indices via PTX bfe)
            int fb0 = lane * EPT * 3;
            int by0 = fb0 >> 3, bo0 = fb0 & 7;
            uint32_t raw0 = (uint32_t)k[by0] | ((uint32_t)k[by0+1] << 8);
            int ki0=bfe(raw0,bo0,3), ki1=bfe(raw0,bo0+3,3);
            int ki2=bfe(raw0,bo0+6,3), ki3=bfe(raw0,bo0+9,3);

            // Batch unpack G0 QJL (1 byte → 4 signs)
            int qby0 = (lane*EPT)>>3, qbo0 = (lane*EPT)&7;
            uint8_t qr0b = k[G0_QJL_OFF + qby0];
            float s0=((qr0b>>(qbo0  ))&1)?1.f:-1.f, s1=((qr0b>>(qbo0+1))&1)?1.f:-1.f;
            float s2=((qr0b>>(qbo0+2))&1)?1.f:-1.f, s3=((qr0b>>(qbo0+3))&1)?1.f:-1.f;

            float partial = vn0 * (s_cb0[ki0]*qr0[0]
                                 + s_cb0[ki1]*qr0[1]
                                 + s_cb0[ki2]*qr0[2]
                                 + s_cb0[ki3]*qr0[3])
                          + vn0*rn0 * (s0*qq0[0] + s1*qq0[1] + s2*qq0[2] + s3*qq0[3]);

            // Batch unpack G1 MSE (2-bit: 1 byte → 4 indices via PTX bfe)
            uint32_t g1b = (uint32_t)k[G1_OFF + lane];
            int gi0=bfe(g1b,0,2), gi1=bfe(g1b,2,2), gi2=bfe(g1b,4,2), gi3=bfe(g1b,6,2);

            // G1 QJL
            int qby1 = (lane*EPT)>>3, qbo1 = (lane*EPT)&7;
            uint8_t qr1b = k[G1_QJL_OFF + qby1];
            float t0=((qr1b>>(qbo1  ))&1)?1.f:-1.f, t1=((qr1b>>(qbo1+1))&1)?1.f:-1.f;
            float t2=((qr1b>>(qbo1+2))&1)?1.f:-1.f, t3=((qr1b>>(qbo1+3))&1)?1.f:-1.f;

            partial += vn1 * (s_cb1[gi0]*qr1[0]
                            + s_cb1[gi1]*qr1[1]
                            + s_cb1[gi2]*qr1[2]
                            + s_cb1[gi3]*qr1[3])
                     + vn1*rn1 * (t0*qq1[0] + t1*qq1[1] + t2*qq1[2] + t3*qq1[3]);

            float score = warp_sum(partial) * sm_scale;

            // Online softmax
            float mp = m_val;
            m_val = fmaxf(m_val, score);
            float rs = expf(mp - m_val);
            d_val = d_val * rs + expf(score - m_val);
            float w = expf(score - m_val);
            #pragma unroll
            for (int e = 0; e < EPT; e++) {
                am0[e]*=rs; aq0[e]*=rs; am1[e]*=rs; aq1[e]*=rs;
            }

            // V accumulation (same unpack pattern)
            float wvn0 = w * rd16(v + G0_VNORM_OFF), wrn0 = wvn0 * rd16(v + G0_RNORM_OFF);
            uint32_t vraw0 = (uint32_t)v[by0] | ((uint32_t)v[by0+1] << 8);
            int vi0=bfe(vraw0,bo0,3), vi1=bfe(vraw0,bo0+3,3), vi2=bfe(vraw0,bo0+6,3), vi3=bfe(vraw0,bo0+9,3);
            uint8_t vq0b = v[G0_QJL_OFF + qby0];
            float vs0=((vq0b>>(qbo0  ))&1)?1.f:-1.f, vs1=((vq0b>>(qbo0+1))&1)?1.f:-1.f;
            float vs2=((vq0b>>(qbo0+2))&1)?1.f:-1.f, vs3=((vq0b>>(qbo0+3))&1)?1.f:-1.f;
            am0[0]+=wvn0*s_cb0[vi0];
            am0[1]+=wvn0*s_cb0[vi1];
            am0[2]+=wvn0*s_cb0[vi2];
            am0[3]+=wvn0*s_cb0[vi3];
            aq0[0]+=wrn0*vs0; aq0[1]+=wrn0*vs1; aq0[2]+=wrn0*vs2; aq0[3]+=wrn0*vs3;

            float wvn1 = w * rd16(v + G1_VNORM_OFF), wrn1 = wvn1 * rd16(v + G1_RNORM_OFF);
            uint32_t vg1b = (uint32_t)v[G1_OFF + lane];
            int vgi0=bfe(vg1b,0,2), vgi1=bfe(vg1b,2,2), vgi2=bfe(vg1b,4,2), vgi3=bfe(vg1b,6,2);
            uint8_t vq1b = v[G1_QJL_OFF + qby1];
            float vt0=((vq1b>>(qbo1  ))&1)?1.f:-1.f, vt1=((vq1b>>(qbo1+1))&1)?1.f:-1.f;
            float vt2=((vq1b>>(qbo1+2))&1)?1.f:-1.f, vt3=((vq1b>>(qbo1+3))&1)?1.f:-1.f;
            am1[0]+=wvn1*s_cb1[vgi0];
            am1[1]+=wvn1*s_cb1[vgi1];
            am1[2]+=wvn1*s_cb1[vgi2];
            am1[3]+=wvn1*s_cb1[vgi3];
            aq1[0]+=wrn1*vt0; aq1[1]+=wrn1*vt1; aq1[2]+=wrn1*vt2; aq1[3]+=wrn1*vt3;
        }
    }

    int oi = (chunk_idx * batch_size + batch_idx) * num_qo + qoh;
    int ob0 = oi * G0_DIM, ob1 = oi * G1_DIM;
    float inv = d_val > 0.f ? 1.f/d_val : 0.f;
    #pragma unroll
    for (int e = 0; e < EPT; e++) {
        int d = lane * EPT + e;
        p_mse_g0[ob0+d]=am0[e]*inv; p_qjl_g0[ob0+d]=aq0[e]*inv;
        p_mse_g1[ob1+d]=am1[e]*inv; p_qjl_g1[ob1+d]=aq1[e]*inv;
    }
    if (lane == 0) { p_m[oi] = m_val; p_d[oi] = d_val; }
}

// ─── Merge kernel ────────────────────────────────────────────────────────────

__global__ void tq_merge(
    const float* pm0, const float* pq0, const float* pm1, const float* pq1,
    const float* p_m, const float* p_d,
    float* om0, float* oq0, float* om1, float* oq1,
    int nc, int bs, int nqo
) {
    int bid = blockIdx.x, qoh = blockIdx.y, lane = threadIdx.x;
    float gm = -FLT_MAX;
    for (int c = 0; c < nc; c++) gm = fmaxf(gm, p_m[(c*bs+bid)*nqo+qoh]);
    float gd = 0.f;
    float sm0[EPT]={}, sq0[EPT]={}, sm1[EPT]={}, sq1[EPT]={};
    for (int c = 0; c < nc; c++) {
        int i = (c*bs+bid)*nqo+qoh;
        float rs = expf(p_m[i] - gm);
        gd += p_d[i] * rs;
        int b0=i*G0_DIM, b1=i*G1_DIM;
        #pragma unroll
        for (int e = 0; e < EPT; e++) {
            int d = lane*EPT+e;
            sm0[e]+=pm0[b0+d]*rs; sq0[e]+=pq0[b0+d]*rs;
            sm1[e]+=pm1[b1+d]*rs; sq1[e]+=pq1[b1+d]*rs;
        }
    }
    float inv = gd > 0.f ? 1.f/gd : 0.f;
    int o0=(bid*nqo+qoh)*G0_DIM, o1=(bid*nqo+qoh)*G1_DIM;
    #pragma unroll
    for (int e = 0; e < EPT; e++) {
        int d = lane*EPT+e;
        om0[o0+d]=sm0[e]*inv; oq0[o0+d]=sq0[e]*inv;
        om1[o1+d]=sm1[e]*inv; oq1[o1+d]=sq1[e]*inv;
    }
}

// ─── Python binding ──────────────────────────────────────────────────────────

std::vector<torch::Tensor> tq_fused_decode(
    torch::Tensor q_rot_g0, torch::Tensor q_qjl_g0,
    torch::Tensor q_rot_g1, torch::Tensor q_qjl_g1,
    torch::Tensor kv_data,
    torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor kv_last_page_len,
    torch::Tensor codebook_g0, torch::Tensor codebook_g1,
    int64_t page_size, int64_t num_qo, int64_t num_kv, int64_t cache_dim, double sm_scale
) {
    int bs = q_rot_g0.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(q_rot_g0.device());
    int seq_est = kv_last_page_len[0].item<int>() +
        (kv_indptr[1].item<int>() - kv_indptr[0].item<int>() - 1) * (int)page_size;
    int target = 64;
    int cs = max(TILE_N, ((seq_est+target-1)/target/TILE_N)*TILE_N);
    int nc = max(1, (seq_est+cs-1)/cs);

    auto pm0=torch::zeros({nc*bs*(int)num_qo,G0_DIM},opts);
    auto pq0=torch::zeros({nc*bs*(int)num_qo,G0_DIM},opts);
    auto pm1=torch::zeros({nc*bs*(int)num_qo,G1_DIM},opts);
    auto pq1=torch::zeros({nc*bs*(int)num_qo,G1_DIM},opts);
    auto p_m=torch::full({nc*bs*(int)num_qo},-FLT_MAX,opts);
    auto p_d=torch::zeros({nc*bs*(int)num_qo},opts);

    int nqpk=(int)num_qo/(int)num_kv;
    int smem=2*TILE_N*(int)cache_dim + (G0_MSE_LEVELS+G1_MSE_LEVELS)*sizeof(float);
    auto stream=at::cuda::getCurrentCUDAStream();

    tq_fused_v8<128><<<dim3(bs*nc,(int)num_kv),dim3(WARP_SIZE*nqpk),smem,stream>>>(
        q_rot_g0.data_ptr<float>(),q_qjl_g0.data_ptr<float>(),
        q_rot_g1.data_ptr<float>(),q_qjl_g1.data_ptr<float>(),
        kv_data.data_ptr<uint8_t>(),
        kv_indptr.data_ptr<int32_t>(),kv_indices.data_ptr<int32_t>(),
        kv_last_page_len.data_ptr<int32_t>(),
        codebook_g0.data_ptr<float>(),codebook_g1.data_ptr<float>(),
        pm0.data_ptr<float>(),pq0.data_ptr<float>(),
        pm1.data_ptr<float>(),pq1.data_ptr<float>(),
        p_m.data_ptr<float>(),p_d.data_ptr<float>(),
        (int)page_size,(int)num_qo,(int)num_kv,bs,cs,nc,(float)sm_scale);

    auto om0=torch::zeros({bs,(int)num_qo,G0_DIM},opts);
    auto oq0=torch::zeros({bs,(int)num_qo,G0_DIM},opts);
    auto om1=torch::zeros({bs,(int)num_qo,G1_DIM},opts);
    auto oq1=torch::zeros({bs,(int)num_qo,G1_DIM},opts);
    tq_merge<<<dim3(bs,(int)num_qo),WARP_SIZE,0,stream>>>(
        pm0.data_ptr<float>(),pq0.data_ptr<float>(),
        pm1.data_ptr<float>(),pq1.data_ptr<float>(),
        p_m.data_ptr<float>(),p_d.data_ptr<float>(),
        om0.data_ptr<float>(),oq0.data_ptr<float>(),
        om1.data_ptr<float>(),oq1.data_ptr<float>(),nc,bs,(int)num_qo);
    return {om0,oq0,om1,oq1};
}

PYBIND11_MODULE(tq_fused_decode_ext, m) {
    m.def("decode", &tq_fused_decode, "TQ Fused Decode v8");
}
