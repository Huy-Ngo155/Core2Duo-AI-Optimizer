#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <emmintrin.h>

#ifdef _MSC_VER
#include <malloc.h>
#include <intrin.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#define ALIGNED(x) __declspec(align(x))
#define PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#include <stdlib.h>
#include <x86intrin.h>
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#define ALIGNED(x) __attribute__((aligned(x)))
#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#endif

static inline void enable_daz_ftz() {
    unsigned int mxcsr = _mm_getcsr();
    mxcsr |= 0x8040;
    _mm_setcsr(mxcsr);
}

static inline __m128i sign_extend_epi8_to_epi16(__m128i a) {
    __m128i zero = _mm_setzero_si128();
    __m128i sign_mask = _mm_cmpgt_epi8(zero, a);
    __m128i high = _mm_unpacklo_epi8(a, sign_mask);
    __m128i low = _mm_unpackhi_epi8(a, sign_mask);
    return _mm_unpacklo_epi16(high, low);
}

static inline float horizontal_sum_sse(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

class Minillm {
private:
    struct Config {
        int vocab_size = 32000;
        int dim = 256;
        int hidden_dim = 512;
        int n_layers = 4;
        int n_heads = 8;
        int head_dim = 32;
        int seq_len = 512;
        float eps = 1e-6f;
        float rope_theta = 10000.0f;
    };

    Config config;
    
    struct TensorFp32 {
        float* data;
        int rows, cols;
        
        TensorFp32(int r, int c) : rows(r), cols(c) {
            data = (float*)ALIGNED_ALLOC(32, r * c * sizeof(float));
            memset(data, 0, r * c * sizeof(float));
        }
        
        ~TensorFp32() {
            if (data) ALIGNED_FREE(data);
        }
        
        const float* row_ptr(int i) const {
            return data + i * cols;
        }
    };
    
    struct TensorInt8 {
        int8_t* data;
        float* scales;
        int rows, cols;
        int stride;
        
        TensorInt8(int r, int c) : rows(r), cols(c), stride(((c + 63) & ~63)) {
            data = (int8_t*)ALIGNED_ALLOC(32, r * stride * sizeof(int8_t));
            scales = (float*)ALIGNED_ALLOC(32, r * sizeof(float));
        }
        
        ~TensorInt8() {
            if (data) ALIGNED_FREE(data);
            if (scales) ALIGNED_FREE(scales);
        }
        
        const int8_t* row_ptr(int i) const {
            return data + i * stride;
        }
        
        const float get_scale(int i) const {
            return scales[i];
        }
    };
    
    struct LayerWeights {
        TensorInt8 q_proj, k_proj, v_proj, o_proj;
        TensorInt8 gate_proj, up_proj, down_proj;
        TensorFp32 ln1_weight;
        TensorFp32 ln2_weight;
        
        LayerWeights(int dim, int hidden_dim) 
            : q_proj(dim, dim), k_proj(dim, dim), v_proj(dim, dim),
              o_proj(dim, dim), gate_proj(dim, hidden_dim),
              up_proj(dim, hidden_dim), down_proj(hidden_dim, dim),
              ln1_weight(1, dim), ln2_weight(1, dim) {
            for (int i = 0; i < dim; i++) {
                ln1_weight.data[i] = 1.0f;
                ln2_weight.data[i] = 1.0f;
            }
        }
    };
    
    TensorInt8 token_embedding;
    std::vector<LayerWeights> layers;
    TensorInt8 lm_head;
    
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> inv_vocab;
    
    struct KVCacheRing {
        float* k;
        float* v;
        int capacity;
        int start;
        int size;
        int n_heads;
        int head_dim;
        
        KVCacheRing(int cap, int n_h, int h_dim) 
            : capacity(cap), start(0), size(0), n_heads(n_h), head_dim(h_dim) {
            k = (float*)ALIGNED_ALLOC(32, cap * n_h * h_dim * sizeof(float));
            v = (float*)ALIGNED_ALLOC(32, cap * n_h * h_dim * sizeof(float));
            memset(k, 0, cap * n_h * h_dim * sizeof(float));
            memset(v, 0, cap * n_h * h_dim * sizeof(float));
        }
        
        ~KVCacheRing() {
            if (k) ALIGNED_FREE(k);
            if (v) ALIGNED_FREE(v);
        }
        
        void add(const float* new_k, const float* new_v) {
            int pos = (start + size) % capacity;
            int base_offset = pos * n_heads * head_dim;
            
            memcpy(k + base_offset, new_k, n_heads * head_dim * sizeof(float));
            memcpy(v + base_offset, new_v, n_heads * head_dim * sizeof(float));
            
            if (size < capacity) {
                size++;
            } else {
                start = (start + 1) % capacity;
            }
        }
        
        const float* get_k_ptr(int seq_pos, int head_idx) const {
            int actual_pos = (start + seq_pos) % capacity;
            return k + actual_pos * n_heads * head_dim + head_idx * head_dim;
        }
        
        const float* get_v_ptr(int seq_pos, int head_idx) const {
            int actual_pos = (start + seq_pos) % capacity;
            return v + actual_pos * n_heads * head_dim + head_idx * head_dim;
        }
        
        void reset() {
            start = 0;
            size = 0;
            memset(k, 0, capacity * n_heads * head_dim * sizeof(float));
            memset(v, 0, capacity * n_heads * head_dim * sizeof(float));
        }
        
        int get_seq_len() const { return size; }
    };
    
    std::vector<KVCacheRing> kv_caches;
    
    struct RopeCacheEntry {
        float cos;
        float sin;
        float neg_sin;
        float cos_again;
    };
    std::vector<RopeCacheEntry> rope_cache;
    
    float* scores_buffer;
    
    static inline __m128 polynomial_sigmoid_sse(__m128 x) {
        const __m128 zero = _mm_setzero_ps();
        const __m128 one = _mm_set1_ps(1.0f);
        
        const __m128 c1 = _mm_set1_ps(0.0705230784f);
        const __m128 c2 = _mm_set1_ps(0.0422820123f);
        const __m128 c3 = _mm_set1_ps(0.0092705272f);
        const __m128 c4 = _mm_set1_ps(0.0001520143f);
        const __m128 c5 = _mm_set1_ps(0.0002765672f);
        const __m128 c6 = _mm_set1_ps(0.0000430638f);
        
        __m128 abs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
        __m128 sign = _mm_and_ps(x, _mm_set1_ps(-0.0f));
        
        __m128 z = _mm_mul_ps(abs_x, _mm_set1_ps(0.2316419f));
        z = _mm_add_ps(z, one);
        
        __m128 z2 = _mm_mul_ps(z, z);
        __m128 z3 = _mm_mul_ps(z2, z);
        __m128 z4 = _mm_mul_ps(z2, z2);
        __m128 z5 = _mm_mul_ps(z3, z2);
        
        __m128 t = _mm_add_ps(c1, _mm_mul_ps(c2, z));
        t = _mm_add_ps(t, _mm_mul_ps(c3, z2));
        t = _mm_add_ps(t, _mm_mul_ps(c4, z3));
        t = _mm_add_ps(t, _mm_mul_ps(c5, z4));
        t = _mm_add_ps(t, _mm_mul_ps(c6, z5));
        
        __m128 inv = _mm_rcp_ps(t);
        inv = _mm_mul_ps(inv, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(t, inv)));
        
        __m128 cdf = _mm_sub_ps(one, inv);
        
        __m128 mask = _mm_cmpgt_ps(x, zero);
        __m128 result = _mm_or_ps(_mm_and_ps(mask, cdf), 
                                 _mm_andnot_ps(mask, _mm_sub_ps(one, cdf)));
        
        return _mm_mul_ps(x, result);
    }
    
    static inline void rms_norm_aligned(const float* x, const float* weight, int dim, float eps, float* out) {
        __m128 sum_sse = _mm_setzero_ps();
        int i = 0;
        
        for (; i + 31 < dim; i += 32) {
            __m128 x0 = _mm_load_ps(x + i);
            __m128 x1 = _mm_load_ps(x + i + 4);
            __m128 x2 = _mm_load_ps(x + i + 8);
            __m128 x3 = _mm_load_ps(x + i + 12);
            __m128 x4 = _mm_load_ps(x + i + 16);
            __m128 x5 = _mm_load_ps(x + i + 20);
            __m128 x6 = _mm_load_ps(x + i + 24);
            __m128 x7 = _mm_load_ps(x + i + 28);
            
            sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x0, x0));
            sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x1, x1));
            sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x2, x2));
            sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x3, x3));
            sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x4, x4));
            sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x5, x5));
            sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x6, x6));
            sum_sse = _mm_add_ps(sum_sse, _mm_mul_ps(x7, x7));
        }
        
        float sum = horizontal_sum_sse(sum_sse);
        
        for (; i < dim; i++) {
            sum += x[i] * x[i];
        }
        
        float rms = sqrtf(sum / dim + eps);
        __m128 scale_vec = _mm_set1_ps(1.0f / rms);
        
        i = 0;
        for (; i + 31 < dim; i += 32) {
            __m128 x0 = _mm_load_ps(x + i);
            __m128 x1 = _mm_load_ps(x + i + 4);
            __m128 x2 = _mm_load_ps(x + i + 8);
            __m128 x3 = _mm_load_ps(x + i + 12);
            __m128 x4 = _mm_load_ps(x + i + 16);
            __m128 x5 = _mm_load_ps(x + i + 20);
            __m128 x6 = _mm_load_ps(x + i + 24);
            __m128 x7 = _mm_load_ps(x + i + 28);
            
            __m128 w0 = _mm_load_ps(weight + i);
            __m128 w1 = _mm_load_ps(weight + i + 4);
            __m128 w2 = _mm_load_ps(weight + i + 8);
            __m128 w3 = _mm_load_ps(weight + i + 12);
            __m128 w4 = _mm_load_ps(weight + i + 16);
            __m128 w5 = _mm_load_ps(weight + i + 20);
            __m128 w6 = _mm_load_ps(weight + i + 24);
            __m128 w7 = _mm_load_ps(weight + i + 28);
            
            __m128 r0 = _mm_mul_ps(_mm_mul_ps(x0, scale_vec), w0);
            __m128 r1 = _mm_mul_ps(_mm_mul_ps(x1, scale_vec), w1);
            __m128 r2 = _mm_mul_ps(_mm_mul_ps(x2, scale_vec), w2);
            __m128 r3 = _mm_mul_ps(_mm_mul_ps(x3, scale_vec), w3);
            __m128 r4 = _mm_mul_ps(_mm_mul_ps(x4, scale_vec), w4);
            __m128 r5 = _mm_mul_ps(_mm_mul_ps(x5, scale_vec), w5);
            __m128 r6 = _mm_mul_ps(_mm_mul_ps(x6, scale_vec), w6);
            __m128 r7 = _mm_mul_ps(_mm_mul_ps(x7, scale_vec), w7);
            
            _mm_store_ps(out + i, r0);
            _mm_store_ps(out + i + 4, r1);
            _mm_store_ps(out + i + 8, r2);
            _mm_store_ps(out + i + 12, r3);
            _mm_store_ps(out + i + 16, r4);
            _mm_store_ps(out + i + 20, r5);
            _mm_store_ps(out + i + 24, r6);
            _mm_store_ps(out + i + 28, r7);
        }
        
        for (; i < dim; i++) {
            out[i] = x[i] * (1.0f / rms) * weight[i];
        }
    }
    
    static inline void matmul_vec_int8_8x4_block(const TensorInt8& a, const float* vec, float* out) {
        int rows = a.rows;
        int cols = a.cols;
        int stride = a.stride;
        const int8_t* a_data = a.data;
        const float* scales = a.scales;
        
        for (int i = 0; i < rows; i += 8) {
            const int8_t* rows_ptr[8];
            float outputs[8] = {0};
            
            for (int k = 0; k < 8 && (i + k) < rows; k++) {
                rows_ptr[k] = a_data + (i + k) * stride;
            }
            
            int j = 0;
            for (; j + 127 < cols; j += 128) {
                for (int k = 0; k < 8 && (i + k) < rows; k++) {
                    __m128i sum0 = _mm_setzero_si128();
                    __m128i sum1 = _mm_setzero_si128();
                    __m128i sum2 = _mm_setzero_si128();
                    __m128i sum3 = _mm_setzero_si128();
                    
                    const int8_t* row = rows_ptr[k];
                    
                    for (int block = 0; block < 4; block++) {
                        int offset = j + block * 32;
                        
                        __m128i row0 = _mm_load_si128((const __m128i*)(row + offset));
                        __m128i row1 = _mm_load_si128((const __m128i*)(row + offset + 16));
                        
                        if (block == 0 && offset == 0) {
                            PREFETCH(row + offset + 256);
                        }
                        
                        __m128i row0_lo = sign_extend_epi8_to_epi16(row0);
                        __m128i row0_hi = sign_extend_epi8_to_epi16(_mm_srli_si128(row0, 8));
                        __m128i row1_lo = sign_extend_epi8_to_epi16(row1);
                        __m128i row1_hi = sign_extend_epi8_to_epi16(_mm_srli_si128(row1, 8));
                        
                        __m128 vec0 = _mm_load_ps(vec + offset);
                        __m128 vec1 = _mm_load_ps(vec + offset + 4);
                        __m128 vec2 = _mm_load_ps(vec + offset + 8);
                        __m128 vec3 = _mm_load_ps(vec + offset + 12);
                        __m128 vec4 = _mm_load_ps(vec + offset + 16);
                        __m128 vec5 = _mm_load_ps(vec + offset + 20);
                        __m128 vec6 = _mm_load_ps(vec + offset + 24);
                        __m128 vec7 = _mm_load_ps(vec + offset + 28);
                        
                        __m128i vec0_int = _mm_cvtps_epi32(_mm_mul_ps(vec0, _mm_set1_ps(256.0f)));
                        __m128i vec1_int = _mm_cvtps_epi32(_mm_mul_ps(vec1, _mm_set1_ps(256.0f)));
                        __m128i vec2_int = _mm_cvtps_epi32(_mm_mul_ps(vec2, _mm_set1_ps(256.0f)));
                        __m128i vec3_int = _mm_cvtps_epi32(_mm_mul_ps(vec3, _mm_set1_ps(256.0f)));
                        __m128i vec4_int = _mm_cvtps_epi32(_mm_mul_ps(vec4, _mm_set1_ps(256.0f)));
                        __m128i vec5_int = _mm_cvtps_epi32(_mm_mul_ps(vec5, _mm_set1_ps(256.0f)));
                        __m128i vec6_int = _mm_cvtps_epi32(_mm_mul_ps(vec6, _mm_set1_ps(256.0f)));
                        __m128i vec7_int = _mm_cvtps_epi32(_mm_mul_ps(vec7, _mm_set1_ps(256.0f)));
                        
                        __m128i vec0_16 = _mm_packs_epi32(vec0_int, vec1_int);
                        __m128i vec1_16 = _mm_packs_epi32(vec2_int, vec3_int);
                        __m128i vec2_16 = _mm_packs_epi32(vec4_int, vec5_int);
                        __m128i vec3_16 = _mm_packs_epi32(vec6_int, vec7_int);
                        
                        sum0 = _mm_add_epi32(sum0, _mm_madd_epi16(row0_lo, vec0_16));
                        sum1 = _mm_add_epi32(sum1, _mm_madd_epi16(row0_hi, vec1_16));
                        sum2 = _mm_add_epi32(sum2, _mm_madd_epi16(row1_lo, vec2_16));
                        sum3 = _mm_add_epi32(sum3, _mm_madd_epi16(row1_hi, vec3_16));
                    }
                    
                    int32_t* sum0_ptr = (int32_t*)&sum0;
                    int32_t* sum1_ptr = (int32_t*)&sum1;
                    int32_t* sum2_ptr = (int32_t*)&sum2;
                    int32_t* sum3_ptr = (int32_t*)&sum3;
                    
                    outputs[k] += sum0_ptr[0] + sum0_ptr[1] + sum0_ptr[2] + sum0_ptr[3] +
                                 sum1_ptr[0] + sum1_ptr[1] + sum1_ptr[2] + sum1_ptr[3] +
                                 sum2_ptr[0] + sum2_ptr[1] + sum2_ptr[2] + sum2_ptr[3] +
                                 sum3_ptr[0] + sum3_ptr[1] + sum3_ptr[2] + sum3_ptr[3];
                }
            }
            
            for (int k = 0; k < 8 && (i + k) < rows; k++) {
                float scale = scales[i + k];
                const int8_t* row = rows_ptr[k];
                
                float final_sum = outputs[k];
                
                for (; j < cols; j++) {
                    final_sum += row[j] * vec[j];
                }
                
                out[i + k] = final_sum * scale / 256.0f;
            }
        }
    }
    
    static inline void load_embedding_int8_sse(const int8_t* src, float scale, float* dst, int dim) {
        __m128 scale_vec = _mm_set1_ps(scale);
        int i = 0;
        
        for (; i + 31 < dim; i += 32) {
            __m128i data0 = _mm_load_si128((const __m128i*)(src + i));
            __m128i data1 = _mm_load_si128((const __m128i*)(src + i + 16));
            
            __m128i lo0 = sign_extend_epi8_to_epi16(data0);
            __m128i hi0 = sign_extend_epi8_to_epi16(_mm_srli_si128(data0, 8));
            __m128i lo1 = sign_extend_epi8_to_epi16(data1);
            __m128i hi1 = sign_extend_epi8_to_epi16(_mm_srli_si128(data1, 8));
            
            __m128 f0 = _mm_cvtepi32_ps(lo0);
            __m128 f1 = _mm_cvtepi32_ps(hi0);
            __m128 f2 = _mm_cvtepi32_ps(lo1);
            __m128 f3 = _mm_cvtepi32_ps(hi1);
            
            f0 = _mm_mul_ps(f0, scale_vec);
            f1 = _mm_mul_ps(f1, scale_vec);
            f2 = _mm_mul_ps(f2, scale_vec);
            f3 = _mm_mul_ps(f3, scale_vec);
            
            _mm_store_ps(dst + i, f0);
            _mm_store_ps(dst + i + 4, f1);
            _mm_store_ps(dst + i + 8, f2);
            _mm_store_ps(dst + i + 12, f3);
            
            PREFETCH(src + i + 128);
        }
        
        for (; i < dim; i++) {
            dst[i] = src[i] * scale;
        }
    }
    
    void apply_rope_simplified(float* q, float* k, int pos, int head_idx) {
        int head_dim = config.head_dim;
        int rope_dim = head_dim / 2;
        
        for (int i = 0; i < rope_dim; i += 2) {
            const RopeCacheEntry& entry = rope_cache[pos * rope_dim / 2 + i / 2];
            
            int q_base = head_idx * head_dim + i * 2;
            int k_base = head_idx * head_dim + i * 2;
            
            float q0 = q[q_base];
            float q1 = q[q_base + 1];
            float k0 = k[k_base];
            float k1 = k[k_base + 1];
            
            q[q_base] = q0 * entry.cos + q1 * entry.neg_sin;
            q[q_base + 1] = q0 * entry.sin + q1 * entry.cos_again;
            k[k_base] = k0 * entry.cos + k1 * entry.neg_sin;
            k[k_base + 1] = k0 * entry.sin + k1 * entry.cos_again;
        }
    }
    
    void attention_head_causal_optimized(const float* q_head, const KVCacheRing& kv_cache, 
                                        int head_idx, int seq_len, float* out) {
        const int head_dim = config.head_dim;
        float* scores = scores_buffer;
        float max_score = -1e9f;
        
        for (int j = 0; j < seq_len - 1; j++) {
            const float* k_vec = kv_cache.get_k_ptr(j, head_idx);
            
            __m128 sum0 = _mm_setzero_ps();
            __m128 sum1 = _mm_setzero_ps();
            __m128 sum2 = _mm_setzero_ps();
            __m128 sum3 = _mm_setzero_ps();
            __m128 sum4 = _mm_setzero_ps();
            __m128 sum5 = _mm_setzero_ps();
            __m128 sum6 = _mm_setzero_ps();
            __m128 sum7 = _mm_setzero_ps();
            
            int d = 0;
            
            PREFETCH(k_vec);
            
            for (; d + 63 < head_dim; d += 64) {
                __m128 q0 = _mm_load_ps(q_head + d);
                __m128 q1 = _mm_load_ps(q_head + d + 4);
                __m128 q2 = _mm_load_ps(q_head + d + 8);
                __m128 q3 = _mm_load_ps(q_head + d + 12);
                __m128 q4 = _mm_load_ps(q_head + d + 16);
                __m128 q5 = _mm_load_ps(q_head + d + 20);
                __m128 q6 = _mm_load_ps(q_head + d + 24);
                __m128 q7 = _mm_load_ps(q_head + d + 28);
                __m128 q8 = _mm_load_ps(q_head + d + 32);
                __m128 q9 = _mm_load_ps(q_head + d + 36);
                __m128 q10 = _mm_load_ps(q_head + d + 40);
                __m128 q11 = _mm_load_ps(q_head + d + 44);
                __m128 q12 = _mm_load_ps(q_head + d + 48);
                __m128 q13 = _mm_load_ps(q_head + d + 52);
                __m128 q14 = _mm_load_ps(q_head + d + 56);
                __m128 q15 = _mm_load_ps(q_head + d + 60);
                
                __m128 k0 = _mm_load_ps(k_vec + d);
                __m128 k1 = _mm_load_ps(k_vec + d + 4);
                __m128 k2 = _mm_load_ps(k_vec + d + 8);
                __m128 k3 = _mm_load_ps(k_vec + d + 12);
                __m128 k4 = _mm_load_ps(k_vec + d + 16);
                __m128 k5 = _mm_load_ps(k_vec + d + 20);
                __m128 k6 = _mm_load_ps(k_vec + d + 24);
                __m128 k7 = _mm_load_ps(k_vec + d + 28);
                __m128 k8 = _mm_load_ps(k_vec + d + 32);
                __m128 k9 = _mm_load_ps(k_vec + d + 36);
                __m128 k10 = _mm_load_ps(k_vec + d + 40);
                __m128 k11 = _mm_load_ps(k_vec + d + 44);
                __m128 k12 = _mm_load_ps(k_vec + d + 48);
                __m128 k13 = _mm_load_ps(k_vec + d + 52);
                __m128 k14 = _mm_load_ps(k_vec + d + 56);
                __m128 k15 = _mm_load_ps(k_vec + d + 60);
                
                PREFETCH(k_vec + d + 128);
                
                sum0 = _mm_add_ps(sum0, _mm_mul_ps(q0, k0));
                sum1 = _mm_add_ps(sum1, _mm_mul_ps(q1, k1));
                sum2 = _mm_add_ps(sum2, _mm_mul_ps(q2, k2));
                sum3 = _mm_add_ps(sum3, _mm_mul_ps(q3, k3));
                sum4 = _mm_add_ps(sum4, _mm_mul_ps(q4, k4));
                sum5 = _mm_add_ps(sum5, _mm_mul_ps(q5, k5));
                sum6 = _mm_add_ps(sum6, _mm_mul_ps(q6, k6));
                sum7 = _mm_add_ps(sum7, _mm_mul_ps(q7, k7));
                
                sum0 = _mm_add_ps(sum0, _mm_mul_ps(q8, k8));
                sum1 = _mm_add_ps(sum1, _mm_mul_ps(q9, k9));
                sum2 = _mm_add_ps(sum2, _mm_mul_ps(q10, k10));
                sum3 = _mm_add_ps(sum3, _mm_mul_ps(q11, k11));
                sum4 = _mm_add_ps(sum4, _mm_mul_ps(q12, k12));
                sum5 = _mm_add_ps(sum5, _mm_mul_ps(q13, k13));
                sum6 = _mm_add_ps(sum6, _mm_mul_ps(q14, k14));
                sum7 = _mm_add_ps(sum7, _mm_mul_ps(q15, k15));
            }
            
            sum0 = _mm_add_ps(sum0, sum1);
            sum2 = _mm_add_ps(sum2, sum3);
            sum4 = _mm_add_ps(sum4, sum5);
            sum6 = _mm_add_ps(sum6, sum7);
            
            sum0 = _mm_add_ps(sum0, sum2);
            sum4 = _mm_add_ps(sum4, sum6);
            sum0 = _mm_add_ps(sum0, sum4);
            
            float sum = horizontal_sum_sse(sum0);
            
            for (; d < head_dim; d++) {
                sum += q_head[d] * k_vec[d];
            }
            
            float score = sum * (1.0f / sqrtf((float)head_dim));
            scores[j] = score;
            
            if (score > max_score) {
                max_score = score;
            }
        }
        
        scores[seq_len - 1] = -1e9f;
        
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float val = scores[j] - max_score;
            if (val < -16.0f) scores[j] = 0.0f;
            else if (val > 16.0f) scores[j] = 1e9f;
            else {
                val = 1.0f + val / 256.0f;
                val *= val; val *= val; val *= val; val *= val;
                val *= val; val *= val; val *= val; val *= val;
                scores[j] = val;
            }
            sum_exp += scores[j];
        }
        
        float inv_sum_exp = 1.0f / sum_exp;
        
        memset(out, 0, head_dim * sizeof(float));
        
        for (int d = 0; d < head_dim; d += 8) {
            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            
            int j = 0;
            for (; j + 7 < seq_len; j += 8) {
                __m128 s0 = _mm_set1_ps(scores[j] * inv_sum_exp);
                __m128 s1 = _mm_set1_ps(scores[j + 1] * inv_sum_exp);
                __m128 s2 = _mm_set1_ps(scores[j + 2] * inv_sum_exp);
                __m128 s3 = _mm_set1_ps(scores[j + 3] * inv_sum_exp);
                __m128 s4 = _mm_set1_ps(scores[j + 4] * inv_sum_exp);
                __m128 s5 = _mm_set1_ps(scores[j + 5] * inv_sum_exp);
                __m128 s6 = _mm_set1_ps(scores[j + 6] * inv_sum_exp);
                __m128 s7 = _mm_set1_ps(scores[j + 7] * inv_sum_exp);
                
                __m128 v0 = _mm_load_ps(kv_cache.get_v_ptr(j, head_idx) + d);
                __m128 v1 = _mm_load_ps(kv_cache.get_v_ptr(j + 1, head_idx) + d);
                __m128 v2 = _mm_load_ps(kv_cache.get_v_ptr(j + 2, head_idx) + d);
                __m128 v3 = _mm_load_ps(kv_cache.get_v_ptr(j + 3, head_idx) + d);
                __m128 v4 = _mm_load_ps(kv_cache.get_v_ptr(j + 4, head_idx) + d);
                __m128 v5 = _mm_load_ps(kv_cache.get_v_ptr(j + 5, head_idx) + d);
                __m128 v6 = _mm_load_ps(kv_cache.get_v_ptr(j + 6, head_idx) + d);
                __m128 v7 = _mm_load_ps(kv_cache.get_v_ptr(j + 7, head_idx) + d);
                
                PREFETCH(kv_cache.get_v_ptr(j + 16, head_idx) + d);
                
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(s0, v0));
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(s1, v1));
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(s2, v2));
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(s3, v3));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(s4, v4));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(s5, v5));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(s6, v6));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(s7, v7));
            }
            
            acc0 = _mm_add_ps(acc0, acc1);
            
            for (; j < seq_len; j++) {
                __m128 s = _mm_set1_ps(scores[j] * inv_sum_exp);
                __m128 v = _mm_load_ps(kv_cache.get_v_ptr(j, head_idx) + d);
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(s, v));
            }
            
            _mm_store_ps(out + d, acc0);
            if (d + 4 < head_dim) {
                _mm_store_ps(out + d + 4, _mm_setzero_ps());
            }
        }
    }
    
    static inline void silu_polynomial(float* x, int n) {
        int i = 0;
        
        for (; i + 31 < n; i += 32) {
            __m128 x0 = _mm_load_ps(x + i);
            __m128 x1 = _mm_load_ps(x + i + 4);
            __m128 x2 = _mm_load_ps(x + i + 8);
            __m128 x3 = _mm_load_ps(x + i + 12);
            __m128 x4 = _mm_load_ps(x + i + 16);
            __m128 x5 = _mm_load_ps(x + i + 20);
            __m128 x6 = _mm_load_ps(x + i + 24);
            __m128 x7 = _mm_load_ps(x + i + 28);
            
            __m128 s0 = polynomial_sigmoid_sse(x0);
            __m128 s1 = polynomial_sigmoid_sse(x1);
            __m128 s2 = polynomial_sigmoid_sse(x2);
            __m128 s3 = polynomial_sigmoid_sse(x3);
            __m128 s4 = polynomial_sigmoid_sse(x4);
            __m128 s5 = polynomial_sigmoid_sse(x5);
            __m128 s6 = polynomial_sigmoid_sse(x6);
            __m128 s7 = polynomial_sigmoid_sse(x7);
            
            x0 = _mm_mul_ps(x0, s0);
            x1 = _mm_mul_ps(x1, s1);
            x2 = _mm_mul_ps(x2, s2);
            x3 = _mm_mul_ps(x3, s3);
            x4 = _mm_mul_ps(x4, s4);
            x5 = _mm_mul_ps(x5, s5);
            x6 = _mm_mul_ps(x6, s6);
            x7 = _mm_mul_ps(x7, s7);
            
            _mm_store_ps(x + i, x0);
            _mm_store_ps(x + i + 4, x1);
            _mm_store_ps(x + i + 8, x2);
            _mm_store_ps(x + i + 12, x3);
            _mm_store_ps(x + i + 16, x4);
            _mm_store_ps(x + i + 20, x5);
            _mm_store_ps(x + i + 24, x6);
            _mm_store_ps(x + i + 28, x7);
            
            PREFETCH(x + i + 128);
        }
        
        for (; i < n; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-x[i]));
            x[i] = x[i] * sigmoid;
        }
    }
    
    void quantize_tensor(const float* src, int8_t* dst, float* scales, int rows, int cols, int stride) {
        for (int i = 0; i < rows; i++) {
            const float* row_src = src + i * cols;
            int8_t* row_dst = dst + i * stride;
            
            float max_val = 0.0f;
            for (int j = 0; j < cols; j++) {
                float abs_val = fabsf(row_src[j]);
                if (abs_val > max_val) max_val = abs_val;
            }
            
            float scale = max_val / 127.0f;
            scales[i] = scale;
            
            float inv_scale = 127.0f / max_val;
            for (int j = 0; j < cols; j++) {
                row_dst[j] = (int8_t)(row_src[j] * inv_scale + 0.5f);
            }
        }
    }
    
public:
    Minillm() 
        : config(),
          token_embedding(config.vocab_size, config.dim),
          lm_head(config.dim, config.vocab_size) {
        
        scores_buffer = (float*)ALIGNED_ALLOC(32, config.seq_len * sizeof(float));
        
        int rope_dim = config.head_dim / 2;
        rope_cache.resize(config.seq_len * rope_dim / 2);
        for (int pos = 0; pos < config.seq_len; pos++) {
            for (int i = 0; i < rope_dim; i += 2) {
                float inv_freq = 1.0f / powf(config.rope_theta, (float)i / rope_dim);
                float angle = pos * inv_freq;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                rope_cache[pos * rope_dim / 2 + i / 2] = 
                    {cos_val, sin_val, -sin_val, cos_val};
            }
        }
        
        layers.reserve(config.n_layers);
        for (int i = 0; i < config.n_layers; i++) {
            layers.emplace_back(config.dim, config.hidden_dim);
        }
        
        kv_caches.reserve(config.n_layers);
        for (int i = 0; i < config.n_layers; i++) {
            kv_caches.emplace_back(config.seq_len, config.n_heads, config.head_dim);
        }
        
        for (int i = 0; i < config.vocab_size; i++) {
            std::string token = "t" + std::to_string(i);
            vocab[token] = i;
            inv_vocab.push_back(token);
        }
        
        std::srand(42);
        
        float* temp = new float[config.vocab_size * config.dim];
        for (int i = 0; i < config.vocab_size * config.dim; i++) {
            temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
        }
        quantize_tensor(temp, token_embedding.data, token_embedding.scales,
                       config.vocab_size, config.dim, token_embedding.stride);
        delete[] temp;
        
        for (auto& layer : layers) {
            int size;
            
            size = config.dim * config.dim;
            temp = new float[size];
            for (int i = 0; i < size; i++) temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
            quantize_tensor(temp, layer.q_proj.data, layer.q_proj.scales,
                          config.dim, config.dim, layer.q_proj.stride);
            delete[] temp;
            
            size = config.dim * config.dim;
            temp = new float[size];
            for (int i = 0; i < size; i++) temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
            quantize_tensor(temp, layer.k_proj.data, layer.k_proj.scales,
                          config.dim, config.dim, layer.k_proj.stride);
            delete[] temp;
            
            size = config.dim * config.dim;
            temp = new float[size];
            for (int i = 0; i < size; i++) temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
            quantize_tensor(temp, layer.v_proj.data, layer.v_proj.scales,
                          config.dim, config.dim, layer.v_proj.stride);
            delete[] temp;
            
            size = config.dim * config.dim;
            temp = new float[size];
            for (int i = 0; i < size; i++) temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
            quantize_tensor(temp, layer.o_proj.data, layer.o_proj.scales,
                          config.dim, config.dim, layer.o_proj.stride);
            delete[] temp;
            
            size = config.dim * config.hidden_dim;
            temp = new float[size];
            for (int i = 0; i < size; i++) temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
            quantize_tensor(temp, layer.gate_proj.data, layer.gate_proj.scales,
                          config.dim, config.hidden_dim, layer.gate_proj.stride);
            delete[] temp;
            
            size = config.dim * config.hidden_dim;
            temp = new float[size];
            for (int i = 0; i < size; i++) temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
            quantize_tensor(temp, layer.up_proj.data, layer.up_proj.scales,
                          config.dim, config.hidden_dim, layer.up_proj.stride);
            delete[] temp;
            
            size = config.hidden_dim * config.dim;
            temp = new float[size];
            for (int i = 0; i < size; i++) temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
            quantize_tensor(temp, layer.down_proj.data, layer.down_proj.scales,
                          config.hidden_dim, config.dim, layer.down_proj.stride);
            delete[] temp;
        }
        
        temp = new float[config.dim * config.vocab_size];
        for (int i = 0; i < config.dim * config.vocab_size; i++) {
            temp[i] = (std::rand() / float(RAND_MAX) - 0.5f) * 0.02f;
        }
        quantize_tensor(temp, lm_head.data, lm_head.scales,
                       config.dim, config.vocab_size, lm_head.stride);
        delete[] temp;
    }
    
    ~Minillm() {
        if (scores_buffer) ALIGNED_FREE(scores_buffer);
    }
    
    std::vector<float> forward_step(int token_id, int pos) {
        ALIGNED(32) float hidden[256];
        
        load_embedding_int8_sse(token_embedding.row_ptr(token_id), 
                               token_embedding.get_scale(token_id), 
                               hidden, config.dim);
        
        for (int l = 0; l < config.n_layers; l++) {
            ALIGNED(32) float norm_buf[256];
            ALIGNED(32) float q_buf[256];
            ALIGNED(32) float k_buf[256];
            ALIGNED(32) float v_buf[256];
            ALIGNED(32) float attn_out_buf[256];
            ALIGNED(32) float proj_buf[256];
            ALIGNED(32) float gate_buf[512];
            ALIGNED(32) float up_buf[512];
            ALIGNED(32) float silu_buf[512];
            ALIGNED(32) float down_buf[256];
            
            rms_norm_aligned(hidden, layers[l].ln1_weight.data, config.dim, config.eps, norm_buf);
            
            matmul_vec_int8_8x4_block(layers[l].q_proj, norm_buf, q_buf);
            matmul_vec_int8_8x4_block(layers[l].k_proj, norm_buf, k_buf);
            matmul_vec_int8_8x4_block(layers[l].v_proj, norm_buf, v_buf);
            
            for (int hd = 0; hd < config.n_heads; hd++) {
                apply_rope_simplified(q_buf, k_buf, pos, hd);
            }
            
            kv_caches[l].add(k_buf, v_buf);
            
            int seq_len = kv_caches[l].get_seq_len();
            
            for (int hd = 0; hd < config.n_heads; hd++) {
                float* q_head = q_buf + hd * config.head_dim;
                float* attn_head_out = attn_out_buf + hd * config.head_dim;
                
                attention_head_causal_optimized(q_head, kv_caches[l], hd, seq_len, attn_head_out);
            }
            
            matmul_vec_int8_8x4_block(layers[l].o_proj, attn_out_buf, proj_buf);
            
            __m128* h_vec = (__m128*)hidden;
            __m128* p_vec = (__m128*)proj_buf;
            for (int i = 0; i < config.dim / 4; i++) {
                h_vec[i] = _mm_add_ps(h_vec[i], p_vec[i]);
            }
            
            rms_norm_aligned(hidden, layers[l].ln2_weight.data, config.dim, config.eps, norm_buf);
            
            matmul_vec_int8_8x4_block(layers[l].gate_proj, norm_buf, gate_buf);
            matmul_vec_int8_8x4_block(layers[l].up_proj, norm_buf, up_buf);
            
            silu_polynomial(gate_buf, config.hidden_dim);
            
            __m128* g_vec = (__m128*)gate_buf;
            __m128* u_vec = (__m128*)up_buf;
            __m128* s_vec = (__m128*)silu_buf;
            
            for (int i = 0; i < config.hidden_dim / 4; i++) {
                s_vec[i] = _mm_mul_ps(g_vec[i], u_vec[i]);
            }
            
            matmul_vec_int8_8x4_block(layers[l].down_proj, silu_buf, down_buf);
            
            h_vec = (__m128*)hidden;
            __m128* d_vec = (__m128*)down_buf;
            for (int i = 0; i < config.dim / 4; i++) {
                h_vec[i] = _mm_add_ps(h_vec[i], d_vec[i]);
            }
        }
        
        std::vector<float> logits(config.vocab_size);
        matmul_vec_int8_8x4_block(lm_head, hidden, logits.data());
        
        return logits;
    }
    
    int sample_token(const std::vector<float>& logits, float temperature = 0.8f) {
        if (temperature <= 0.0f) {
            int best = 0;
            float best_val = logits[0];
            for (int i = 1; i < config.vocab_size; i++) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best = i;
                }
            }
            return best;
        }
        
        float max_logit = logits[0];
        for (int i = 1; i < config.vocab_size; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        
        float sum = 0.0f;
        std::vector<float> probs(config.vocab_size);
        
        for (int i = 0; i < config.vocab_size; i++) {
            float val = (logits[i] - max_logit) / temperature;
            if (val < -16.0f) probs[i] = 0.0f;
            else if (val > 16.0f) probs[i] = 1e9f;
            else {
                val = 1.0f + val / 256.0f;
                val *= val; val *= val; val *= val; val *= val;
                val *= val; val *= val; val *= val; val *= val;
                probs[i] = val;
            }
            sum += probs[i];
        }
        
        float r = std::rand() / float(RAND_MAX);
        float cum = 0.0f;
        for (int i = 0; i < config.vocab_size; i++) {
            cum += probs[i] / sum;
            if (r <= cum) return i;
        }
        
        return config.vocab_size - 1;
    }
    
    std::vector<std::string> generate(const std::string& prompt, int max_tokens = 20) {
        std::vector<std::string> output;
        
        for (auto& cache : kv_caches) {
            cache.reset();
        }
        
        int start_token = vocab.count(prompt) ? vocab[prompt] : 0;
        output.push_back(inv_vocab[start_token]);
        
        for (int step = 0; step < max_tokens; step++) {
            int current_token_id;
            if (step == 0) {
                current_token_id = start_token;
            } else {
                current_token_id = vocab[output.back()];
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto logits = forward_step(current_token_id, step);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "Step " << step << ": " << duration.count() << " μs" << std::endl;
            
            int next_token = sample_token(logits);
            
            if (next_token == 1 || output.size() >= max_tokens) {
                break;
            }
            
            output.push_back(inv_vocab[next_token]);
        }
        
        return output;
    }
    
    void print_memory_usage() {
        size_t total_memory = 0;
        
        total_memory += token_embedding.stride * config.vocab_size * sizeof(int8_t);
        total_memory += config.vocab_size * sizeof(float);
        
        for (const auto& layer : layers) {
            total_memory += layer.q_proj.stride * config.dim * sizeof(int8_t);
            total_memory += config.dim * sizeof(float);
            total_memory += layer.k_proj.stride * config.dim * sizeof(int8_t);
            total_memory += config.dim * sizeof(float);
            total_memory += layer.v_proj.stride * config.dim * sizeof(int8_t);
            total_memory += config.dim * sizeof(float);
            total_memory += layer.o_proj.stride * config.dim * sizeof(int8_t);
            total_memory += config.dim * sizeof(float);
            total_memory += layer.gate_proj.stride * config.dim * sizeof(int8_t);
            total_memory += config.dim * sizeof(float);
            total_memory += layer.up_proj.stride * config.dim * sizeof(int8_t);
            total_memory += config.dim * sizeof(float);
            total_memory += layer.down_proj.stride * config.hidden_dim * sizeof(int8_t);
            total_memory += config.hidden_dim * sizeof(float);
            total_memory += config.dim * sizeof(float) * 2;
        }
        
        total_memory += lm_head.stride * config.dim * sizeof(int8_t);
        total_memory += config.dim * sizeof(float);
        
        for (const auto& cache : kv_caches) {
            total_memory += cache.capacity * cache.n_heads * cache.head_dim * sizeof(float) * 2;
        }
        
        total_memory += config.seq_len * sizeof(float);
        
        std::cout << "INT8 weights memory: " << total_memory / (1024 * 1024) << " MB" << std::endl;
    }
};

int main() {
    enable_daz_ftz();
    
    std::cout << "Minillm Optimized" << std::endl;
    std::cout << "8x4 block matmul" << std::endl;
    std::cout << "Software pipelining" << std::endl;
    std::cout << "Zero spill target" << std::endl;
    
    Minillm model;
    
    model.print_memory_usage();
    
    std::cout << "Generation Test" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto tokens = model.generate("t0", 5);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Generated: ";
    for (const auto& token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    
    return 0;
}
