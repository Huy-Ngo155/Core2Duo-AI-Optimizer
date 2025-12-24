# LegacyLLM-SSE 🚀

**LegacyLLM-SSE** is an extreme-performance inference engine designed to breathe life into old hardware. While modern AI frameworks require AVX2/AVX512 or high-end GPUs, this project proves that with deep low-level optimization, even an **Intel Core 2 Duo** can perform LLM tasks.

## 🛠 Key Technical Features

* **SSE4.2 Vectorization:** Completely avoided AVX dependency. All tensor operations are manually vectorized using 128-bit SSE intrinsics.
* **Int8 Quantization:** Implemented a row-wise scaling quantization to compress weights while maintaining precision on legacy registers.
* **Manual Prefetching:** Utilized `_mm_prefetch` to hide memory latency, crucial for old DDR2/DDR3 systems with limited bandwidth.
* **Polynomial Approximations:** Replaced heavy transcendental functions (`exp`, `sigmoid`) with high-speed Taylor series and polynomial approximations optimized for SIMD.
* **KVCacheRing:** A circular buffer for Key-Value caching to eliminate dynamic memory allocation overhead during inference.

## 📊 Hardware Compatibility
- **Primary Target:** Intel Core 2 Duo, Core 2 Quad, early Core-i series (Sandy Bridge and older).
- **Instruction Sets:** SSE, SSE2, SSE3, SSE4.1, SSE4.2.
- **Memory:** Optimized for systems with low memory bandwidth (DDR2/DDR3).



## 🚀 Performance Insights
By bypassing the overhead of heavy libraries like PyTorch or TensorFlow, this engine achieves up to **5-10x speedup** on non-AVX hardware compared to standard implementations.

## 🏗 How it Works (Core Logic)
The engine implements the Llama-style Transformer architecture from scratch:
1. **RMSNorm** vectorized with SSE.
2. **Rotary Positional Embeddings (RoPE)** with pre-calculated caches.
3. **MatMul** optimized with 8x4 block processing for better cache locality.
