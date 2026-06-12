# Speculative Decoding Benchmark Results

*Hardware: TPU v6e-4 (4 Chips)*  
*Target Model: google/gemma-4-31B-it (FP8 Quantized)*  
*Draft Model: google/gemma-4-31B-it-assistant (FP8 Quantized)*  
*Memory Configuration: --gpu-memory-utilization 0.65 (Stable HBM profile)*

Below are the performance comparisons of speculative decoding on TPU using the official companion Assistant model against the established baseline.

---

## 📊 Performance Summary

### 1. Text-Only Performance (ShareGPT Dataset)

| Metric | BF16 Baseline (Non-Speculative) | FP8 Speculative Decoding (MTP Assistant) | Improvement / Notes |
| :--- | :---: | :---: | :--- |
| **Successful requests** | 100 | 100 | Identical run config |
| **Benchmark duration** | 61.05s | **30.24s** | **🚀 50.4% shorter run** |
| **Output Token Throughput** | 374.31 tok/s | **723.91 tok/s** | **🚀 93.4% throughput increase** |
| **Median TPOT (Token Latency)** | **63.55 ms** | **33.85 ms** | **🚀 46.7% faster generation** |
| **Mean TPOT** | 65.92 ms | **42.13 ms** | **🚀 36.1% faster generation** |
| **Median ITL** | 60.81 ms | 99.07 ms | Higher step time due to verification loop |
| **Median TTFT** | 201.73 ms | 424.15 ms | Queue wait time at 10 RPS |
| **Draft Acceptance Rate** | - | **63.51%** | High accuracy (MTP Assistant) |
| **Average Acceptance Length** | - | **3.54 tokens** | Proposes 4, accepts ~3.5 per step |

### 2. Multimodal Performance (Text + 1 Image)

| Metric | BF16 Baseline (Non-Speculative) | FP8 Speculative Decoding (MTP Assistant) | Improvement / Notes |
| :--- | :---: | :---: | :--- |
| **Successful requests** | 100 | 100 | Identical run config |
| **Benchmark duration** | 52.60s | 54.21s | Comparable total duration |
| **Output Token Throughput** | 243.33 tok/s | 236.10 tok/s | ~3% decrease (KV Cache queue constraint) |
| **Median TPOT (Token Latency)** | **131.43 ms** | **106.61 ms** | **🚀 18.9% faster generation** |
| **Mean TPOT** | 133.51 ms | **124.23 ms** | **🚀 6.9% faster generation** |
| **Median ITL** | 58.82 ms | 150.82 ms | Higher step time due to verification loop |
| **Median TTFT** | 34.55s | **20.35s** | **🚀 41.1% TTFT reduction** |
| **Draft Acceptance Rate** | - | **48.03%** | Solid prediction accuracy |
| **Average Acceptance Length** | - | **2.92 tokens** | Proposes 4, accepts ~2.9 per step |

---

## 🔍 Key Insights & Analysis

### 1. ⚡ The Speculative Decoding Metric Divergence
We observe a characteristic divergence in metrics when using speculative decoding:
*   **Inter-Token Latency (ITL)** increases. Since the engine groups draft generation and verification steps, the time between engine steps increases (e.g., from 60ms to 99ms for text).
*   **Time per Output Token (TPOT)** significantly decreases. Because each engine step returns multiple tokens (e.g., **3.54 accepted tokens** for text), the effective average generation cost per token drops dramatically, yielding a **46.7% speedup for text** and a **18.9% speedup for multimodal**.

### 2. 🚀 Throughput Dynamics (Text vs Multimodal)
*   **Text workloads** saw a near-doubling of output token throughput (**723.91 tok/s** vs **374.31 tok/s**).
*   **Multimodal workloads** achieved a substantial reduction in per-token latency (106ms vs 131ms), but overall throughput was constrained (236 tok/s vs 243 tok/s). 
    *   *Why?* To run speculative decoding stably without HBM OOM crashes, we had to decrease `--gpu-memory-utilization` from `0.90` (baseline) to `0.65`.
    *   This reduced the total available KV Cache block count. Under concurrent image-processing load (`request-rate: inf`), the smaller KV Cache pool hit capacity sooner, forcing the scheduler to process smaller request batches and queue the rest, throttling max concurrent throughput.

### 3. 🎯 Draft Model Performance and Acceptance
*   For text, the MTP assistant achieves a **63.51% acceptance rate**, confirming high alignment.
*   For multimodal, the assistant achieves a **48.03% acceptance rate**. This is a solid result, as the assistant model does not process image embeddings but successfully speculates the text output tokens following the image description context.
