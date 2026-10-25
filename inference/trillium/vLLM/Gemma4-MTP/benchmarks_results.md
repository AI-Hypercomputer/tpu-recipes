# Gemma 4 MTP Speculative Decoding GKE Benchmark Results

This report documents the performance results of running **Gemma 4 31B IT** with **Multi-Token Prediction (MTP)** speculative decoding on a Cloud TPU v6e-4 node pool under GKE.

---

## 1. Test Environment

*   **Platform:** Google Kubernetes Engine (GKE)
*   **TPU Hardware:** Cloud TPU v6e-4 slice (4 chips, 64GB total HBM)
*   **TPU Reservation:** `<tpu-reservation>`
*   **Model:** `google/gemma-4-31B-it` (FP8 quantized using Qwix)
*   **Speculative Assistant:** `google/gemma-4-31B-it-assistant` (FP8 quantized, 5 draft tokens)
*   **VLLM Container:** `vllm/vllm-tpu:nightly-20260611-1043491-248e33c`
*   **Quantization:** Weight FP8, Activation FP8 (using dynamic runtime patches)

---

## 2. Benchmark 1: ShareGPT serving dataset (100 prompts)

The benchmark client sent 100 prompts from the ShareGPT dataset at a rate of 10.0 requests per second.

### Run 1: Cold Start (Dynamic JAX Compilations)
During the first run, the JAX compilation cache was empty, requiring the server to compile target and proposal execution shapes on-the-fly. This blocked the engine event loop during compilation events.
*   **Successful requests:** 100 / 100
*   **Acceptance rate:** 60.78% (Mean acceptance length: 4.04 out of 5 tokens)
*   **Peak output token throughput:** 416.00 tokens/s
*   **Total duration:** 1660.32 seconds (27.6 minutes, heavily skewed by compilations)
*   **Mean TTFT:** 579.75 seconds

### Run 2: Warmed Up (Cached JAX Compilation Graphs)
The second run reused the populated JAX compilation cache (`/root/.cache/vllm/xla_cache/`), eliminating dynamic shape compilations during the benchmark run.
*   **Successful requests:** 100 / 100
*   **Benchmark duration:** 73.57 seconds
*   **Request throughput:** 1.36 req/s
*   **Output token throughput:** 315.92 tokens/s
*   **Peak output token throughput:** 448.00 tokens/s
*   **Total token throughput:** 648.65 tokens/s
*   **Mean TTFT:** 50.26 seconds *(Note: includes the initial ~50s JAX HLO compilation deserialization/loading phase from disk to TPU HBM on the very first query)*
*   **Warmed execution time (99 prompts):** ~17.01 seconds (~5.8 req/s throughput)
*   **Time per Output Token (TPOT):**
    *   **Mean TPOT:** 38.46 ms
    *   **Median TPOT:** 26.59 ms
    *   **P99 TPOT:** 121.84 ms
*   **Inter-token Latency (ITL):**
    *   **Mean ITL:** 122.42 ms
    *   **Median ITL:** 113.59 ms
    *   **P99 ITL:** 176.32 ms
*   **Speculative Decoding Alignment:**
    *   **Acceptance rate (%):** 64.87%
    *   **Acceptance length:** 4.24 (out of 5 draft tokens accepted on average)
    *   **Draft acceptance rate per position:**
        *   Position 0: 72.56%
        *   Position 1: 66.06%
        *   Position 2: 63.55%
        *   Position 3: 61.95%
        *   Position 4: 60.26%

---

## 3. Benchmark 2: Prefix Repetition (6k Context)

This test measured performance under heavy context length and concurrency. 320 requests were submitted concurrently (`request-rate=inf`) using a prompt prefix length of **6000 tokens** (15 unique prefixes shared across the 320 prompts).

*   **Successful requests:** 315 / 315
*   **Benchmark duration:** 291.67 seconds (4.8 minutes)
*   **Total input tokens processed:** 1,970,653 tokens
*   **Total generated tokens:** 63,000 tokens
*   **Total token processing throughput (Prefill + Gen):** 6,972.36 tokens/s
*   **Output token throughput:** 216.00 tokens/s
*   **Peak output token throughput:** 233.00 tokens/s
*   **Prefix Cache Hit Rate:** **62.9%** (reused KV cache keys for shared prefixes)
*   **Mean TTFT:** 176.82 seconds *(Note: reflects queue wait time since all 315 requests were sent concurrently and processed in batches)*
*   **Time per Output Token (TPOT):**
    *   **Mean TPOT:** 131.21 ms
    *   **Median TPOT:** 58.35 ms
    *   **P99 TPOT:** 851.82 ms
*   **Inter-token Latency (ITL):**
    *   **Mean ITL:** 446.72 ms
    *   **Median ITL:** 233.73 ms
    *   **P99 ITL:** 841.56 ms
*   **Speculative Decoding Alignment:**
    *   **Acceptance rate (%):** 48.98%
    *   **Acceptance length:** 3.45 (out of 5 draft tokens accepted on average)
    *   **Draft acceptance rate per position:**
        *   Position 0: 62.25%
        *   Position 1: 49.99%
        *   Position 2: 46.38%
        *   Position 3: 43.76%
        *   Position 4: 42.53%

---

## 4. Key Performance Insights

1.  **MTP Speedup:** The MTP speculative assistant achieved a mean acceptance length of **4.24 tokens** on ShareGPT and **3.45 tokens** on Prefix Repetition. This provides a **3.5x to 4.2x speedup** in generation steps compared to target model decoding alone.
2.  **Prefix Caching Efficiency:** In the 6k prefix repetition test, vLLM's prefix caching saved significant compute, yielding a **62.9% prefix hit rate**. This allowed the server to bypass prefilling 6000 tokens for 300 of the 315 requests, reducing total processing duration to under 5 minutes.
3.  **Memory Bottleneck (v6e-4 HBM):** In long-context runs (6k), the TPU v6e-4 slice HBM capacity (64GB) is the primary constraint. Running a 31B model leaves limited KV cache block space. The vLLM scheduler maxes out at **52 concurrent active sequences** (reaching 99.9% cache usage). Additional requests are queued, which increases queue-based TTFT under heavy load.
4.  **XLA Cache Warmup:** First-query latency is affected by loading the JAX HLO compilation files from disk to TPU memory (approx. 50s). Subsequent queries process with native latency (26ms median TPOT).
