# Technical Hotpatch & Configuration Guide

This guide details the specific code hotpatches and configuration adjustments required to run **Gemma 4 Speculative Decoding** stably on the TPU `v6e-4` accelerator using vLLM and the official `gemma-4-31B-it-assistant` Multi-Token Prediction (MTP) companion model.

---

## 🛠️ The Core Challenges & Solutions

To run speculative decoding on TPU v6e without OOMs or program load crashes, we had to resolve four key issues:

### 1. Heterogeneous KV Cache Allocation Mismatch
*   **The Issue:** vLLM's internal Qwix quantization helper assumes that the KV Cache shapes for the draft and target models are identical. Since the draft model is 0.5B parameters and the target model is 31B parameters, their layer and head counts differ. This causes runtime assertions in shape verification during tracing.
*   **The Patch:** [patch_qwix.py](file:///usr/local/google/home/jawadamin/Repos/tpu-recipes/custom_scripts/gemma/patch_qwix.py) overrides the rules inside vLLM's `qwix_utils.py` to correctly allocate separate, heterogeneous KV cache shapes for the target and draft model runners.

### 2. Eager Multimodal Validation Failure (Assistant Tracing)
*   **The Issue:** The target model (`google/gemma-4-31B-it`) is multimodal, but its companion assistant (`google/gemma-4-31B-it-assistant`) is text-only. During tracing, Hugging Face's `transformers` processor throws verification errors when a multimodal processor is initialized for a text-only model configuration.
*   **The Patch:** The server startup script automatically patches the Hugging Face `processing_gemma4.py` script inside the Docker container to bypass the dummy multimodal validation rule when loading assistant models.

### 3. Rejection Sampler Multimodal Token Wipeout
*   **The Issue:** During speculative validation of multimodal inputs, the JAX speculative validation loop wipes out the `input_ids` parameter (replacing it with `None`) to force the draft model sampler to read from the cached tokens. However, the multimodal vision path requires the presence of `input_ids` to map image tokens. Wiping it causes a null-pointer error crash.
*   **The Patch:** We hotpatched the container's `tpu_runner.py` to preserve the original `input_ids` instead of wiping them during speculative validation steps when multimodal inputs are detected.

### 4. Compiler Out-Of-Memory (HBM Squeeze)
*   **The Issue:** Compiling and holding the XLA graphs for both the 31B target model and the 0.5B draft model simultaneously in host and TPU memory exceeds the safety threshold of `0.90` memory utilization, leading to `RESOURCE_EXHAUSTED` OOM crashes during JAX program loading.
*   **The Adjustment:** We configured the server to run with `--gpu-memory-utilization 0.65`. This reduces the size of the KV cache pool, freeing up enough TPU HBM to safely load both model binaries and run the XLA graph compilations.

---

## 🚀 Execution Guide

### 1. Starting the Speculative Server
Use the [run_speculative_server.sh](file:///usr/local/google/home/jawadamin/Repos/tpu-recipes/custom_scripts/gemma/run_speculative_server.sh) script to boot up the container, apply the patches, and start the Uvicorn server:

```bash
./custom_scripts/gemma/run_speculative_server.sh <YOUR_HF_TOKEN>
```

The script will automatically:
1. Boot the nightly vLLM TPU container.
2. Install git-source `transformers` dependencies.
3. Apply the container-level Python files patches (`processing_gemma4.py`, `tpu_runner.py`, and `gemma4_mtp.py`).
4. Apply the `patch_qwix.py` patch.
5. Launch the API server with speculative decoding configurations on port `8000`.
6. Poll the models endpoint until the server is fully ready.

### 2. Running the Performance Benchmarks
Use [gate_speculative_performance.sh](file:///usr/local/google/home/jawadamin/Repos/tpu-recipes/custom_scripts/gemma/gate_speculative_performance.sh) to run the performance evaluations:

```bash
./custom_scripts/gemma/gate_speculative_performance.sh
```

This will run:
*   A text benchmark using the ShareGPT dataset.
*   A multimodal benchmark using randomly generated prompts containing 1 image.
*   The output stats (TPOT, throughput, acceptance rates) will print to stdout.
