# Qwen3-VL-Embedding-8B E2E Correctness Validation on TPU Ironwood

This guide provides step-by-step instructions for running end-to-end correctness and multimodal validation of the **Qwen3-VL-Embedding-8B** model on Ironwood (TPU7x).

## 1. Resource Provisioning

Provision an Ironwood (v7x) TPU VM.
```bash
gcloud alpha compute tpus queued-resources create <RESOURCE_ID> \
    --node-id <NODE_ID> \
    --project <PROJECT_ID> \
    --zone <ZONE> \
    --accelerator-type v7x-8 \
    --runtime-version v2-alpha-tpuv7 \
    --reserved
```

## 2. Environment Setup

### TPU Environment Setup
After connecting to your TPU VM, set up the environment. Select either Option A (prebuilt installation) or Option B (compiling from source).

```bash
# Set up uv and activate venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv embed_env --python 3.12
source embed_env/bin/activate
```

**Option A:** Install official stable release

```bash
uv pip install vllm-tpu
```

**Option B:** Install from source code

```bash
# Clone the official tpu-inference repository
git clone https://github.com/vllm-project/tpu-inference.git
cd tpu-inference

# Sync to validated LKG version of vLLM
VLLM_COMMIT=$(cat .buildkite/vllm_lkg.version)
cd ..

# Install vLLM from the specified validated LKG commit
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout $VLLM_COMMIT
VLLM_TARGET_DEVICE="tpu" uv pip install -e .
cd ..

# Finalize tpu-inference installation
cd tpu-inference
uv pip install -e .
```

### GPU Environment Setup (Reference baseline machine)
Ensure your reference GPU (Blackwell/Hopper) VM is set up with the same relative vLLM version:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv embed_env --python 3.12
source embed_env/bin/activate

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout a51376b3f05a2f74eac6ceeed7e52598b871a0fb
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

### 3. Validation Scripts
The following validation scripts are provided in this directory to execute the parity check:

* [embed_vl_script.py](file:///usr/local/google/home/aysu/projects/tpu-recipes/inference/ironwood/vLLM/Qwen3-VL-Embedding-8B/embed_vl_script.py): Validates both multimodal inputs and pure text baselines.
* [compare_precision.py](file:///usr/local/google/home/aysu/projects/tpu-recipes/inference/ironwood/vLLM/Qwen3-VL-Embedding-8B/compare_precision.py): Calculates and compares cosine similarities.

## 4. Run Parity Comparison
Execute the following runs on both TPU and GPU devices to generate the outputs:

```bash
# Run TPU Pooling
TEST_MODE=text VLLM_TARGET_DEVICE=tpu python embed_vl_script.py
TEST_MODE=multimodal VLLM_TARGET_DEVICE=tpu python embed_vl_script.py

# Run GPU Pooling
TEST_MODE=text VLLM_TARGET_DEVICE=cuda python embed_vl_script.py
TEST_MODE=multimodal VLLM_TARGET_DEVICE=cuda python embed_vl_script.py

# Run Comparison
python compare_precision.py
```


## 5. Expected Results
Expected cosine similarity on v7x (Ironwood) against GPU baselines:

```bash
Numerical Alignment Summary (TPU vs GPU)
==================================================

--- Result for [TEXT] Mode ---
TPU File: embed-text-tpu.json
GPU File: embed-text-cuda.json
Similarity: xxx

--- Result for [MULTIMODAL] Mode ---
TPU File: embed-multimodal-tpu.json
GPU File: embed-multimodal-cuda.json
Similarity: xxx

```

Below are the alignment metrics Ironwood(v7x) compared to Blackwell (B200) and Hopper (H200) targets:

| Input Type | Chunked Prefill* | B200 Cosine Similarity | H200 Cosine Similarity |
|---|---|---|---|
| Text | True | 0.99986249 | 0.999819165 |
| Multimodal | True | 0.99749935 | 0.99912983 |
| Text | False | 0.99986249 | 0.99985290 |
| Multimodal | False | 0.99749935 | 0.99850363 |

\*vLLM-TPU only chunks the text portion of multimodal prefill.
