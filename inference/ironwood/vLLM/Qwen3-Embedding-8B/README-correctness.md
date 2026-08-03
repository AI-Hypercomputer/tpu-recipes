# Qwen3-Embedding-8B E2E Correctness Validation on TPUs

This guide provides step-by-step instructions for running end-to-end correctness validation of the **Qwen3-Embedding-8B** model on Google Cloud TPUs. The validation process ensures numerical parity between TPU and CPU baselines, specifically for long-context workloads.

## 1. Resource Provisioning

Choose the provisioning command corresponding to your target hardware:

### For v7x (Ironwood)
```bash
gcloud alpha compute tpus queued-resources create <RESOURCE_ID> \
    --node-id <NODE_ID> \
    --project <PROJECT_ID> \
    --zone <ZONE> \
    --accelerator-type v7x-8 \
    --runtime-version v2-alpha-tpuv7 \
    --reserved
```

### For v6e (Trillium)
```bash
gcloud compute tpus tpu-vm create <NODE_ID> \
    --project <PROJECT_ID> \
    --zone <ZONE> \
    --accelerator-type v6e-4 \
    --version v2-alpha-tpuv6e
```

## 2. Environment Setup
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

# Sync to the validated LKG version of vLLM containing the StepPool chunked prefill fix
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

### 3. Validation Scripts
The following validation scripts are provided in this directory to execute the parity check:

* [embed_script.py](file:///usr/local/google/home/aysu/projects/tpu-recipes/inference/ironwood/vLLM/Qwen3-Embedding-8B/embed_script.py): Uses the pooling runner to generate embeddings.
* [compare_precision.py](file:///usr/local/google/home/aysu/projects/tpu-recipes/inference/ironwood/vLLM/Qwen3-Embedding-8B/compare_precision.py): Calculates the Cosine Similarity between CPU and TPU output tensors.

## 4. Generate Embeddings and Verify Numerical Parity
Run the validation suite sequentially:

```bash
# Run CPU Baseline
export VLLM_TARGET_DEVICE="cpu"
python embed_script.py

# Run TPU Validation
export VLLM_TARGET_DEVICE="tpu"
python embed_script.py

# Run Comparison
python compare_precision.py
```

## 5. Expected Results
The Cosine Similarity metrics comparing Cloud TPUs against CPU reference implementations using 7K token sequences.

| Input Type | v7x Cosine Similarity | v6e Cosine Similarity |
|---|---|---|
| English (Short) | 0.9995546074 | 0.9996544555 |
| English (Long) | 0.9997845670 | 0.9997664366 |
| Japanese (Long) | 0.9997769296 | 0.9997792912 |
| Korean (Long) | 0.9997860590 | 0.9998040753 |
