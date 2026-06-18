# Serve Gemma 4 IT with Speculative Decoding (MTP) on Trillium TPU VMs

This guide shows how to serve the Gemma 4 IT model (google/gemma-4-31B-it) with vLLM using speculative decoding on Trillium (TPU v6e) VMs. We use the official google/gemma-4-31B-it-assistant companion model as the draft model.

Note: Speculative decoding for Gemma 4 on TPU currently requires specific python package hotpatches and source overrides to run stably without Out-of-Memory (OOM) or shape mismatch crashes. All necessary python files and hotpatch scripts are included in this directory.

---

## Verified Models and Hardware

| Model | Draft Model (Assistant) | Topology | TP Size | Hugging Face |
| :--- | :--- | :---: | :---: | :--- |
| Gemma 4 31B IT (FP8) | Gemma 4 31B IT Assistant (FP8) | v6e-4 (4 chips) | 4 | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |

---

## Technical Details: Patch Manifest

Running speculative decoding with Gemma 4 on TPU v6e requires specific hotpatches to prevent compilation crashes and optimize memory layout. Below is a manifest detailing the need and source package for each patch applied by this recipe:

| File Patched | Source Package / Path | Local File | Issue | Fix / Resolution |
| :--- | :--- | :--- | :--- | :--- |
| `model_loader.py` | `tpu_inference.models.common.model_loader` | `model_loader.py` | Draft model sharding across multiple TPU chips causes high inter-chip communication latency for small tensors. | Traverses and forces the draft model parameters and states to be fully replicated (`NamedSharding(mesh, PartitionSpec())`) across all chips. |
| `weight_utils.py` | `tpu_inference.models.jax.utils.weight_utils` | `weight_utils.py` | Draft model weights are partitioned across chips by default during weights loading. | Overrides JAX weight loading sharding configurations to load draft model weights as fully replicated. |
| `qwix_utils.py` | `tpu_inference.models.jax.utils.qwix.qwix_utils` | Run via `patch_qwix.py` | Qwix assumes draft and target KV caches have identical head/layer dimensions, causing crashes for heterogeneous configurations. | Dynamic read is added to pull head/layer configurations from target and draft model configs individually. |
| `gemma4_mtp.py` | `tpu_inference.models.jax.gemma4_mtp` | Run via `patch_gemma4_mtp.py` | JAX model compilation fails because `hidden_states` from the target model are required but unavailable during assistant-only passes. | Modifies call signatures to make `hidden_states` optional and instantiates empty zero tensors for compilation. |
| `processing_gemma4.py`| `transformers.models.gemma4.processing_gemma4` | Applied in-place via regex | Eager validation throws errors when loading a text-only assistant config with a multimodal processor. | Bypasses the dummy validation check on container startup. |
| `tpu_runner.py` | `tpu_inference.runner.tpu_runner` | Applied in-place via regex | Multimodal validation wipes out `input_ids` during speculative steps, crashing the vision processing pipeline. | Preserves `input_ids` variables during speculative validation when images are present. |

In addition to these code modifications, we configure `--gpu-memory-utilization 0.65` (down from 0.90) to lower the size of the KV cache pool, freeing up enough TPU HBM to safely load both model binaries and run the XLA graph compilations concurrently.

---

## Step 1: Create a v6e TPU Instance

Create a single TPU v6e VM with 4 chips (topology 2x2).

```bash
export TPU_NAME=gemma-mtp-vm
export ZONE=us-east5-b
export PROJECT=your-gcp-project

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --type v6e --topology 2x2 \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv6e
```

---

## Step 2: SSH to the TPU Instance and Clone Recipes

SSH into the newly created TPU VM:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

Once inside the VM, clone the tpu-recipes repository to access the patch files:

```bash
git clone https://github.com/AI-Hypercomputer/tpu-recipes.git
cd tpu-recipes/inference/trillium/vLLM/Gemma4-MTP
```

---

## Step 3: Run the Docker Container

Start the nightly vLLM TPU container with host volume mounts and privileged access:

```bash
export HF_TOKEN=your_huggingface_token

sudo docker run -d --name vllm-gemma4 --privileged --network host --shm-size 16g \
  -v /dev/shm:/dev/shm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HOME='/root/.cache/huggingface' \
  -e USE_BATCHED_RPA_KERNEL=1 \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
  vllm/vllm-tpu:nightly-20260611-1043491-248e33c \
  sleep infinity
```

---

## Step 4: Apply the Hotpatches

Copy the python files and patch runners from the cloned folder into the container:

```bash
# Copy python file overrides
sudo docker cp model_loader.py vllm-gemma4:/workspace/tpu_inference/tpu_inference/models/common/model_loader.py
sudo docker cp weight_utils.py vllm-gemma4:/workspace/tpu_inference/tpu_inference/models/jax/utils/weight_utils.py
sudo docker cp compilation_manager.py vllm-gemma4:/workspace/tpu_inference/tpu_inference/runner/compilation_manager.py
sudo docker cp configs.py vllm-gemma4:/workspace/tpu_inference/tpu_inference/kernels/experimental/batched_rpa/configs.py

# Copy patch runners
sudo docker cp patch_gemma4_mtp.py vllm-gemma4:/tmp/patch_gemma4_mtp.py
sudo docker cp patch_qwix.py vllm-gemma4:/tmp/patch_qwix.py
```

Execute the patches inside the container:

```bash
# Install transformers git source
sudo docker exec vllm-gemma4 pip install git+https://github.com/huggingface/transformers.git

# Apply processing_gemma4.py patch
sudo docker exec vllm-gemma4 python3 -c "
path = '/usr/local/lib/python3.12/site-packages/transformers/models/gemma4/processing_gemma4.py'
with open(path, 'r') as f:
    text = f.read()
bad_str = 'raise ValueError(\n                    f\"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed.\\\"\n                )'
if bad_str in text:
    text = text.replace(bad_str, 'pass')
    with open(path, 'w') as f:
        f.write(text)
"

# Apply tpu_runner.py patch
sudo docker exec vllm-gemma4 python3 -c "
path = '/workspace/tpu_inference/tpu_inference/runner/tpu_runner.py'
with open(path, 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'input_ids, inputs_embeds = self._get_input_ids_embeds' in line:
        lines[i] = line.replace('input_ids, inputs_embeds', 'forward_input_ids, inputs_embeds')
    elif 'self.kv_caches,' in lines[i-1] and 'input_ids,' in line and 'attn_metadata,' in lines[i+1]:
        lines[i] = line.replace('input_ids,', 'forward_input_ids,')
with open(path, 'w') as f:
    f.writelines(lines)
"

# Run the patch scripts
sudo docker exec vllm-gemma4 python3 /tmp/patch_gemma4_mtp.py
sudo docker exec vllm-gemma4 python3 /tmp/patch_qwix.py
```

---

## Step 5: Serve the Model

Start the API server inside the container with speculative config flags:

```bash
sudo docker exec -d vllm-gemma4 python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --speculative-config '{"model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 5}' \
  --additional_config '{"quantization": { "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}' \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype fp8 \
  --block-size 32 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

Wait about 3-5 minutes for JAX compilation to complete. You can verify readiness by checking:

```bash
curl http://localhost:8000/v1/models
```

---

## Step 6: Test Inference

Submit a test request using curl:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-31B-it",
        "messages": [
            {
                "role": "user",
                "content": "Explain the concept of speculative decoding in LLMs."
            }
        ],
        "max_tokens": 200,
        "temperature": 0.0
    }'
```

---

## Benchmark Results

Below are the performance comparisons of speculative decoding on TPU using the official companion Assistant model against the non-speculative baseline on a TPU v6e-4 (4 chips) node.

### 1. Text-Only Performance (ShareGPT Dataset)

| Metric | BF16 Baseline (Non-Speculative) | FP8 Speculative Decoding (MTP Assistant) | Improvement / Notes |
| :--- | :---: | :---: | :--- |
| Successful requests | 100 | 100 | Identical run config |
| Benchmark duration | 61.05s | 30.24s | 50.4% shorter run |
| Output Token Throughput | 374.31 tok/s | 723.91 tok/s | 93.4% throughput increase |
| Median TPOT (Token Latency) | 63.55 ms | 33.85 ms | 46.7% faster generation |
| Mean TPOT | 65.92 ms | 42.13 ms | 36.1% faster generation |
| Median ITL | 60.81 ms | 99.07 ms | Higher step time due to verification loop |
| Median TTFT | 201.73 ms | 424.15 ms | Queue wait time at 10 RPS |
| Draft Acceptance Rate | - | 63.51% | High accuracy (MTP Assistant) |
| Average Acceptance Length | - | 3.54 tokens | Proposes 4, accepts ~3.5 per step |

### 2. Multimodal Performance (Text + 1 Image)

| Metric | BF16 Baseline (Non-Speculative) | FP8 Speculative Decoding (MTP Assistant) | Improvement / Notes |
| :--- | :---: | :---: | :--- |
| Successful requests | 100 | 100 | Identical run config |
| Benchmark duration | 52.60s | 54.21s | Comparable total duration |
| Output Token Throughput | 243.33 tok/s | 236.10 tok/s | ~3% decrease (KV Cache queue constraint) |
| Median TPOT (Token Latency) | 131.43 ms | 106.61 ms | 18.9% faster generation |
| Mean TPOT | 133.51 ms | 124.23 ms | 6.9% faster generation |
| Median ITL | 58.82 ms | 150.82 ms | Higher step time due to verification loop |
| Median TTFT | 34.55s | 20.35s | 41.1% TTFT reduction |
| Draft Acceptance Rate | - | 48.03% | Solid prediction accuracy |
| Average Acceptance Length | - | 2.92 tokens | Proposes 4, accepts ~2.9 per step |
