# Serve Gemma 4 IT with Speculative Decoding (MTP) on GKE (TPU v6e)

This guide shows how to serve the Gemma 4 IT model (`google/gemma-4-31B-it`) with vLLM using speculative decoding on GKE with a Trillium (TPU v6e) node pool. We use the official `google/gemma-4-31B-it-assistant` companion model as the draft model.

Note: Speculative decoding for Gemma 4 on TPU currently requires specific python package hotpatches and source overrides to run stably without Out-of-Memory (OOM) or shape mismatch crashes. This recipe automatically applies all patches dynamically at container startup using a Kubernetes ConfigMap.

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

## Step 1: Create a GKE Nodepool with TPU v6e

Before deploying the workload, ensure your GKE cluster is configured and create a TPU v6e (Trillium) node pool with a `2x2` topology (4 chips).

```bash
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export PROJECT_ID=<YOUR_PROJECT_ID>
export REGION=<YOUR_REGION>
export ZONE=<YOUR_ZONE>
export NODEPOOL_NAME=gemma4-v6e-pool

gcloud container node-pools create ${NODEPOOL_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --node-locations=${ZONE} \
  --num-nodes=1 \
  --machine-type=ct6e-standard-4t \
  --cluster=${CLUSTER_NAME}
```

---

## Step 2: Configure kubectl and Secret

1. Configure kubectl to communicate with your GKE cluster:

    ```bash
    gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION}
    ```

2. Create a Kubernetes Namespace:

    ```bash
    kubectl create namespace gemma4-mtp
    ```

3. Create a Kubernetes Secret containing your Hugging Face Access Token (ensure your token has permissions to access `google/gemma-4-31B-it`):

    ```bash
    export HF_TOKEN=YOUR_HF_TOKEN
    kubectl create secret generic hf-secret \
        --from-literal=hf_api_token=${HF_TOKEN} \
        --namespace=gemma4-mtp
    ```

---

## Step 3: Create the ConfigMap with Hotpatch Files

Create a Kubernetes ConfigMap from the local python override files and patch scripts in this directory. These files will be dynamically mounted and applied when the vLLM server container starts.

```bash
kubectl create configmap gemma4-mtp-patches \
  --from-file=model_loader.py \
  --from-file=weight_utils.py \
  --from-file=patch_gemma4_mtp.py \
  --from-file=patch_qwix.py \
  --namespace=gemma4-mtp
```

---

## Step 4: Deploy the vLLM Serving Manifest

Apply the GKE serving manifest using the provided `gemma4-mtp-gke.yaml` file:

```bash
kubectl apply -f gemma4-mtp-gke.yaml
```

The manifest provisions a storage disk (`PersistentVolumeClaim`), creates a service to expose the model API, and deploys the vLLM server deployment. 

The server startup takes about **13-15 minutes on the first cold boot** due to downloading weights and compiling the JAX XLA graphs. Subsequent restarts are extremely fast (~40 seconds) because the compilation cache and model weights are persisted on the mounted `/data` volume.

You can monitor the server startup logs by running:

```bash
kubectl logs -n gemma4-mtp deployment/vllm-gemma4-server -f
```

---

## Step 5: Test Serving and Inference

Once the deployment readiness probe passes (you can check with `kubectl get pods -n gemma4-mtp`), forward the service port to test inference locally:

```bash
kubectl port-forward -n gemma4-mtp service/vllm-gemma4-service 8000:8000
```

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

## Verified GKE Serving Performance

Below are the verified performance results of serving Gemma 4 IT with MTP speculative decoding on GKE (v6e-4, 4 chips) across different workloads.

### 1. ShareGPT Dataset (100 Prompts, 10 RPS, 8k max context)
Comparing GKE (Hot run with XLA compilation cache warmed up) against the Bare Metal TPU VM baseline.

| Metric | Bare Metal TPU VM (README Baseline) | GKE Cluster (Warmed-up Hot Run) | Delta / Analysis |
| :--- | :---: | :---: | :--- |
| **Output Token Throughput** | 723.91 tok/s | **957.60 tok/s** | **+32.2% faster serving** (with JIT compile overhead removed) |
| **TPOT (Token Latency) P50** | 33.85 ms | **27.69 ms** | **18% faster token generation** due to optimized JAX kernel parameters (`ATTN_BUCKETIZED_NUM_REQS="1"`) |
| **Mean TPOT** | 42.13 ms | **36.17 ms** | **14% lower mean latency** |
| **Draft Acceptance Rate** | 63.51% | 60.41% | Consistent predictor accuracy |
| **Average Acceptance Length** | 3.54 tokens | 4.02 tokens | Better speculative decoding block predictions |

### 2. Large Context Shared Prefix (320 Prompts, inf RPS, 12,000 Shared Prefix)
Comparison of GKE performance during cold compilation boot versus a warmed-up hot serving state.

| Metric | GKE Cold Run (On-the-fly JIT Compile) | GKE Hot Run (Cached XLA compilation) | Delta / Performance Gain |
| :--- | :---: | :---: | :--- |
| **Benchmark Duration** | 999.88 s | **418.06 s** | **-58.1% (2.4x faster overall run)** |
| **Output Token Throughput** | 63.01 tok/s | **150.70 tok/s** | **+139.1% (2.4x higher output throughput)** |
| **TPOT (Token Latency) P50** | 14.26 ms | **13.97 ms** | Stable token generation speed |
| **TPOT (Token Latency) Mean** | 38.01 ms | **17.21 ms** | **-54.7% (eliminates compile stalls)** |
| **TTFT (Time to First Token) P50** | 779.45 s | **205.45 s** | **-73.6% reduction in TTFT** (reduced pure compilation wait, remaining latency is purely KV cache queuing delay) |
| **Draft Acceptance Rate** | 59.77% | 59.77% | Identical verification accuracy |

---

## uBench Internal Verification

This GKE speculative decoding recipe has been verified and registered on the internal uBench Telemetry system:
* **uBench Run Name:** `jawadamin-ubench-cytfr6c9`
* **uBench Dashboard Query Link:** [go/ubench-dash](http://go/ubench-dash) (Search for Run ID `vllm_inference-gemma-4-31B-it-2026-07-08_184512-47095651-3bd0-45ba-9259-e0227d703f62`)
* **Verified Output Throughput:** 63.15 tokens/s (Mean TPOT: 38.00 ms, Speculative Acceptance Rate: 61.05%)

