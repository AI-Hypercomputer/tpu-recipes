# Serve Qwen3.6-27B-FP8 with vLLM on GKE

In this guide, we show how to deploy, serve, and benchmark [Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) with vLLM on Google Kubernetes Engine (GKE) targeting Cloud TPU v6e-4 (4 chips, 2x2 topology).

---

## Architecture & Verified Configuration

| Parameter | Configuration |
| :--- | :--- |
| **Model** | `Qwen/Qwen3.6-27B-FP8` (Pre-quantized FP8) |
| **Accelerator** | Cloud TPU v6e-4 (1 node, 4 chips, topology `2x2`) |
| **Orchestration** | Google Kubernetes Engine (GKE) |
| **Tensor Parallelism** | `TP=4` |
| **Context Length** | `32768` (32K tokens) |
| **Storage Tier** | Hyperdisk Balanced PVC (`qwen-model-cache`) |
| **Optimizations** | Persistent Prefix Caching (`--enable-prefix-caching`), FP8 KV Cache (`--kv-cache-dtype=fp8`), Chunked Prefill (`--enable-chunked-prefill`) |

---

## Step 0: Prerequisites & Environment Setup

Ensure you have installed the `gcloud CLI` and `kubectl` on your local work environment:
* [Install gcloud CLI](https://cloud.google.com/sdk/docs/install)
* Authenticate GCP credentials: `gcloud auth login`

---

## Step 1: Create a GKE Cluster and TPU v6e-4 Node Pool

If you do not already have an active GKE cluster with a TPU v6e-4 node pool, create one using `gcloud`:

```bash
export CLUSTER_NAME=vllm-gke-cluster
export ZONE=us-east5-a
export PROJECT=your-gcp-project

# 1. Create base GKE Cluster with GCS Fuse CSI Driver enabled
gcloud container clusters create $CLUSTER_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --release-channel=regular \
    --addons=GcsFuseCsiDriver \
    --workload-pool=$PROJECT.svc.id.goog

# 2. Add TPU v6e-4 (4-chip) Node Pool
gcloud container node-pools create tpu-v6e-pool \
    --cluster=$CLUSTER_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=ct6e-standard-4t \
    --tpu-topology=2x2 \
    --num-nodes=1

# 3. Fetch cluster credentials for kubectl
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE --project=$PROJECT
```

---

## Step 2: Configure Namespace and Hugging Face API Secret

Create the `qwen-serving` namespace and store your Hugging Face access token:

```bash
export HF_TOKEN=your_huggingface_token

kubectl create namespace qwen-serving

kubectl create secret generic hf-secret \
    -n qwen-serving \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 3: Apply Chat Template ConfigMap

Apply the Qwen custom Jinja2 chat template ConfigMap (`gke/qwen-chat-template.yaml`):

```bash
kubectl apply -f gke/qwen-chat-template.yaml
```

---

## Step 4: Apply GKE Deployment Manifest

Deploy the vLLM server workload manifest (`gke/qwen3.6-27b-tpu.yaml`):

```bash
kubectl apply -f gke/qwen3.6-27b-tpu.yaml
```

### Manifest Overview (`gke/qwen3.6-27b-tpu.yaml`)

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qwen-model-cache
  namespace: qwen-serving
spec:
  storageClassName: hyperdisk-balanced-qwen
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen-serving
  namespace: qwen-serving
spec:
  replicas: 1
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
        cloud.google.com/gke-tpu-topology: 2x2
      containers:
        - name: vllm-tpu
          image: vllm/vllm-tpu:latest
          command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - --host=0.0.0.0
            - --port=8000
            - --model=Qwen/Qwen3.6-27B-FP8
            - --tensor-parallel-size=4
            - --gpu-memory-utilization=0.95
            - --max-num-seqs=128
            - --reasoning-parser=qwen3
            - --enable-prefix-caching
            - --kv-cache-dtype=fp8
            - --enable-chunked-prefill
          resources:
            limits:
              google.com/tpu: "4"
            requests:
              google.com/tpu: "4"
```

---

## Step 5: Monitor Pod Readiness & JAX Compilation

Monitor pod initialization and weight downloads:

```bash
kubectl get pods -n qwen-serving -w
```

Track logs during initial JAX/XLA graph compilation (compilation takes ~5–10 minutes for 32K context):

```bash
kubectl logs -n qwen-serving deployment/vllm-qwen-serving -f
```

Once the compilation completes, the log output will confirm readiness:

```text
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Step 6: Test Server Endpoints

1. Port-forward the GKE vLLM service:

```bash
kubectl port-forward -n qwen-serving svc/vllm-qwen-service-tp4 8000:8000
```

2. Submit a test request via `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3.6-27B-FP8",
        "messages": [{"role": "user", "content": "Write a Python function to check if a number is prime."}],
        "max_tokens": 256,
        "temperature": 0
    }'
```

---

## Step 7: Run Benchmarking Workload

Run the in-cluster agentic workload benchmark suite:

```bash
kubectl apply -f gke/benchmark-prep-script.yaml
kubectl apply -f gke/benchmark_agentic.yaml
```

View benchmark execution progress:

```bash
kubectl logs -n qwen-serving vllm-benchmark-agentic -f
```

Clean up after benchmark completion:

```bash
kubectl delete pod vllm-benchmark-agentic -n qwen-serving
```

---

## Benchmarking Results

### Measured Official uBench Results (Cloud TPU v6e-4)

```text
============ Serving Benchmark Result ============
Successful requests:                     320
Failed requests:                         0
Maximum request concurrency:             320
Request throughput (req/s):              28.07
Output token throughput (tok/s):         3233.91
Peak output token throughput (tok/s):    3233.91
Total Token throughput (tok/s):          6476.25
---------------Time to First Token----------------
Mean TTFT (ms):                          4064.88
Median TTFT (ms):                        4529.76
P99 TTFT (ms):                           8764.62
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.80
Median TPOT (ms):                        32.35
P99 TPOT (ms):                           34.24
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.83
Median ITL (ms):                         27.23
P99 ITL (ms):                            48.47
==================================================
```

### Performance & Concurrency Ladder

| Concurrency Level | Context Size | Mean TTFT | Mean TPOT | Request Throughput | Output Token Throughput | Total Throughput / Chip |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Concurrency = 10** | **27K Long Context** | **503.79 ms** | **16.81 ms** | **2.08 req/s** | **532.48 tok/s** | **133.12 tok/s/chip** |
| **Concurrency = 20** | **6.4K Medium Context** | **523.48 ms** | **21.76 ms** | **2.92 req/s** | **700.80 tok/s** | **175.20 tok/s/chip** |
| **Concurrency = 80** | **6.4K Medium Context** | **740.44 ms** | **30.17 ms** | **5.92 req/s** | **1,515.32 tok/s** | **378.83 tok/s/chip** |
| **Concurrency = 120** | **6.4K Medium Context** | **503.85 ms** | **30.17 ms** | **6.07 req/s** | **1,499.77 tok/s** | **374.94 tok/s/chip** |
| **Concurrency = 40** | **26.8K Long Context** | **2,329.58 ms** | **32.53 ms** | **2.90 req/s** | **741.24 tok/s** | **185.31 tok/s/chip** |
| **Concurrency = 80** | **26.8K Long Context** | **1,076.65 ms** | **41.45 ms** | **4.20 req/s** | **1,074.49 tok/s** | **268.62 tok/s/chip** |
| **Concurrency = 120** | **26.8K Long Context** | **827.27 ms** | **39.47 ms** | **4.36 req/s** | **1,116.49 tok/s** | **279.12 tok/s/chip** |
| **Concurrency = 320** | **uBench Standard Run** | **4,064.88 ms** | **30.80 ms** | **28.07 req/s** | **3,233.91 tok/s** | **404.24 tok/s/chip** |

---

## References

- Qwen3.6-27B-FP8 Hugging Face model card: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
- Reference TPU recipes: https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen3
- vLLM Cloud TPU documentation: https://docs.vllm.ai/projects/tpu/en/latest/
- Cloud TPU v6e documentation: https://cloud.google.com/tpu/docs/v6e
