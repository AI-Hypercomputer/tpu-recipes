# Serve Kimi-K3.1-Pro with SGLang (TorchTPU) on Ironwood TPU (Multi-Host)

In this guide, we show how to serve Kimi-K3.1-Pro (`moonshotai/Kimi-K3.1-Pro`, hybrid DeepSeek Sparse Attention MLA + KDA linear-attention MoE model) with [SGLang](https://github.com/sgl-project/sglang) using the **[SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu)** backend on Ironwood (TPU7x) using Google Kubernetes Engine (GKE).

We deploy it across a multi-host **tpu7x-standard-4t** slice (`2x2x4` topology, 4 hosts, 16 TPU7x chips / 32 TensorCores) as a Kubernetes **[LeaderWorkerSet](https://github.com/kubernetes-sigs/lws)** (`size: 4`), using **Tensor Parallelism (TP) size of 32** and **Expert Parallelism (EP) size of 8** with `pallas_tpu_mla_sparse` attention and `fp8_e4m3` KV cache.

## Verified Models

The following model is verified for multi-host deployment on TPU7x with the SGLang-TorchTPU backend.

| Model | Architecture / Quantization | Min TPUs (Chips) | Topology | HuggingFace |
| :---- | :---- | :---- | :---- | :---- |
| Kimi-K3.1-Pro | Hybrid Sparse-MLA + KDA MoE, FP8 / `fp8_e4m3` KV | 16× (`tpu7x-standard-4t`) | `2x2x4` (4 hosts) | [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) |

> [!NOTE]
> - The server flags follow the upstream SGLang-TorchTPU Kimi-K3.1 recipe (`benchmark/tpu/recipe/configs/kimi_k3_1/server.py`): `--context-length 491520` (480K context), `--max-total-tokens 4000000`, `--chunked-prefill-size 4096`, and `--max-running-requests 64`.
> - By default, [`kimi-k3-1-pro-server.yaml`](./kimi-k3-1-pro-server.yaml) passes `--disable-radix-cache` for synthetic random-token benchmarks (`1k/1k` and `8k/1k` in Step 6). For multi-turn or agentic coding workloads with shared prefix history, replace `--disable-radix-cache` with host-DRAM hierarchical KV caching (`--enable-hierarchical-cache --hicache-io-backend raiden --hicache-mem-layout page_first_fused --hicache-write-policy write_through --hicache-ratio 2 --enable-cache-report`).

## Prerequisites

1. `gcloud` CLI and `kubectl` installed and configured.
2. [Cluster Toolkit](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x#deploy-tpu-7x-cluster) installed for provisioning a GKE cluster with TPU7x (Ironwood) support.
3. A Hugging Face account and access token with read access to `moonshotai/Kimi-K3.1-Pro`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export LOCATION="<YOUR_CLUSTER_LOCATION>"
```

---

## Step 2: Create the GKE Cluster and Install LeaderWorkerSet

Create your TPU7x cluster with [Cluster Toolkit](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x#deploy-tpu-7x-cluster), starting from the `examples/gke-tpu-7x/gke-tpu-7x.yaml` blueprint. Set these variables in your deployment file to provision one 4-host `2x2x4` slice (16 chips / 32 TensorCores):

- `tpu_topology: 2x2x4`, `num_slices: 1`

This recipe runs the 4-host server as a [LeaderWorkerSet](https://lws.sigs.k8s.io/docs/installation/), which Cluster Toolkit does not install by default. After creating the cluster, fetch credentials and install the LeaderWorkerSet controller:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${LOCATION} --project=${PROJECT_ID}
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v1.0.0/manifests.yaml
```

---

## Step 3: Setup Kubernetes Configurations

1. Create the isolated namespace `sglang-kimi-k3-1-pro`:

```shell
kubectl create namespace sglang-kimi-k3-1-pro
```

2. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=sglang-kimi-k3-1-pro \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the SGLang Server Manifest

Apply the multi-host SGLang server manifest using [`kimi-k3-1-pro-server.yaml`](./kimi-k3-1-pro-server.yaml):

```shell
cd tpu-recipes/inference/ironwood/SGLang/SGLang-TorchTPU/Kimi-K3.1-Pro/multi-host
kubectl apply -f kimi-k3-1-pro-server.yaml
```

The manifest creates a `LeaderWorkerSet` of 4 pods (one on each TPU host of the `2x2x4` slice, each backed by an ephemeral 2500 GiB Hyperdisk Balanced volume) and a `ClusterIP` Service routing port `8000` to rank `0` (`sglang-kimi-k3-1-pro-0`).

Monitor the progress of the server pods:

```shell
kubectl get pods -n sglang-kimi-k3-1-pro -w
```

Follow the rank-0 leader pod logs during model download and TPU compilation (first start downloads the ~1 TB checkpoint to each host's Hyperdisk, then precompiles the decode and chunked-prefill XLA programs across all 32 TensorCores):

```shell
kubectl logs -n sglang-kimi-k3-1-pro sglang-kimi-k3-1-pro-0 -c sglang-torchtpu -f
```

Once the server has loaded the model and completed compilation, you will see:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
The server is fired up and ready to roll!
```

---

## Step 5: Interact with the Model (Optional)

1. Port-forward the SGLang service to port `8000` on your local machine:

```shell
kubectl port-forward -n sglang-kimi-k3-1-pro service/sglang-kimi-k3-1-pro-service 8000:8000
```

2. Send a test chat completion request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "moonshotai/Kimi-K3.1-Pro",
        "messages": [
            {"role": "user", "content": "Write a Python function that parses an ISO-8601 timestamp."}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }'
```

---

## Step 6: Run Serving Benchmarks

We use SGLang's built-in serving benchmark (`python3 -m sglang.bench_serving`, shipped in the same image) against the running server across two workloads: **1K Input / 1K Output (`1k/1k`)** at concurrency 64 and **8K Input / 1K Output (`8k/1k`)** at concurrency 32.

### Workload A: 1K Input / 1K Output (`1k/1k`)

Save the following manifest as `kimi-k3-1-pro-benchmark-1k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kimi-k3-1-pro-bench-1k1k
  namespace: sglang-kimi-k3-1-pro
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: sglang-bench
      image: lmsysorg/sglang:dev-kimi-k31-pro-tpu
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          while ! curl -sf http://sglang-kimi-k3-1-pro-service:8000/health; do
            sleep 30
            echo "Waiting for sglang-kimi-k3-1-pro-service:8000/health..."
          done
          python3 -m sglang.bench_serving \
            --backend sglang \
            --host sglang-kimi-k3-1-pro-service \
            --port 8000 \
            --model moonshotai/Kimi-K3.1-Pro \
            --trust-remote-code \
            --dataset-name random \
            --random-input-len 1024 \
            --random-output-len 1024 \
            --random-range-ratio 0.8 \
            --num-prompts 320 \
            --max-concurrency 64 \
            --disable-tqdm
      env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
              optional: true
```

Apply it using `kubectl`:

```shell
kubectl apply -f kimi-k3-1-pro-benchmark-1k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n sglang-kimi-k3-1-pro kimi-k3-1-pro-bench-1k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f kimi-k3-1-pro-benchmark-1k1k.yaml
```

---

### Workload B: 8K Input / 1K Output (`8k/1k`)

Save the following manifest as `kimi-k3-1-pro-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kimi-k3-1-pro-bench-8k1k
  namespace: sglang-kimi-k3-1-pro
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: sglang-bench
      image: lmsysorg/sglang:dev-kimi-k31-pro-tpu
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          while ! curl -sf http://sglang-kimi-k3-1-pro-service:8000/health; do
            sleep 30
            echo "Waiting for sglang-kimi-k3-1-pro-service:8000/health..."
          done
          python3 -m sglang.bench_serving \
            --backend sglang \
            --host sglang-kimi-k3-1-pro-service \
            --port 8000 \
            --model moonshotai/Kimi-K3.1-Pro \
            --trust-remote-code \
            --dataset-name random \
            --random-input-len 8192 \
            --random-output-len 1024 \
            --random-range-ratio 0.8 \
            --num-prompts 96 \
            --max-concurrency 32 \
            --disable-tqdm
      env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
              optional: true
```

Apply it using `kubectl`:

```shell
kubectl apply -f kimi-k3-1-pro-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n sglang-kimi-k3-1-pro kimi-k3-1-pro-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f kimi-k3-1-pro-benchmark-8k1k.yaml
```

---

### Example Benchmark Output

The benchmark client will print a table resembling the following:

```text
============ Serving Benchmark Result ============
Backend:                                 sglang
Successful requests:                     xxx
Benchmark duration (s):                  xxx
Total input tokens:                      xxx
Total generated tokens:                  xxx
Request throughput (req/s):              xxx
Input token throughput (tok/s):          xxx
Output token throughput (tok/s):         xxx
Total token throughput (tok/s):          xxx
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   xxx
Median E2E Latency (ms):                 xxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxx
Median TTFT (ms):                        xxx
P99 TTFT (ms):                           xxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx
Median TPOT (ms):                        xxx
P99 TPOT (ms):                           xxx
---------------Inter-Token Latency----------------
Mean ITL (ms):                           xxx
Median ITL (ms):                         xxx
P99 ITL (ms):                            xxx
==================================================
```

---

## Step 7: Cleanup

1. Delete the SGLang server resources (including the ephemeral Hyperdisk volumes):

```shell
kubectl delete -f kimi-k3-1-pro-server.yaml
```

2. Delete the namespace:

```shell
kubectl delete namespace sglang-kimi-k3-1-pro
```
