# Serve Qwen3.5-35B-A3B with vLLM (TorchTPU) on Trillium TPU (Single-Host)

In this guide, we show how to serve Qwen3.5-35B-A3B (`Qwen/Qwen3.5-35B-A3B`, BF16 weights + FP8 KV Cache) with vLLM using the **TorchTPU** backend on Trillium (TPU v6e) using Google Kubernetes Engine (GKE).

We deploy it on a single **ct6e-standard-8t** node pool (`2x4` topology, 8 TPU v6e chips, 256 GB HBM total) using **Tensor Parallelism (TP) size of 8** and **Expert Parallelism (EP)**.

## Verified Models

The following Qwen3.5 Mixture-of-Experts (MoE) hybrid Gated DeltaNet + Attention model is verified for single-host deployment on TPU v6e with the vLLM TorchTPU backend.

| Model | Parameters | Min TPUs (Chips) | Topology | HuggingFace |
| :---- | :---- | :---- | :---- | :---- |
| Qwen3.5-35B-A3B | 35B MoE (3B active) | 8× (`ct6e-standard-8t`) | `2x4` (1 host) | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |

## Prerequisites

1. A GKE cluster with TPU v6e support.
2. `gcloud` CLI and `kubectl` installed and configured.
3. A Hugging Face account and access token with read access to `Qwen/Qwen3.5-35B-A3B`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="qwen35-35b-v6e-pool"
```

---

## Step 2: Create the Single-Host TPU Node Pool

Create a TPU v6e (Trillium) node pool with the `ct6e-standard-8t` machine type and `2x4` topology (1 host, 8 chips):

```shell
gcloud container node-pools create ${NODEPOOL_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --node-locations=${ZONE} \
    --num-nodes=1 \
    --machine-type=ct6e-standard-8t \
    --tpu-topology=2x4 \
    --cluster=${CLUSTER_NAME}
```

---

## Step 3: Setup Kubernetes Configurations

1. Configure `kubectl` to communicate with your cluster:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID}
```

2. Create the isolated namespace `vllm-qwen35-35b`:

```shell
kubectl create namespace vllm-qwen35-35b
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-qwen35-35b \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Apply the vLLM server manifest using [qwen3.5-35b-a3b-server.yaml](./qwen3.5-35b-a3b-server.yaml):

```shell
cd tpu-recipes/inference/trillium/vLLM/vLLM-TorchTPU/Qwen3.5-35B-A3B/single-host
kubectl apply -f qwen3.5-35b-a3b-server.yaml
```

Monitor the progress of the server deployment:

```shell
kubectl get pods -n vllm-qwen35-35b -w
```

Follow the server logs during model download and XLA compilation:

```shell
kubectl logs -n vllm-qwen35-35b deployment/vllm-qwen35-35b -c vllm-torchtpu -f
```

Once the server has loaded the model and completed compilation, you will see:

```text
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

---

## Step 5: Interact with the Model (Optional)

1. Port-forward the vLLM service to port `8000` on your local machine:

```shell
kubectl port-forward -n vllm-qwen35-35b service/vllm-qwen35-35b-service 8000:8000
```

2. Send a test chat completion request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3.5-35B-A3B",
        "messages": [
            {"role": "user", "content": "Explain how Mixture-of-Experts and Gated DeltaNet work together."}
        ],
        "max_tokens": 128,
        "temperature": 0.7
    }'
```

---

## Step 6: Run Serving Benchmarks

We use the `InferenceX` client (`benchmark_serving.py`) from SemiAnalysisAI to benchmark the running vLLM server across two standard workloads: **1K Input / 1K Output (`1k/1k`)** and **8K Input / 1K Output (`8k/1k`)**.

### Workload A: 1K Input / 1K Output (`1k/1k`)

Save the following manifest as `qwen3.5-35b-a3b-benchmark-1k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen35-35b-bench-1k1k
  namespace: vllm-qwen35-35b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: vllm-bench
      image: vllm/vllm-torchtpu:v0.29.0
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          apt-get update && apt-get install -y git curl
          while ! curl -sf http://vllm-qwen35-35b-service:8000/health; do
            sleep 30
            echo 'Waiting for vLLM server...'
          done
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex
          cd /workspace/inferencex
          git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd
          python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-qwen35-35b-service \
            --port=8000 \
            --model=Qwen/Qwen3.5-35B-A3B \
            --tokenizer=Qwen/Qwen3.5-35B-A3B \
            --dataset-name=random \
            --random-input-len=1024 \
            --random-output-len=1024 \
            --random-range-ratio=0.8 \
            --max-concurrency=64 \
            --num-prompts=320 \
            --num-warmups=4 \
            --ignore-eos
      env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
              optional: true
```

Apply it using `kubectl`:

```shell
kubectl apply -f qwen3.5-35b-a3b-benchmark-1k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-qwen35-35b qwen35-35b-bench-1k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3.5-35b-a3b-benchmark-1k1k.yaml
```

---

### Workload B: 8K Input / 1K Output (`8k/1k`)

Save the following manifest as `qwen3.5-35b-a3b-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen35-35b-bench-8k1k
  namespace: vllm-qwen35-35b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: vllm-bench
      image: vllm/vllm-torchtpu:v0.29.0
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          apt-get update && apt-get install -y git curl
          while ! curl -sf http://vllm-qwen35-35b-service:8000/health; do
            sleep 30
            echo 'Waiting for vLLM server...'
          done
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex
          cd /workspace/inferencex
          git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd
          python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-qwen35-35b-service \
            --port=8000 \
            --model=Qwen/Qwen3.5-35B-A3B \
            --tokenizer=Qwen/Qwen3.5-35B-A3B \
            --dataset-name=random \
            --random-input-len=8192 \
            --random-output-len=1024 \
            --random-range-ratio=0.8 \
            --max-concurrency=64 \
            --num-prompts=320 \
            --num-warmups=4 \
            --ignore-eos
      env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
              optional: true
```

Apply it using `kubectl`:

```shell
kubectl apply -f qwen3.5-35b-a3b-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-qwen35-35b qwen35-35b-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3.5-35b-a3b-benchmark-8k1k.yaml
```

---

### Example Benchmark Output

The benchmark client will print a table resembling the following:

```text
============ Serving Benchmark Result ============
Successful requests:                     xxx
Maximum request concurrency:             xxx
Benchmark duration (s):                  xxx
Total input tokens:                      xxx
Total generated tokens:                  xxx
Request throughput (req/s):              xxx
Output token throughput (tok/s):         xxx
Total Token throughput (tok/s):          xxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxx
Median TTFT (ms):                        xxx
P99 TTFT (ms):                           xxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx
Median TPOT (ms):                        xxx
P99 TPOT (ms):                           xxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxx
Median ITL (ms):                         xxx
P99 ITL (ms):                            xxx
----------------End-to-end Latency----------------
Mean E2EL (ms):                          xxx
Median E2EL (ms):                        xxx
P99 E2EL (ms):                           xxx
==================================================
```

---

## Step 7: Cleanup

1. Delete the vLLM server resources:

```shell
kubectl delete -f qwen3.5-35b-a3b-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
