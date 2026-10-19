# Serve Qwen3-32B with vLLM (TorchTPU) on Trillium TPU (Single-Host)

In this guide, we show how to serve Qwen3-32B (`Qwen/Qwen3-32B`, BF16 weights + FP8 KV Cache) with vLLM using the **TorchTPU** backend on Trillium (TPU v6e) using Google Kubernetes Engine (GKE).

We deploy it on a single **ct6e-standard-8t** node pool (`2x4` topology, 8 TPU v6e chips, 256 GB HBM total) using **Tensor Parallelism (TP) size of 8**.

## Verified Models

The following Qwen3 model is verified for single-host deployment on TPU v6e with the vLLM TorchTPU backend.

| Model | Parameters | Min TPUs (Chips) | Topology | HuggingFace |
| :---- | :---- | :---- | :---- | :---- |
| Qwen3-32B | 32.8B Dense | 8× (`ct6e-standard-8t`) | `2x4` (1 host) | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |

> [!TIP]
> `Qwen/Qwen3-32B` (~65 GB BF16 weights) can also fit on a 4-chip `ct6e-standard-4t` (`2x2`) node by changing `cloud.google.com/gke-tpu-topology` to `"2x2"`, `google.com/tpu` to `"4"`, and `--tensor-parallel-size=4`. The `2x4` (`TP=8`) configuration is used here for higher KV cache capacity and lower latency.

## Prerequisites

1. A GKE cluster with TPU v6e support.
2. `gcloud` CLI and `kubectl` installed and configured.
3. A Hugging Face account and access token with read access to `Qwen/Qwen3-32B`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="qwen3-32b-v6e-pool"
export TORCH_TPU_VLLM_IMAGE="<YOUR_TORCH_TPU_VLLM_IMAGE>"
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

2. Create the isolated namespace `vllm-qwen3-32b`:

```shell
kubectl create namespace vllm-qwen3-32b
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-qwen3-32b \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Set your TorchTPU vLLM container image and apply the server manifest using [qwen3-32b-server.yaml](./qwen3-32b-server.yaml):

```shell
cd tpu-recipes/inference/trillium/vLLM/vLLM-TorchTPU/Qwen3-32B/single-host
sed -i "s|REPLACE_TORCH_TPU_VLLM_IMAGE:latest|${TORCH_TPU_VLLM_IMAGE}|g" qwen3-32b-server.yaml
kubectl apply -f qwen3-32b-server.yaml
```

Monitor the progress of the server deployment:

```shell
kubectl get pods -n vllm-qwen3-32b -w
```

Follow the server logs during model download and XLA compilation:

```shell
kubectl logs -n vllm-qwen3-32b deployment/vllm-qwen3-32b -c vllm-torchtpu -f
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
kubectl port-forward -n vllm-qwen3-32b service/vllm-qwen3-32b-service 8000:8000
```

2. Send a test chat completion request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-32B",
        "messages": [
            {"role": "user", "content": "Explain how tensor parallelism works on TPU."}
        ],
        "max_tokens": 128,
        "temperature": 0.7
    }'
```

---

## Step 6: Run Serving Benchmarks

We use the `InferenceX` client (`benchmark_serving.py`) from SemiAnalysisAI to benchmark the running vLLM server across three standard workloads: **1K Input / 1K Output (`1k/1k`)**, **1K Input / 8K Output (`1k/8k`)**, and **8K Input / 1K Output (`8k/1k`)**.

### Workload A: 1K Input / 1K Output (`1k/1k`)

Save the following manifest as `qwen3-32b-benchmark-1k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-32b-bench-1k1k
  namespace: vllm-qwen3-32b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: vllm-bench
      # Set to your TorchTPU vLLM container image.
      image: REPLACE_TORCH_TPU_VLLM_IMAGE:latest
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          apt-get update && apt-get install -y git curl
          while ! curl -sf http://vllm-qwen3-32b-service:8000/health; do
            sleep 15
            echo "Waiting for vllm-qwen3-32b-service:8000/health..."
          done
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
          cd /workspace/inferencex && \
          git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
          python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-qwen3-32b-service \
            --port=8000 \
            --model=Qwen/Qwen3-32B \
            --tokenizer=Qwen/Qwen3-32B \
            --dataset-name=random \
            --random-input-len=1024 \
            --random-output-len=1024 \
            --random-range-ratio=0.8 \
            --num-prompts=320 \
            --max-concurrency=64 \
            --num-warmups=16 \
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
sed -i "s|REPLACE_TORCH_TPU_VLLM_IMAGE:latest|${TORCH_TPU_VLLM_IMAGE}|g" qwen3-32b-benchmark-1k1k.yaml
kubectl apply -f qwen3-32b-benchmark-1k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-qwen3-32b qwen3-32b-bench-1k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3-32b-benchmark-1k1k.yaml
```

---

### Workload B: 1K Input / 8K Output (`1k/8k`)

Save the following manifest as `qwen3-32b-benchmark-1k8k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-32b-bench-1k8k
  namespace: vllm-qwen3-32b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: vllm-bench
      # Set to your TorchTPU vLLM container image.
      image: REPLACE_TORCH_TPU_VLLM_IMAGE:latest
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          apt-get update && apt-get install -y git curl
          while ! curl -sf http://vllm-qwen3-32b-service:8000/health; do
            sleep 15
            echo "Waiting for vllm-qwen3-32b-service:8000/health..."
          done
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
          cd /workspace/inferencex && \
          git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
          python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-qwen3-32b-service \
            --port=8000 \
            --model=Qwen/Qwen3-32B \
            --tokenizer=Qwen/Qwen3-32B \
            --dataset-name=random \
            --random-input-len=1024 \
            --random-output-len=8192 \
            --random-range-ratio=0.8 \
            --num-prompts=180 \
            --max-concurrency=64 \
            --num-warmups=16 \
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
sed -i "s|REPLACE_TORCH_TPU_VLLM_IMAGE:latest|${TORCH_TPU_VLLM_IMAGE}|g" qwen3-32b-benchmark-1k8k.yaml
kubectl apply -f qwen3-32b-benchmark-1k8k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-qwen3-32b qwen3-32b-bench-1k8k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3-32b-benchmark-1k8k.yaml
```

---

### Workload C: 8K Input / 1K Output (`8k/1k`)

Save the following manifest as `qwen3-32b-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-32b-bench-8k1k
  namespace: vllm-qwen3-32b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: vllm-bench
      # Set to your TorchTPU vLLM container image.
      image: REPLACE_TORCH_TPU_VLLM_IMAGE:latest
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          apt-get update && apt-get install -y git curl
          while ! curl -sf http://vllm-qwen3-32b-service:8000/health; do
            sleep 15
            echo "Waiting for vllm-qwen3-32b-service:8000/health..."
          done
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
          cd /workspace/inferencex && \
          git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
          python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-qwen3-32b-service \
            --port=8000 \
            --model=Qwen/Qwen3-32B \
            --tokenizer=Qwen/Qwen3-32B \
            --dataset-name=random \
            --random-input-len=8192 \
            --random-output-len=1024 \
            --random-range-ratio=0.8 \
            --num-prompts=180 \
            --max-concurrency=64 \
            --num-warmups=16 \
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
sed -i "s|REPLACE_TORCH_TPU_VLLM_IMAGE:latest|${TORCH_TPU_VLLM_IMAGE}|g" qwen3-32b-benchmark-8k1k.yaml
kubectl apply -f qwen3-32b-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-qwen3-32b qwen3-32b-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3-32b-benchmark-8k1k.yaml
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
kubectl delete -f qwen3-32b-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
