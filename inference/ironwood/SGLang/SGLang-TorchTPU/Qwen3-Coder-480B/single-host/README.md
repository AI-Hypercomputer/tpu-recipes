# Serve Qwen3-Coder-480B-A35B-Instruct-FP8 with SGLang (TorchTPU) on Ironwood TPU (Single-Host)

In this guide, we show how to serve Qwen3-Coder-480B-A35B-Instruct-FP8 (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, FP8 weights + FP8 KV cache) with [SGLang](https://github.com/sgl-project/sglang) using the **[SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu)** backend on Ironwood (TPU7x) using Google Kubernetes Engine (GKE).

We deploy it on a single **tpu7x-standard-4t** node pool (`2x2x1` topology, 4 TPU7x chips / 8 TensorCores) using **Tensor Parallelism (TP) size of 8** and **Expert Parallelism (EP) size of 8**.

## Verified Models

The following model is verified for single-host deployment on TPU7x with the SGLang-TorchTPU backend.

| Model | Parameters | Min TPUs (Chips) | Topology | HuggingFace |
| :---- | :---- | :---- | :---- | :---- |
| Qwen3-Coder-480B-A35B-Instruct-FP8 | 480B MoE (35B active) | 4× (`tpu7x-standard-4t`) | `2x2x1` (1 host) | [Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) |

> [!NOTE]
> The server flags follow the upstream SGLang-TorchTPU recipe for this model, [`benchmark/tpu/recipe/qwen3_coder_480b/server.sh`](https://github.com/sgl-project/sglang-torchtpu/blob/tpu-v0.1.0/benchmark/tpu/recipe/qwen3_coder_480b/server.sh), sized for 8K-input / 1K-output requests at 64 concurrent sequences. For longer prompts raise `--context-length` and `--max-total-tokens` (64 × context length) together.

## Prerequisites

1. A GKE cluster with TPU7x (Ironwood) support.
2. `gcloud` CLI and `kubectl` installed and configured.
3. A Hugging Face account and access token with read access to `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="qwen3-coder-480b-tpu7x-pool"
```

---

## Step 2: Create the Single-Host TPU Node Pool

Create a TPU7x (Ironwood) node pool with the `tpu7x-standard-4t` machine type and `2x2x1` topology (1 host, 4 chips):

```shell
gcloud container node-pools create ${NODEPOOL_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --node-locations=${ZONE} \
    --num-nodes=1 \
    --machine-type=tpu7x-standard-4t \
    --tpu-topology=2x2x1 \
    --cluster=${CLUSTER_NAME}
```

---

## Step 3: Setup Kubernetes Configurations

1. Configure `kubectl` to communicate with your cluster:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID}
```

2. Create the isolated namespace `sglang-qwen3-coder-480b`:

```shell
kubectl create namespace sglang-qwen3-coder-480b
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=sglang-qwen3-coder-480b \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the SGLang Server Manifest

Apply the SGLang server manifest using [qwen3-coder-480b-server.yaml](./qwen3-coder-480b-server.yaml):

```shell
cd tpu-recipes/inference/ironwood/SGLang/SGLang-TorchTPU/Qwen3-Coder-480B/single-host
kubectl apply -f qwen3-coder-480b-server.yaml
```

Monitor the progress of the server deployment:

```shell
kubectl get pods -n sglang-qwen3-coder-480b -w
```

Follow the server logs during model download and TPU compilation (the first start downloads ~480 GB of weights to the attached Hyperdisk, then compiles the decode and prefill programs):

```shell
kubectl logs -n sglang-qwen3-coder-480b deployment/sglang-qwen3-coder-480b -c sglang-torchtpu -f
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
kubectl port-forward -n sglang-qwen3-coder-480b service/sglang-qwen3-coder-480b-service 8000:8000
```

2. Send a test chat completion request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "messages": [
            {"role": "user", "content": "Write a Python function that parses an ISO-8601 timestamp."}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }'
```

---

## Step 6: Run Serving Benchmarks

We use SGLang's built-in serving benchmark (`python3 -m sglang.bench_serving`, shipped in the same image) against the running server across two workloads: **1K Input / 1K Output (`1k/1k`)** and **8K Input / 1K Output (`8k/1k`)**, both at 64 concurrent requests.

### Workload A: 1K Input / 1K Output (`1k/1k`)

Save the following manifest as `qwen3-coder-480b-benchmark-1k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-coder-480b-bench-1k1k
  namespace: sglang-qwen3-coder-480b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: sglang-bench
      image: lmsysorg/sglang:v0.1.0-tpu
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          while ! curl -sf http://sglang-qwen3-coder-480b-service:8000/health; do
            sleep 30
            echo "Waiting for sglang-qwen3-coder-480b-service:8000/health..."
          done
          python3 -m sglang.bench_serving \
            --backend sglang \
            --host sglang-qwen3-coder-480b-service \
            --port 8000 \
            --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
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
kubectl apply -f qwen3-coder-480b-benchmark-1k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n sglang-qwen3-coder-480b qwen3-coder-480b-bench-1k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3-coder-480b-benchmark-1k1k.yaml
```

---

### Workload B: 8K Input / 1K Output (`8k/1k`)

Save the following manifest as `qwen3-coder-480b-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-coder-480b-bench-8k1k
  namespace: sglang-qwen3-coder-480b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: sglang-bench
      image: lmsysorg/sglang:v0.1.0-tpu
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          while ! curl -sf http://sglang-qwen3-coder-480b-service:8000/health; do
            sleep 30
            echo "Waiting for sglang-qwen3-coder-480b-service:8000/health..."
          done
          python3 -m sglang.bench_serving \
            --backend sglang \
            --host sglang-qwen3-coder-480b-service \
            --port 8000 \
            --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
            --dataset-name random \
            --random-input-len 8192 \
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
kubectl apply -f qwen3-coder-480b-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n sglang-qwen3-coder-480b qwen3-coder-480b-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3-coder-480b-benchmark-8k1k.yaml
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

1. Delete the SGLang server resources:

```shell
kubectl delete -f qwen3-coder-480b-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
