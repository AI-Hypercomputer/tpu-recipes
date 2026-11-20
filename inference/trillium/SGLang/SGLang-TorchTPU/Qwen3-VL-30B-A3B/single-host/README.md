# Serve Qwen3-VL-30B-A3B-Instruct with SGLang (TorchTPU) on Trillium TPU (Single-Host)

In this guide, we show how to serve Qwen3-VL-30B-A3B-Instruct (`Qwen/Qwen3-VL-30B-A3B-Instruct`, BF16 weights + BF16 KV cache) with [SGLang](https://github.com/sgl-project/sglang) using the **[SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu)** backend on Trillium (TPU v6e) using Google Kubernetes Engine (GKE).

We deploy it on a single **ct6e-standard-8t** node pool (`2x4` topology, 8 TPU v6e chips, 256 GB HBM total) using **Tensor Parallelism (TP) size of 8** and **Expert Parallelism (EP) size of 8**.

## Verified Models

The following Qwen3-VL Mixture-of-Experts vision-language model is verified for single-host deployment on TPU v6e with the SGLang-TorchTPU backend.

| Model | Parameters | Min TPUs (Chips) | Topology | HuggingFace |
| :---- | :---- | :---- | :---- | :---- |
| Qwen3-VL-30B-A3B-Instruct | 30B MoE (3B active) + ViT | 8× (`ct6e-standard-8t`) | `2x4` (1 host) | [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) |

> [!NOTE]
> SGLang-TorchTPU `tpu-v0.1.0` documents TPU7x (Ironwood) as its primary target. This recipe adapts the upstream [`benchmark/tpu/recipe/qwen3_vl_30b_a3b/server.sh`](https://github.com/sgl-project/sglang-torchtpu/blob/tpu-v0.1.0/benchmark/tpu/recipe/qwen3_vl_30b_a3b/server.sh) recipe to Trillium's 32 GB HBM per chip (`--mem-fraction-static 0.8`, `--max-total-tokens 73728`) and is sized for up to 8 concurrent requests with a 9216-token context. Raise `--max-running-requests`, `--torch-compile-max-bs` and `--max-total-tokens` together to serve more concurrent requests.

## Prerequisites

1. A GKE cluster with TPU v6e support.
2. `gcloud` CLI and `kubectl` installed and configured.
3. A Hugging Face account and access token with read access to `Qwen/Qwen3-VL-30B-A3B-Instruct`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="qwen3-vl-30b-v6e-pool"
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

2. Create the isolated namespace `sglang-qwen3-vl-30b`:

```shell
kubectl create namespace sglang-qwen3-vl-30b
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=sglang-qwen3-vl-30b \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the SGLang Server Manifest

Apply the SGLang server manifest using [qwen3-vl-30b-a3b-server.yaml](./qwen3-vl-30b-a3b-server.yaml):

```shell
cd tpu-recipes/inference/trillium/SGLang/SGLang-TorchTPU/Qwen3-VL-30B-A3B/single-host
kubectl apply -f qwen3-vl-30b-a3b-server.yaml
```

Monitor the progress of the server deployment:

```shell
kubectl get pods -n sglang-qwen3-vl-30b -w
```

Follow the server logs during model download and TPU compilation (the vision encoder and decoder programs are compiled on first start, which takes about 35 minutes; later starts reuse the compile cache on the attached Hyperdisk):

```shell
kubectl logs -n sglang-qwen3-vl-30b deployment/sglang-qwen3-vl-30b -c sglang-torchtpu -f
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
kubectl port-forward -n sglang-qwen3-vl-30b service/sglang-qwen3-vl-30b-service 8000:8000
```

2. Send a test chat completion request with an image:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png"}},
                {"type": "text", "text": "Describe this image in one sentence."}
            ]
        }],
        "max_tokens": 128,
        "temperature": 0.7
    }'
```

---

## Step 6: Run Serving Benchmarks

We use SGLang's built-in serving benchmark (`python3 -m sglang.bench_serving`, shipped in the same image) with text-only random prompts against the running server across two workloads: **1K Input / 1K Output (`1k/1k`)** and **8K Input / 1K Output (`8k/1k`)**, both at 8 concurrent requests (the concurrency this configuration is sized for).

### Workload A: 1K Input / 1K Output (`1k/1k`)

Save the following manifest as `qwen3-vl-30b-a3b-benchmark-1k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-vl-30b-bench-1k1k
  namespace: sglang-qwen3-vl-30b
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
          while ! curl -sf http://sglang-qwen3-vl-30b-service:8000/health; do
            sleep 30
            echo "Waiting for sglang-qwen3-vl-30b-service:8000/health..."
          done
          python3 -m sglang.bench_serving \
            --backend sglang \
            --host sglang-qwen3-vl-30b-service \
            --port 8000 \
            --model Qwen/Qwen3-VL-30B-A3B-Instruct \
            --dataset-name random \
            --random-input-len 1024 \
            --random-output-len 1024 \
            --random-range-ratio 0.8 \
            --num-prompts 80 \
            --max-concurrency 8 \
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
kubectl apply -f qwen3-vl-30b-a3b-benchmark-1k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n sglang-qwen3-vl-30b qwen3-vl-30b-bench-1k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3-vl-30b-a3b-benchmark-1k1k.yaml
```

---

### Workload B: 8K Input / 1K Output (`8k/1k`)

Save the following manifest as `qwen3-vl-30b-a3b-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-vl-30b-bench-8k1k
  namespace: sglang-qwen3-vl-30b
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
          while ! curl -sf http://sglang-qwen3-vl-30b-service:8000/health; do
            sleep 30
            echo "Waiting for sglang-qwen3-vl-30b-service:8000/health..."
          done
          python3 -m sglang.bench_serving \
            --backend sglang \
            --host sglang-qwen3-vl-30b-service \
            --port 8000 \
            --model Qwen/Qwen3-VL-30B-A3B-Instruct \
            --dataset-name random \
            --random-input-len 8192 \
            --random-output-len 1024 \
            --random-range-ratio 0.8 \
            --num-prompts 80 \
            --max-concurrency 8 \
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
kubectl apply -f qwen3-vl-30b-a3b-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n sglang-qwen3-vl-30b qwen3-vl-30b-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3-vl-30b-a3b-benchmark-8k1k.yaml
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
kubectl delete -f qwen3-vl-30b-a3b-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
