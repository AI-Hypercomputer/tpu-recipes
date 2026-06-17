# Serve Qwen3.5 397B with vLLM on Ironwood TPU

In this guide, we show how to serve Qwen3.5 397B models (e.g., `Qwen/Qwen3.5-397B-A17B-FP8`) with vLLM on Ironwood (TPU v7x) using GKE.

Since Qwen3.5-397B is a very large mixture-of-experts model (~400GB in FP8), it requires a multi-chip setup. We deploy it on a single **tpu7x-standard-4t** node pool (4 TPU v7x chips, 8 TensorCores) using **Tensor Parallelism (TP) size of 8**.

## Verified Models

The following larger Qwen 3.5 models are verified for deployment on TPU.

### Verified Models

| Model | Parameters | Min TPUs (Chips) | HuggingFace |
| :---- | :---- | :---- | :---- |
| Qwen 3.5 397B FP8 | 397B (17B active) | 4× | [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) |

## Prerequisites

1. A GKE cluster with TPU v7x support (GKE version `1.34.0-gke.2201000` or later).
2. GCloud CLI and Kubectl installed and configured.
3. A Hugging Face account and access token (needed if Qwen3.5-397B-A17B-FP8 requires authentication or for general HF hub usage).

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="qwen3-5-tpu-pool"
```

---

## Step 2: Create the TPU Node Pool

Create a TPU v7x (Ironwood) nodepool with the `tpu7x-standard-4t` machine type. This machine type provides 4 chips.

```shell
gcloud container node-pools create ${NODEPOOL_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --node-locations=${ZONE} \
    --num-nodes=1 \
    --machine-type=tpu7x-standard-4t \
    --cluster=${CLUSTER_NAME}
```

---

## Step 3: Setup Kubernetes Configurations

1. Configure `kubectl` to communicate with your cluster:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${ZONE} --project=${PROJECT_ID}
```

2. Create the Namespace `qwen3-5-test` if you want to run it in a isolated namespace (recommended):

```shell
kubectl create namespace qwen3-5-test
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=qwen3-5-test \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Apply the vLLM server manifest using the provided `qwen3_5-server.yaml` file in this directory:

```shell
kubectl apply -f qwen3_5-server.yaml
```

Monitor the progress of the server deployment:

```shell
kubectl get pods -n qwen3-5-test -w
```

You can check the server logs using:

```shell
kubectl logs -n qwen3-5-test deployment/vllm-qwen3-5 -c vllm-tpu -f
```

Once the server has successfully loaded the model, you should see logs like:

```
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

---

## Step 5: Interact with the Model (Optional)

You can test the server using a port forward and curl:

1. Port forward the vLLM service to port `8000` on your local machine:

```shell
kubectl port-forward -n qwen3-5-test service/vllm-qwen3-5-service 8000:8000
```

2. Run a curl request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3.5-397B-A17B-FP8",
        "messages": [
            {"role": "user", "content": "Hello! Tell me a joke."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

---

## Step 6: Run Serving Benchmarks

We use the `InferenceX` client from SemiAnalysisAI to perform serving benchmarks against our running vLLM server. We test two workloads: **1K Input / 8K Output** and **8K Input / 1K Output**.

### Workload A: 1K Input / 8K Output (1k/8k)

Save the following manifest as `qwen3_5-benchmark-1k8k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-5-bench-1k8k
  namespace: qwen3-5-test
spec:
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-tpu:nightly-20260616-df47e95-a30addc
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl http://vllm-qwen3-5-service:8000/ping; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /ubench/inferencex && \
      cd /ubench/inferencex && \
      git checkout 89ce6098ef2bc4576a735c43f39c7d972b091cfc && \
      python3 /ubench/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-qwen3-5-service \
        --port=8000 \
        --model=Qwen/Qwen3.5-397B-A17B-FP8 \
        --tokenizer=Qwen/Qwen3.5-397B-A17B-FP8 \
        --dataset-name=random \
        --random-input-len=1024 \
        --random-output-len=8192 \
        --random-range-ratio=0.8 \
        --num-prompts=320 \
        --max-concurrency=64 \
        --ignore-eos
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          key: hf_api_token
          name: hf-secret
```

Apply it using `kubectl`:

```shell
kubectl apply -f qwen3_5-benchmark-1k8k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n qwen3-5-test qwen3-5-bench-1k8k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3_5-benchmark-1k8k.yaml
```

---

### Workload B: 8K Input / 1K Output (8k/1k)

Save the following manifest as `qwen3_5-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-5-bench-8k1k
  namespace: qwen3-5-test
spec:
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-tpu:nightly-20260616-df47e95-a30addc
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl http://vllm-qwen3-5-service:8000/ping; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /ubench/inferencex && \
      cd /ubench/inferencex && \
      git checkout 89ce6098ef2bc4576a735c43f39c7d972b091cfc && \
      python3 /ubench/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-qwen3-5-service \
        --port=8000 \
        --model=Qwen/Qwen3.5-397B-A17B-FP8 \
        --tokenizer=Qwen/Qwen3.5-397B-A17B-FP8 \
        --dataset-name=random \
        --random-input-len=8192 \
        --random-output-len=1024 \
        --random-range-ratio=0.8 \
        --num-prompts=320 \
        --max-concurrency=64 \
        --ignore-eos
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          key: hf_api_token
          name: hf-secret
```

Apply it using `kubectl`:

```shell
kubectl apply -f qwen3_5-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n qwen3-5-test qwen3-5-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f qwen3_5-benchmark-8k1k.yaml
```

---

### Example Benchmark Output

The benchmark client will print a table resembling the following:

```
============ Serving Benchmark Result ============
Successful requests:                     320
Failed requests:                         0
Benchmark duration (s):                  xxx
Total input tokens:                      xxx
Total generated tokens:                  xxx
Request throughput (req/s):              xx
Output token throughput (tok/s):         xxx
Peak output token throughput (tok/s):    xxx
Peak concurrent requests:                64.00
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
==================================================
```

Workload (input tokens/output tokens) | Output Token Throughput (tok/s) Per Chip
:------- | :---------------------------------------
1k/8k    | 2436.35 tok/s (609.09 tok/s/chip)
8k/1k    | 1513.05 tok/s (378.26 tok/s/chip)

**Note**: These benchmark results are based on the `InferenceX` client. The development team is continuously improving and optimizing performance; as such, these results are subject to change, and improved or optimized figures may be published in the future.

---

## Step 7: Cleanup

When you are finished with the benchmark and no longer need the server, you can tear down the resources:

1. Delete the vLLM server:

```shell
kubectl delete -f qwen3_5-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
