# Serve Gemma 4 26B with vLLM (TorchTPU) on Trillium TPU

In this guide, we show how to serve Gemma 4 26B (`google/gemma-4-26B-A4B-it`, BF16 weights + FP8 KV Cache) with vLLM using the **TorchTPU** backend on Trillium (TPU v6e) using GKE.

We deploy it on a single **ct6e-standard-8t** node pool (`2x4` topology, 8 TPU v6e chips) using **Tensor Parallelism (TP) size of 8**.

## Verified Models

The following Gemma 4 model is verified for deployment on TPU with the vLLM TorchTPU backend.

| Model | Parameters | Min TPUs (Chips) | HuggingFace |
| :---- | :---- | :---- | :---- |
| Gemma 4 26B-A4B IT | 26B (4B active) | 8× | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |

## Prerequisites

1. A GKE cluster with TPU v6e support.
2. GCloud CLI and Kubectl installed and configured.
3. A Hugging Face account and access token with access to `google/gemma-4-26B-A4B-it`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="gemma4-tpu-pool"
```

---

## Step 2: Create the TPU Node Pool

Create a TPU v6e (Trillium) nodepool with the `ct6e-standard-8t` machine type and `2x4` topology. This machine type provides 8 chips.

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

2. Create the Namespace `vllm-gemma4` if you want to run it in a isolated namespace (recommended):

```shell
kubectl create namespace vllm-gemma4
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-gemma4 \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Apply the vLLM server manifest using the provided [gemma4-server.yaml](./gemma4-server.yaml) file in this directory:

```shell
cd tpu-recipes/inference/trillium/vLLM/vLLM-TorchTPU/Gemma4-26B
kubectl apply -f gemma4-server.yaml
```

> [!NOTE]
> By default, `gemma4-server.yaml` is configured with `--max-model-len=9216` and `--max-num-seqs=64` (used for the `1k/8k` and `8k/1k` benchmarks, and compatible with all three workloads). To reproduce the peak throughput for the `1k/500` workload, update these flags in `gemma4-server.yaml` to `--max-model-len=1524` and `--max-num-seqs=256` before running Workload A.

Monitor the progress of the server deployment:

```shell
kubectl get pods -n vllm-gemma4 -w
```

You can check the server logs using:

```shell
kubectl logs -n vllm-gemma4 deployment/vllm-gemma4 -c vllm-torchtpu -f
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
kubectl port-forward -n vllm-gemma4 service/vllm-gemma4-service 8000:8000
```

2. Run a curl request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-26B-A4B-it",
        "messages": [
            {"role": "user", "content": "Hello! Tell me a joke."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

---

## Step 6: Run Serving Benchmarks

We use the `InferenceX` client from SemiAnalysisAI to perform serving benchmarks against our running vLLM server. We test three workloads: **1K Input / 500 Output**, **1K Input / 8K Output** and **8K Input / 1K Output**.

### Workload A: 1K Input / 500 Output (1k/500)

Save the following manifest as `gemma4-benchmark-1k500.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gemma4-bench-1k500
  namespace: vllm-gemma4
spec:
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-gemma4-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
      cd /workspace/inferencex && \
      git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
      python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-gemma4-service \
        --port=8000 \
        --model=google/gemma-4-26B-A4B-it \
        --tokenizer=google/gemma-4-26B-A4B-it \
        --dataset-name=random \
        --random-input-len=1024 \
        --random-output-len=500 \
        --random-range-ratio=0.8 \
        --num-prompts=2048 \
        --num-warmups=4 \
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
kubectl apply -f gemma4-benchmark-1k500.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-gemma4 gemma4-bench-1k500 -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f gemma4-benchmark-1k500.yaml
```

---

### Workload B: 1K Input / 8K Output (1k/8k)

Save the following manifest as `gemma4-benchmark-1k8k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gemma4-bench-1k8k
  namespace: vllm-gemma4
spec:
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-gemma4-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
      cd /workspace/inferencex && \
      git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
      python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-gemma4-service \
        --port=8000 \
        --model=google/gemma-4-26B-A4B-it \
        --tokenizer=google/gemma-4-26B-A4B-it \
        --dataset-name=random \
        --random-input-len=1024 \
        --random-output-len=8192 \
        --random-range-ratio=0.8 \
        --num-prompts=128 \
        --num-warmups=2 \
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
kubectl apply -f gemma4-benchmark-1k8k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-gemma4 gemma4-bench-1k8k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f gemma4-benchmark-1k8k.yaml
```

---

### Workload C: 8K Input / 1K Output (8k/1k)

Save the following manifest as `gemma4-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gemma4-bench-8k1k
  namespace: vllm-gemma4
spec:
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-gemma4-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
      cd /workspace/inferencex && \
      git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
      python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-gemma4-service \
        --port=8000 \
        --model=google/gemma-4-26B-A4B-it \
        --tokenizer=google/gemma-4-26B-A4B-it \
        --dataset-name=random \
        --random-input-len=8192 \
        --random-output-len=1024 \
        --random-range-ratio=0.8 \
        --num-prompts=180 \
        --num-warmups=2 \
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
kubectl apply -f gemma4-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-gemma4 gemma4-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f gemma4-benchmark-8k1k.yaml
```

---

### Example Benchmark Output

The benchmark client will print a table resembling the following:

```
============ Serving Benchmark Result ============
Successful requests:                     xxx
Failed requests:                         xxx
Benchmark duration (s):                  xxx
Total input tokens:                      xxx
Total generated tokens:                  xxx
Request throughput (req/s):              xx
Output token throughput (tok/s):         xxx
Peak output token throughput (tok/s):    xxx
Peak concurrent requests:                xxx
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

When you are finished with the benchmark and no longer need the server, you can tear down the resources:

1. Delete the vLLM server:

```shell
kubectl delete -f gemma4-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
