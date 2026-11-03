# Serve DeepSeek-V4-Flash with vLLM (TorchTPU) on Ironwood TPU

In this guide, we show how to serve DeepSeek-V4-Flash (`deepseek-ai/DeepSeek-V4-Flash`, Mixture-of-Experts model with Multi-head Latent Attention) with vLLM using the **TorchTPU** backend on Ironwood (TPU v7x) using GKE.

We deploy it on a single **tpu7x-standard-4t** node pool (`2x2x1` topology, 4 TPU v7x chips, 8 TensorCores) using **Data Parallelism (DP) size of 8**, **Expert Parallelism (EP) size of 8**, and **Tensor Parallelism (TP) size of 1**, with `deepseek_v4_fp8` quantization, load-time FP4 MoE weight requantization (`W4A8`), and `fp8_e4m3` KV cache.

## Verified Models

The following DeepSeek-V4-Flash model is verified for deployment on TPU with the vLLM TorchTPU backend.

| Model | Quantization | Min TPUs (Chips) | Topology | HuggingFace |
| :---- | :---- | :---- | :---- | :---- |
| DeepSeek-V4-Flash | `deepseek_v4_fp8` (W4A8 MoE), `fp8_e4m3` KV | 4× (`tpu7x-standard-4t`) | `2x2x1` (1 host) | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |

## Prerequisites

1. A GKE cluster with TPU v7x support (GKE version `1.34.0-gke.2201000` or later).
2. GCloud CLI and Kubectl installed and configured.
3. A Hugging Face account and access token with access to `deepseek-ai/DeepSeek-V4-Flash`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="dsv4-flash-tpu-pool"
```

---

## Step 2: Create the TPU Node Pool

Create a TPU v7x (Ironwood) node pool with the `tpu7x-standard-4t` machine type. This machine type provides 4 chips (8 logical TensorCore devices):

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
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID}
```

2. Create the Namespace `vllm-deepseek-v4-flash` if you want to run it in an isolated namespace (recommended):

```shell
kubectl create namespace vllm-deepseek-v4-flash
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-deepseek-v4-flash \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Apply the vLLM server manifest using the provided [deepseek_v4_flash-server.yaml](./deepseek_v4_flash-server.yaml) file in this directory:

```shell
cd inference/ironwood/vLLM/vLLM-TorchTPU/DeepSeek-V4-Flash
kubectl apply -f deepseek_v4_flash-server.yaml
```

Monitor the progress of the server deployment:

```shell
kubectl get pods -n vllm-deepseek-v4-flash -w
```

You can check the server logs using:

```shell
kubectl logs -n vllm-deepseek-v4-flash deployment/vllm-deepseek-v4-flash -c vllm-torchtpu -f
```

Once the server has successfully loaded and compiled the model, you should see logs like:

```
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

---

## Step 5: Interact with the Model (Optional)

You can test the server using a port forward and `curl`:

1. Port forward the vLLM service to port `8000` on your local machine:

```shell
kubectl port-forward -n vllm-deepseek-v4-flash service/vllm-deepseek-v4-flash-service 8000:8000
```

2. Run a `curl` request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "messages": [
            {"role": "user", "content": "Hello! Tell me a joke."}
        ],
        "max_tokens": 100,
        "temperature": 0.0
    }'
```

---

## Step 6: Run Serving Benchmarks

We use `vllm bench serve` to evaluate serving throughput and latency against our running vLLM server for an **8K Input / 1K Output (8k/1k)** workload.

### Workload: 8K Input / 1K Output (8k/1k)

Save the following manifest as `deepseek_v4_flash-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dsv4-flash-bench-8k1k
  namespace: vllm-deepseek-v4-flash
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-deepseek-v4-flash-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      vllm bench serve \
        --backend=vllm \
        --base-url=http://vllm-deepseek-v4-flash-service:8000 \
        --model=deepseek-ai/DeepSeek-V4-Flash \
        --tokenizer=deepseek-ai/DeepSeek-V4-Flash \
        --trust-remote-code \
        --dataset-name=random \
        --random-input-len=8192 \
        --random-output-len=1024 \
        --num-prompts=2048 \
        --ignore-eos \
        --skip-chat-template
    env:
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          key: hf_api_token
          name: hf-secret
```

Apply it using `kubectl`:

```shell
kubectl apply -f deepseek_v4_flash-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-deepseek-v4-flash dsv4-flash-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f deepseek_v4_flash-benchmark-8k1k.yaml
```

---

### Example Benchmark Output

The benchmark client will print a table resembling the following:

```
============ Serving Benchmark Result ============
Successful requests:                     2048
Failed requests:                         0
Benchmark duration (s):                  389.73
Total input tokens:                      16777216
Total generated tokens:                  2097152
Request throughput (req/s):              5.25
Output token throughput (tok/s):         5380.99
Peak output token throughput (tok/s):    13816.00
Peak concurrent requests:                2048.00
Total Token throughput (tok/s):          48428.95
---------------Time to First Token----------------
Mean TTFT (ms):                          173417.51
Median TTFT (ms):                        174208.15
P99 TTFT (ms):                           341178.90
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          157.86
Median TPOT (ms):                        155.73
P99 TPOT (ms):                           201.95
---------------Inter-token Latency----------------
Mean ITL (ms):                           157.87
Median ITL (ms):                         147.82
P99 ITL (ms):                            664.94
==================================================
```

---

## Step 7: Cleanup

When you are finished with the benchmark and no longer need the server, you can tear down the resources:

1. Delete the vLLM server:

```shell
kubectl delete -f deepseek_v4_flash-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
