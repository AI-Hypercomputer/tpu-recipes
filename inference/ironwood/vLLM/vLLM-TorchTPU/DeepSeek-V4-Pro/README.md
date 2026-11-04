# Serve DeepSeek-V4-Pro with vLLM (TorchTPU) on Ironwood TPU (Multi-Host)

In this guide, we show how to serve DeepSeek-V4-Pro (`deepseek-ai/DeepSeek-V4-Pro-0813`, Mixture-of-Experts model with Multi-head Latent Attention) with vLLM using the **TorchTPU** backend on Ironwood (TPU v7x) using GKE.

We deploy it across a multi-host **tpu7x-standard-4t** node pool (`2x2x2` topology, 2 hosts, 8 TPU v7x chips, 16 TensorCores) using a Kubernetes **LeaderWorkerSet** and **Ray**, with **Data Parallelism (DP) size of 16**, **Expert Parallelism (EP) size of 16**, and **Tensor Parallelism (TP) size of 1**, using `deepseek_v4_fp8` quantization, fused EP MoE v2 with load-time FP4 MoE weight requantization (`W4A8`), and `fp8_e4m3` KV cache.

## Verified Models

The following DeepSeek-V4-Pro model is verified for deployment on TPU with the vLLM TorchTPU backend.

| Model | Quantization | Min TPUs (Chips) | Topology | HuggingFace |
| :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V4-Pro-0813 | `deepseek_v4_fp8` (W4A8 MoE), `fp8_e4m3` KV | 8× (`tpu7x-standard-4t`) | `2x2x2` (2 hosts) | [deepseek-ai/DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) |

## Prerequisites

1. A GKE cluster with TPU v7x support (GKE version `1.34.0-gke.2201000` or later) and the LeaderWorkerSet (LWS) API enabled.
2. GCloud CLI and Kubectl installed and configured.
3. A Hugging Face account and access token with access to `deepseek-ai/DeepSeek-V4-Pro-0813`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="dsv4-pro-tpu-pool"
```

---

## Step 2: Create the Multi-Host TPU Node Pool

Create a multi-host TPU v7x (Ironwood) node pool with the `tpu7x-standard-4t` machine type and a `2x2x2` topology. This creates 2 nodes (hosts) with 4 chips each, for a total of 8 chips (16 logical TensorCore devices):

```shell
gcloud container node-pools create ${NODEPOOL_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --node-locations=${ZONE} \
    --num-nodes=2 \
    --machine-type=tpu7x-standard-4t \
    --tpu-topology=2x2x2 \
    --cluster=${CLUSTER_NAME}
```

---

## Step 3: Setup Kubernetes Configurations

1. Configure `kubectl` to communicate with your cluster:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID}
```

2. Create the Namespace `vllm-deepseek-v4-pro` if you want to run it in an isolated namespace (recommended):

```shell
kubectl create namespace vllm-deepseek-v4-pro
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-deepseek-v4-pro \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Apply the vLLM server manifest using the provided [deepseek_v4_pro-server.yaml](./deepseek_v4_pro-server.yaml) file in this directory:

```shell
cd inference/ironwood/vLLM/vLLM-TorchTPU/DeepSeek-V4-Pro
kubectl apply -f deepseek_v4_pro-server.yaml
```

The manifest creates a `LeaderWorkerSet` with one leader pod and one worker pod, one on each TPU host. The leader starts a Ray head node, waits for the worker to join the Ray cluster (16 TPU devices total), and then launches `vllm serve` across both hosts.

Monitor the progress of the server deployment:

```shell
kubectl get pods -n vllm-deepseek-v4-pro -w
```

You can check the server logs on the leader pod using:

```shell
kubectl logs -n vllm-deepseek-v4-pro vllm-deepseek-v4-pro-0 -c vllm-torchtpu -f
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
kubectl port-forward -n vllm-deepseek-v4-pro service/vllm-deepseek-v4-pro-service 8000:8000
```

2. Run a `curl` request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "messages": [
            {"role": "user", "content": "Write a short poem that includes fruits like apples, bananas and pineapples."}
        ],
        "max_tokens": 200,
        "temperature": 0.0
    }'
```

---

## Step 6: Run Serving Benchmarks

We use `vllm bench serve` to evaluate serving throughput and latency against our running vLLM server for an **8K Input / 1K Output (8k/1k)** workload.

### Workload: 8K Input / 1K Output (8k/1k)

Save the following manifest as `deepseek_v4_pro-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dsv4-pro-bench-8k1k
  namespace: vllm-deepseek-v4-pro
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-deepseek-v4-pro-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      vllm bench serve \
        --backend=vllm \
        --base-url=http://vllm-deepseek-v4-pro-service:8000 \
        --model=deepseek-ai/DeepSeek-V4-Pro-0813 \
        --tokenizer=deepseek-ai/DeepSeek-V4-Pro-0813 \
        --trust-remote-code \
        --dataset-name=random \
        --random-input-len=8192 \
        --random-output-len=1024 \
        --num-prompts=2048 \
        --random-range-ratio=0 \
        --ignore-eos \
        --max-concurrency=1024
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
kubectl apply -f deepseek_v4_pro-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-deepseek-v4-pro dsv4-pro-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f deepseek_v4_pro-benchmark-8k1k.yaml
```

---

### Example Benchmark Output

The benchmark client will print a table resembling the following:

```
============ Serving Benchmark Result ============
Successful requests:                     2048
Failed requests:                         0
Maximum request concurrency:             1024
Benchmark duration (s):                  xxx
Total input tokens:                      16777216
Total generated tokens:                  2097152
Request throughput (req/s):              xx
Output token throughput (tok/s):         xxx
Peak output token throughput (tok/s):    xxx
Peak concurrent requests:                1024.00
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

---

## Step 7: Cleanup

When you are finished with the benchmark and no longer need the server, you can tear down the resources:

1. Delete the vLLM server:

```shell
kubectl delete -f deepseek_v4_pro-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
