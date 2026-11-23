# Serve GLM 5.3 with vLLM (TorchTPU) on Ironwood TPU (Multi-Host)

In this guide, we show how to serve GLM 5.3 (`zai-org/GLM-5.3`, a 753B Mixture-of-Experts model with DeepSeek Sparse Multi-head Latent Attention) with vLLM using the **TorchTPU** backend (`vllm/vllm-torchtpu:v0.33.0`) on Ironwood (TPU v7x) using GKE.

We deploy it across a single multi-host **tpu7x-standard-4t** slice (`2x2x2` topology, 2 hosts, 8 TPU v7x chips, 16 TensorCores) using a Kubernetes **LeaderWorkerSet** and vLLM's native multi-node `mp` executor with **Tensor Parallelism (TP) size of 16**, **Intra-Host Decode Context Parallelism (DCP) size of 8** (`--cp-kv-cache-interleave-size=4`), and **Expert Parallelism (EP) size of 16**, with FP8 KV cache.

## Verified Models

The following GLM 5.3 model is verified for deployment on TPU with the vLLM TorchTPU backend.

| Model | Quantization | Min TPUs (Chips) | Topology | HuggingFace |
| :--- | :--- | :--- | :--- | :--- |
| GLM 5.3 | FP8 (`e4m3`, 128×128 block-wise), `fp8` KV | 8× (`tpu7x-standard-4t`) | `2x2x2` (2 hosts) | [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) |

## Prerequisites

1. A GKE cluster with TPU v7x support (GKE version `1.34.0-gke.2201000` or later) and the LeaderWorkerSet (LWS) API enabled.
2. GCloud CLI and Kubectl installed and configured.
3. A Hugging Face account and access token with access to `zai-org/GLM-5.3`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="glm53-tpu-pool"
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

2. Create the Namespace `vllm-glm53` if you want to run it in an isolated namespace (recommended):

```shell
kubectl create namespace vllm-glm53
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-glm53 \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Apply the vLLM server manifest using the provided [glm53-server.yaml](./glm53-server.yaml) file in this directory:

```shell
cd inference/ironwood/vLLM/vLLM-TorchTPU/GLM-5.3
kubectl apply -f glm53-server.yaml
```

The manifest creates a `LeaderWorkerSet` with one leader pod (`node-rank=0`) and one headless worker pod (`node-rank=1`), one on each TPU host. Both hosts stage the `zai-org/GLM-5.3` checkpoint onto local `hyperdisk-balanced` storage, synchronize, and launch `vllm serve` across all 16 TensorCore devices (`TP=16`, `DCP=8`, `EP=16`).

Monitor the progress of the server deployment:

```shell
kubectl get pods -n vllm-glm53 -w
```

You can check the server logs on the leader pod using:

```shell
kubectl logs -n vllm-glm53 vllm-glm53-0 -c vllm-torchtpu -f
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
kubectl port-forward -n vllm-glm53 service/vllm-glm53-service 8000:8000
```

2. Run a `curl` request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zai-org/GLM-5.3",
        "messages": [
            {"role": "user", "content": "Hello! Tell me a joke."}
        ],
        "max_tokens": 100,
        "temperature": 0.0
    }'
```

---

## Step 6: Run Serving Benchmarks

We use the `InferenceX` client (`benchmark_serving.py`) to evaluate serving throughput and latency against our running vLLM server for a **6K Input / 2K Output** (`--random-input-len=6144`, `--random-output-len=2048`, `--random-range-ratio=0.8`) workload at concurrency `128` (or `64`).

### Workload: 6K Input / 2K Output (`6k/2k`)

Save the following manifest as `glm53-benchmark-6k2k.yaml` (to benchmark at concurrency `64`, change `--max-concurrency=128` to `--max-concurrency=64`):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: glm53-bench-6k2k
  namespace: vllm-glm53
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.33.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-glm53-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
      cd /workspace/inferencex && \
      git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
      python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-glm53-service \
        --port=8000 \
        --model=zai-org/GLM-5.3 \
        --tokenizer=zai-org/GLM-5.3 \
        --dataset-name=random \
        --random-input-len=6144 \
        --random-output-len=2048 \
        --random-range-ratio=0.8 \
        --num-prompts=512 \
        --max-concurrency=128 \
        --ignore-eos
    env:
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          key: hf_api_token
          name: hf-secret
    resources:
      requests:
        cpu: "8"
        memory: "16Gi"
      limits:
        cpu: "8"
        memory: "16Gi"
```

Apply it using `kubectl`:

```shell
kubectl apply -f glm53-benchmark-6k2k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-glm53 glm53-bench-6k2k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f glm53-benchmark-6k2k.yaml
```

---

### Example Benchmark Output

The benchmark client will print a table resembling the following:

```
============ Serving Benchmark Result ============
Successful requests:                     512
Failed requests:                         0
Maximum request concurrency:             128
Benchmark duration (s):                  xxx
Total input tokens:                      xxx
Total generated tokens:                  xxx
Request throughput (req/s):              xx
Output token throughput (tok/s):         xxx
Peak output token throughput (tok/s):    xxx
Peak concurrent requests:                128.00
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

1. Delete the vLLM server (associated `data-volume` PVCs are reclaimed automatically via `persistentVolumeClaimRetentionPolicy`):

```shell
kubectl delete -f glm53-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
