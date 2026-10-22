# Serve GLM 5.2 FP8 with vLLM (TorchTPU) on Ironwood TPU (Multi-Host)

In this guide, we show how to serve GLM 5.2 FP8 (`zai-org/GLM-5.2-FP8`, a 753B Mixture-of-Experts model with Multi-head Latent Attention) with vLLM using the **TorchTPU** backend on Ironwood (TPU v7x) using GKE.

We deploy it across a multi-host **tpu7x-standard-4t** node pool (`2x2x2` topology, 2 hosts, 8 TPU v7x chips, 16 TensorCores) using a Kubernetes **LeaderWorkerSet** and **Ray**, with **Data Parallelism (DP) size of 16**, **Expert Parallelism (EP) size of 16** and **Tensor Parallelism (TP) size of 1**.

This recipe is a functional multi-host reference configuration.

## Verified Models

The following GLM 5.2 model is verified for deployment on TPU with the vLLM TorchTPU backend.

| Model | Parameters | Min TPUs (Chips) | HuggingFace |
| :--- | :--- | :--- | :--- |
| GLM 5.2 FP8 | 753B MoE (256 experts, top-8) | 8× (`2x2x2`, 2 hosts) | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) |

## Prerequisites

1. A GKE cluster with TPU v7x support (GKE version `1.34.0-gke.2201000` or later) and the LeaderWorkerSet (LWS) API enabled.
2. GCloud CLI and Kubectl installed and configured.
3. A Hugging Face account and access token with access to `zai-org/GLM-5.2-FP8`.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="glm52-tpu-pool"
```

---

## Step 2: Create the Multi-Host TPU Node Pool

Create a multi-host TPU v7x (Ironwood) nodepool with the `tpu7x-standard-4t` machine type and a `2x2x2` topology. This creates 2 nodes (hosts) with 4 chips each, for a total of 8 chips.

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

2. Create the Namespace `vllm-glm52` if you want to run it in a isolated namespace (recommended):

```shell
kubectl create namespace vllm-glm52
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-glm52 \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 4: Apply the vLLM Server Manifest

Apply the vLLM server manifest using the provided [glm52-server.yaml](./glm52-server.yaml) file in this directory:

```shell
cd inference/ironwood/vLLM/vLLM-TorchTPU/GLM-5.2
kubectl apply -f glm52-server.yaml
```

The manifest creates a `LeaderWorkerSet` with one leader pod and one worker pod, one on each TPU host. The leader starts a Ray head node, waits for the worker to join the Ray cluster, and then launches `vllm serve` across both hosts.

> [!NOTE]
> By default, `glm52-server.yaml` is configured for the `8k/1k` workload (`--max-model-len=9216` and `--max-num-seqs=64`). To run the long-context `128k/1k` workload, update these flags in `glm52-server.yaml` to `--max-model-len=132096` and `--max-num-seqs=16` before running Workload B.

Monitor the progress of the server deployment:

```shell
kubectl get pods -n vllm-glm52 -w
```

You can check the server logs on the leader pod using:

```shell
kubectl logs -n vllm-glm52 vllm-glm52-0 -c vllm-torchtpu -f
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
kubectl port-forward -n vllm-glm52 service/vllm-glm52-service 8000:8000
```

2. Run a curl request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zai-org/GLM-5.2-FP8",
        "messages": [
            {"role": "user", "content": "Hello! Tell me a joke."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

---

## Step 6: Run Serving Benchmarks

We use the `InferenceX` client from SemiAnalysisAI to perform serving benchmarks against our running vLLM server. We test two workloads: **8K Input / 1K Output** and **128K Input / 1K Output**.

### Workload A: 8K Input / 1K Output (8k/1k)

Save the following manifest as `glm52-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: glm52-bench-8k1k
  namespace: vllm-glm52
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-glm52-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
      cd /workspace/inferencex && \
      git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
      python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-glm52-service \
        --port=8000 \
        --model=zai-org/GLM-5.2-FP8 \
        --tokenizer=zai-org/GLM-5.2-FP8 \
        --dataset-name=random \
        --random-input-len=8192 \
        --random-output-len=1024 \
        --random-range-ratio=0.8 \
        --num-prompts=4096 \
        --max-concurrency=1024 \
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
kubectl apply -f glm52-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-glm52 glm52-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f glm52-benchmark-8k1k.yaml
```

---

### Workload B: 128K Input / 1K Output (128k/1k)

> [!NOTE]
> Before running this workload, update `glm52-server.yaml` to `--max-model-len=132096` and `--max-num-seqs=16` and re-apply it (see [Step 4](#step-4-apply-the-vllm-server-manifest)).

Save the following manifest as `glm52-benchmark-128k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: glm52-bench-128k1k
  namespace: vllm-glm52
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-glm52-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git && \
      git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
      cd /workspace/inferencex && \
      git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
      python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
        --backend=vllm \
        --request-rate=inf \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --host=vllm-glm52-service \
        --port=8000 \
        --model=zai-org/GLM-5.2-FP8 \
        --tokenizer=zai-org/GLM-5.2-FP8 \
        --dataset-name=random \
        --random-input-len=131072 \
        --random-output-len=1024 \
        --random-range-ratio=0.8 \
        --num-prompts=32 \
        --max-concurrency=16 \
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
kubectl apply -f glm52-benchmark-128k1k.yaml
```

Check the logs of the benchmark run:

```shell
kubectl logs -n vllm-glm52 glm52-bench-128k1k -f
```

When complete, clean up the benchmark pod:

```shell
kubectl delete -f glm52-benchmark-128k1k.yaml
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
kubectl delete -f glm52-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
