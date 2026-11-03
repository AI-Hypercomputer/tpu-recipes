Serve Qwen3-Coder-480B-A35B-Instruct-FP8 with vLLM (TorchTPU) on Ironwood TPU (Single-Host)
==========================================================================================

In this guide, we show how to serve Qwen3-Coder-480B-A35B-Instruct-FP8 (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, a Mixture-of-Experts model with 384 routed experts) with vLLM using the **TorchTPU** backend on Ironwood (TPU v7x) using GKE.

We deploy it across a single-host **tpu7x-standard-4t** node pool (`2x2x1` topology, 1 host, 4 TPU v7x chips, 8 TensorCores) using a Kubernetes **Deployment**, with **Tensor Parallelism (TP) size of 8**, **Data Parallelism (DP) size of 1**, and **Expert Parallelism (EP)**.

This recipe is a functional single-host reference configuration.

Verified Models
---------------

The following Qwen3-Coder-480B model is verified for deployment on TPU with the vLLM TorchTPU backend.

| Model | Parameters | Min TPUs (Chips) | HuggingFace |
| :--- | :--- | :--- | :--- |
| Qwen3-Coder-480B-FP8 | 480B MoE (384 experts, top-8) | 4x (`2x2x1`, 1 host) | [Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) |

Prerequisites
-------------

1. A GKE cluster with TPU v7x support (GKE version `1.34.0-gke.2201000` or later).
2. GCloud CLI and Kubectl installed and configured.
3. A Hugging Face account and access token with access to `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`.

---

Step 1: Define Parameters
-------------------------

Export the necessary environment variables for your Google Cloud project and GKE cluster:

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="qwen3-coder-480b-pool"
```

---

Step 2: Create the Single-Host TPU Node Pool
--------------------------------------------

Create a single-host TPU v7x (Ironwood) nodepool with the `tpu7x-standard-4t` machine type and a `2x2x1` topology. This creates 1 node (host) with 4 chips (8 TensorCores).

```bash
gcloud container node-pools create ${NODEPOOL_NAME}     --project=${PROJECT_ID}     --location=${REGION}     --node-locations=${ZONE}     --num-nodes=1     --machine-type=tpu7x-standard-4t     --tpu-topology=2x2x1     --cluster=${CLUSTER_NAME}
```

---

Step 3: Setup Kubernetes Configurations
---------------------------------------

1. Configure `kubectl` to communicate with your cluster:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID}
```

2. Create the Namespace `vllm-qwen3-coder-480b` if you want to run it in an isolated namespace (recommended):

```bash
kubectl create namespace vllm-qwen3-coder-480b
```

3. Create a Kubernetes secret for your Hugging Face token in the namespace:

```bash
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret     --namespace=vllm-qwen3-coder-480b     --from-literal=hf_api_token=${HF_TOKEN}
```

---

Step 4: Apply the vLLM Server Manifest
--------------------------------------

Apply the vLLM server manifest using the provided [qwen3-coder-480b-server.yaml](qwen3-coder-480b-server.yaml) file in this directory:

```bash
cd inference/ironwood/vLLM/vLLM-TorchTPU/Qwen3-Coder-480B
kubectl apply -f qwen3-coder-480b-server.yaml
```

Monitor the progress of the server deployment:

```bash
kubectl get pods -n vllm-qwen3-coder-480b -w
```

You can check the server logs on the pod using:

```bash
kubectl logs -n vllm-qwen3-coder-480b deployment/vllm-qwen3-coder-480b -c vllm-torchtpu -f
```

Once the server has successfully loaded the model, you should see logs like:

```text
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

---

Step 5: Interact with the Model (Optional)
------------------------------------------

You can test the server using a port forward and curl:

1. Port forward the vLLM service to port `8000` on your local machine:

```bash
kubectl port-forward -n vllm-qwen3-coder-480b service/vllm-qwen3-coder-480b-service 8000:8000
```

2. Run a curl request:

```bash
curl http://localhost:8000/v1/chat/completions     -H "Content-Type: application/json"     -d '{
 "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
 "messages": [
 {"role": "user", "content": "Hello! Write a Python quicksort function."}
 ],
 "max_tokens": 100,
 "temperature": 0.7
 }'
```

---

Step 6: Run Serving Benchmarks
------------------------------

We use the `InferenceX` client from SemiAnalysisAI to perform serving benchmarks against our running vLLM server. We test three workloads: **1K Input / 1K Output**, **1K Input / 8K Output**, and **8K Input / 1K Output**.

### Workload A: 1K Input / 1K Output (1k/1k)

Save the following manifest as `qwen3-coder-480b-benchmark-1k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-coder-480b-bench-1k1k
  namespace: vllm-qwen3-coder-480b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-qwen3-coder-480b-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git &&       git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex &&       cd /workspace/inferencex &&       git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd &&       python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py         --backend=vllm         --request-rate=inf         --percentile-metrics='ttft,tpot,itl,e2el'         --host=vllm-qwen3-coder-480b-service         --port=8000         --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8         --tokenizer=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8         --dataset-name=random         --random-input-len=1024         --random-output-len=1024         --random-range-ratio=0.8         --num-prompts=256         --max-concurrency=256         --ignore-eos
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          key: hf_api_token
          name: hf-secret
```

Apply it using `kubectl`:

```bash
kubectl apply -f qwen3-coder-480b-benchmark-1k1k.yaml
```

Check the logs of the benchmark run:

```bash
kubectl logs -n vllm-qwen3-coder-480b qwen3-coder-480b-bench-1k1k -f
```

When complete, clean up the benchmark pod:

```bash
kubectl delete -f qwen3-coder-480b-benchmark-1k1k.yaml
```

---

### Workload B: 1K Input / 8K Output (1k/8k)

Save the following manifest as `qwen3-coder-480b-benchmark-1k8k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-coder-480b-bench-1k8k
  namespace: vllm-qwen3-coder-480b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-qwen3-coder-480b-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git &&       git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex &&       cd /workspace/inferencex &&       git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd &&       python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py         --backend=vllm         --request-rate=inf         --percentile-metrics='ttft,tpot,itl,e2el'         --host=vllm-qwen3-coder-480b-service         --port=8000         --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8         --tokenizer=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8         --dataset-name=random         --random-input-len=1024         --random-output-len=8192         --random-range-ratio=0.8         --num-prompts=128         --max-concurrency=128         --ignore-eos
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          key: hf_api_token
          name: hf-secret
```

Apply it using `kubectl`:

```bash
kubectl apply -f qwen3-coder-480b-benchmark-1k8k.yaml
```

Check the logs of the benchmark run:

```bash
kubectl logs -n vllm-qwen3-coder-480b qwen3-coder-480b-bench-1k8k -f
```

When complete, clean up the benchmark pod:

```bash
kubectl delete -f qwen3-coder-480b-benchmark-1k8k.yaml
```

---

### Workload C: 8K Input / 1K Output (8k/1k)

Save the following manifest as `qwen3-coder-480b-benchmark-8k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-coder-480b-bench-8k1k
  namespace: vllm-qwen3-coder-480b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
  - name: vllm-bench
    image: vllm/vllm-torchtpu:v0.29.0
    command: ["/bin/bash", "-c"]
    args:
    - |
      while ! curl -sf http://vllm-qwen3-coder-480b-service:8000/health; do sleep 30 && echo 'Waiting for server...'; done
      apt-get update && apt-get install -y git &&       git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex &&       cd /workspace/inferencex &&       git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd &&       python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py         --backend=vllm         --request-rate=inf         --percentile-metrics='ttft,tpot,itl,e2el'         --host=vllm-qwen3-coder-480b-service         --port=8000         --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8         --tokenizer=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8         --dataset-name=random         --random-input-len=8192         --random-output-len=1024         --random-range-ratio=0.8         --num-prompts=128         --max-concurrency=128         --ignore-eos
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          key: hf_api_token
          name: hf-secret
```

Apply it using `kubectl`:

```bash
kubectl apply -f qwen3-coder-480b-benchmark-8k1k.yaml
```

Check the logs of the benchmark run:

```bash
kubectl logs -n vllm-qwen3-coder-480b qwen3-coder-480b-bench-8k1k -f
```

When complete, clean up the benchmark pod:

```bash
kubectl delete -f qwen3-coder-480b-benchmark-8k1k.yaml
```

---

### Example Benchmark Output

The benchmark client will print a table resembling the following:

```text
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

Verified Benchmark Results (TPU v7x 2x2x1)
------------------------------------------

The following benchmark metrics were verified using InferenceX (`benchmark_serving.py`) on a single-host TPU v7x node with 100% request completion:

| Context Window | Input Length (ISL) | Output Length (OSL) | Request Rate (req/s) | Output Tok/s (Aggregate) | Output Tok/s per Chip | TTFT Mean (ms) | TTFT P50 (ms) | TPOT Mean (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard (1K / 1K)** | 1024 | 1024 | 4.79 | 4,393.69 | 1,098.42 | 6,234.47 | 6,221.46 | 46.69 |
| **Long Generation (1K / 8K)** | 1024 | 8192 | 0.44 | 3,271.06 | 817.77 | 3,107.37 | 3,214.11 | 35.63 |
| **Long Prompt (8K / 1K)** | 8192 | 1024 | 1.53 | 1,401.75 | 350.44 | 23,736.20 | 23,628.50 | 61.56 |

---

Step 7: Cleanup
---------------

When you are finished with the benchmark and no longer need the server, you can tear down the resources:

1. Delete the vLLM server:

```bash
kubectl delete -f qwen3-coder-480b-server.yaml
```

2. (Optional) Delete the GKE TPU node pool:

```bash
gcloud container node-pools delete ${NODEPOOL_NAME}     --cluster=${CLUSTER_NAME}     --location=${REGION}     --project=${PROJECT_ID}
```
