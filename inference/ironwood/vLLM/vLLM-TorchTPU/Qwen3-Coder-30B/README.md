# Qwen3-Coder-30B-A3B-Instruct-FP8 on Ironwood (TPU v7x) with vLLM-TorchTPU

This recipe provides the deployment manifest and benchmarking instructions for serving **Qwen3-Coder-30B-A3B-Instruct-FP8** on a single-host Google Cloud TPU v7x (Ironwood) node (`2x2x1` topology, 4 chips / 8 TensorCores) using the PyTorch-native [vLLM-TorchTPU](https://github.com/vllm-project/vllm-torchtpu) engine.

---

## Model & Hardware Overview

* **Model**: [Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)
* **Architecture**: Mixture-of-Experts (MoE) with 64 routed experts
* **Hardware**: Google Cloud TPU v7x (`2x2x1`, 4 chips / 8 TensorCores)
* **Parallelism**: Tensor Parallelism (`tp=8`), Data Parallelism (`dp=1`), Expert Parallelism (`--enable-expert-parallel`)
* **Attention Backend**: Pallas Custom Ragged Paged Attention (`--attention-backend=CUSTOM`)
* **Precision**: FP8 Weights and KV Cache (`--kv-cache-dtype=fp8`)

---

## Verified Benchmark Performance (TPU v7x 2x2x1)

The following benchmark metrics were verified using InferenceX (`benchmark_serving.py`) on a single-host TPU v7x node with 100% request completion:

| Context Window | Input Length (ISL) | Output Length (OSL) | Request Rate (req/s) | Output Tok/s (Aggregate) | Output Tok/s per Chip | TTFT Mean (ms) | TTFT P50 (ms) | TPOT Mean (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard (1K / 1K)** | 1024 | 1024 | 9.29 | 8,581.64 | 2,145.41 | 8,157.79 | 6,448.04 | 17.81 |
| **Extended (1K / 2K)** | 1024 | 2048 | 6.77 | 12,496.00 | 3,124.01 | 6,996.25 | 2,145.70 | 14.60 |

---

## Deployment Instructions

### 1. Create Kubernetes Namespace and Secret

```bash
kubectl create namespace vllm-qwen3-coder-30b

# Create secret with your Hugging Face API token
kubectl create secret generic hf-secret \
  --namespace=vllm-qwen3-coder-30b \
  --from-literal=hf_api_token=INSERT_YOUR_TOKEN
```

### 2. Deploy vLLM Server

Deploy the server using the companion manifest:

```bash
kubectl apply -f qwen3-coder-30b-server.yaml
```

Wait until the pod enters `Running` state and the vLLM server is initialized:

```bash
kubectl get pods -n vllm-qwen3-coder-30b -w
kubectl logs -f -n vllm-qwen3-coder-30b deployment/vllm-qwen3-coder-30b
```

---

## Benchmarking with InferenceX

Run the benchmark client against the deployed server:

```bash
python3 benchmark_serving.py \
  --host=vllm-qwen3-coder-30b-service.vllm-qwen3-coder-30b \
  --port=8000 \
  --dataset-name=random \
  --tokenizer=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --random-input-len=1024 \
  --random-output-len=1024 \
  --num-prompts=320 \
  --max-concurrency=320 \
  --random-range-ratio=0.8 \
  --ignore-eos \
  --skip-chat-template
```

---

## Cleanup

```bash
kubectl delete -f qwen3-coder-30b-server.yaml
kubectl delete namespace vllm-qwen3-coder-30b
```
