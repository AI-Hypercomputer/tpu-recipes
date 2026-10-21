# Serve and benchmark Qwen3.8-27B with vLLM on TPU7x (Ironwood)

This recipe serves [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense 27B, hybrid linear-attention/full-attention architecture, Apache-2.0) on a single `tpu7x-standard-4t` node pool (4 TPU7x chips, 8 TensorCores) with Tensor Parallelism size 8, and benchmarks it with `vllm bench serve`.

> **Image requirement:** Qwen3.8-27B needs a vLLM TPU build that ships the GDN **v3** decode kernel (`vllm/vllm-tpu:nightly-20260816-11250b4-1d2d83a` or newer, pinned in the manifest). Older builds with the v2 kernel fail Mosaic compilation for this model's linear-attention head geometry on TPU7x (`Not implemented: Lane broadcast`).
>
> **Sharding note (measured):** TP=8 is the best-performing configuration for this model on current builds — measured head-to-head against `--tensor-parallel-size=2 --data-parallel-size=4` (the Gemma4/GPT-OSS pattern), which serves correctly but lands ~14% lower (3,704 vs 4,287 tok/s at concurrency 128) because each DP replica re-reads the full 54 GB of weights from a single chip's HBM. Two practical constraints to know: (1) the mesh must tile all 8 TensorCores — partial-host configurations (TP=4 or TP=2 without DP) either never finish warmup or serve pathologically slowly on current builds; (2) DP warmup compilation exceeds vLLM's default engine-ready timeout — if you experiment with DP, set `VLLM_ENGINE_READY_TIMEOUT_S` generously (5400) or set `SKIP_JAX_PRECOMPILE=1` and absorb first-request compilation.

## Set up environment variables

```bash
export CLUSTER_NAME=your-gke-cluster
export ZONE=your-zone
export PROJECT=your-project
export HF_TOKEN=<your Hugging Face token>   # the model is ungated; any valid token works
```

## Create the TPU node pool

```bash
gcloud container node-pools create qwen38-pool \
  --cluster=${CLUSTER_NAME} \
  --location=${ZONE} \
  --node-locations=${ZONE} \
  --project=${PROJECT} \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=1
```

Add `--reservation-affinity=specific --reservation=<your-reservation>` to draw from a reservation, or `--spot` for spot capacity.

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT}
```

## Deploy the vLLM server

Create the namespace-scoped resources and the Hugging Face token secret, then apply the server manifest:

```bash
kubectl apply -f qwen3_8-server.yaml
kubectl create secret generic hf-secret \
  --namespace=vllm-qwen38 \
  --from-literal=hf_api_token=${HF_TOKEN}
```

Wait for the server to come up (image pull + ~54 GB weight download to the hyperdisk volume + compilation; roughly 20–30 minutes on a fresh pool):

```bash
kubectl wait --for=condition=ready pod -l app=vllm-qwen38 -n vllm-qwen38 --timeout=2400s
kubectl logs -f deployment/vllm-qwen38 -n vllm-qwen38
```

The server is ready when the log shows:

```
(APIServer pid=1) INFO:     Application startup complete.
```

## Test the server

```bash
kubectl port-forward -n vllm-qwen38 service/qwen38-service 8000:8000 &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "In one short sentence: what is a TPU?"}],
    "max_tokens": 200
  }'
```

The response carries the answer in `content` and the model's thinking trace in `reasoning` (split by `--reasoning-parser qwen3`).

## Run the benchmark

```bash
kubectl exec -n vllm-qwen38 deploy/vllm-qwen38 -- bash -c \
  "vllm bench serve --backend vllm --model Qwen/Qwen3.8-27B \
   --dataset-name random --random-input-len 1024 --random-output-len 1024 \
   --num-prompts 128 --max-concurrency 64 --ignore-eos --seed 42"
```

Expected output (measured 2026-08-17 on tpu7x-standard-4t, image as pinned):

```
============ Serving Benchmark Result ============
Successful requests:                     128
Failed requests:                         0
Maximum request concurrency:             64
Benchmark duration (s):                  41.00
Total input tokens:                      131072
Total generated tokens:                  131072
Request throughput (req/s):              3.12
Output token throughput (tok/s):         3196.91
Peak output token throughput (tok/s):    3712.00
Peak concurrent requests:                87.00
Total token throughput (tok/s):          6393.83
---------------Time to First Token----------------
Mean TTFT (ms):                          1004.91
Median TTFT (ms):                        264.10
P99 TTFT (ms):                           3214.40
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.81
Median TPOT (ms):                        19.16
P99 TPOT (ms):                           19.52
---------------Inter-token Latency----------------
Mean ITL (ms):                           18.81
Median ITL (ms):                         17.49
P99 ITL (ms):                            44.46
==================================================
```

Measured across shapes and concurrencies with the same protocol:

| input/output | max-concurrency | num-prompts | Output tok/s | Mean TPOT | Median TTFT |
|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 64 | 128 | 3,197 | 18.8 ms | 264 ms |
| 1024/1024 | 128 | 256 | 4,287 | 27.7 ms | 597 ms |
| 1024/1024 | 256 | 512 | 4,750 | 47.9 ms | 2,550 ms |
| 1024/8192 | 128 | 128 | 4,879 | 25.6 ms | 3,872 ms |
| 8192/1024 | 128 | 128 | 1,961 | 34.8 ms | 22,001 ms |

Throughput on current builds is bounded by the gated-delta-rule (linear-attention) decode kernel rather than by memory bandwidth, so these numbers are expected to improve with future kernel releases.

## Clean up

```bash
kubectl delete namespace vllm-qwen38
kubectl delete storageclass hyperdisk-balanced-qwen38
gcloud container node-pools delete qwen38-pool \
  --cluster=${CLUSTER_NAME} --location=${ZONE} --project=${PROJECT} --quiet
```
