# Benchmark Qwen3-Reranker-0.6B with vLLM on TPU v6e (torchax / tpu-inference)

This recipe shows how to run and benchmark the [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
reranker on a single Trillium (v6e) chip using the public JAX/torchax vLLM TPU backend
([`tpu-inference`](https://github.com/vllm-project/tpu-inference)). Scoring uses vLLM's pooling
`score()` API: each (query, document) pair returns one relevance score, and you rank documents by
that score.

Note on the model architecture: `tpu-inference` registers CausalLM-style architectures, so the
sequence-classification reranker is not loaded from a pre-converted seq-cls checkpoint. Instead we load
the original `Qwen/Qwen3-Reranker-0.6B` weights and attach the seq-cls scoring head at load time via vLLM
`hf_overrides` (`classifier_from_token=["no","yes"]`, `is_original_qwen3_reranker=True`). Same weights,
same `score()` API.

## Step 0: Install `gcloud` CLI

Follow [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install), then `gcloud auth login`.

## Step 1: Create a v6e-1 TPU VM

One chip is sufficient for this 0.6B model.

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --accelerator-type v6e-1 \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv6e
```

If on-demand capacity is unavailable, use [Queued Resources](https://cloud.google.com/tpu/docs/queued-resources):

```bash
export QR_ID=your-qr-id
gcloud alpha compute tpus queued-resources create $QR_ID \
    --node-id $TPU_NAME --project $PROJECT --zone $ZONE \
    --accelerator-type v6e-1 --runtime-version v2-alpha-tpuv6e
```

## Step 2: SSH into the VM

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 3: Install the public vLLM TPU backend

We recommend a fresh Python 3.12 virtual environment with [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv --python 3.12 "$HOME/tpuinf_venv"
source "$HOME/tpuinf_venv/bin/activate"
uv pip install vllm-tpu transformers
```

Verify the TPU backend is active:

```bash
python -c "import jax, vllm; from vllm.platforms import current_platform; \
    print(vllm.__version__, current_platform.get_device_name(), jax.devices())"
# expected: <vllm version>  TPU V6E  [TpuDevice(id=0, ...)]
```

## Step 4: Set environment variables

```bash
export HF_HOME=/dev/shm
export HF_TOKEN=<your HF token>
```

## Step 5: Quick correctness check

Copy `rerank_script.py` (in this folder) to the VM and run it. Relevant pairs should score high.

```bash
python rerank_script.py
```

Example output (scores are relevance probabilities; higher = more relevant):

```
0.9741  What is the capital of China?
0.8867  What is photosynthesis?
0.8214  Who wrote Pride and Prejudice?
0.6355  How do vaccines work?
```

## Step 6: Run the throughput benchmark

Copy `benchmark_reranker.py` to the VM and run it. It sweeps batch sizes and reports peak
pairs/sec plus single-request latency.

```bash
python benchmark_reranker.py \
    --max-model-len 1024 \
    --batch-sizes 8,16,32,64,128,256,512,1024 \
    --result-filename reranker-bench-tpu.json
```

## Results (v6e-1, torchax / tpu-inference)

Measured on a single v6e chip. Throughput is query/document pairs scored per second (one saturating
`score()` call per batch size, best of 3). Numbers vary with vLLM/tpu-inference version.

| Batch size | Throughput (pairs/s) |
|-----------:|---------------------:|
| 8 | 707.575 |
| 16 | 1040.238 |
| 32 | 1171.068 |
| 64 | 1312.284 |
| 128 | 1402.996 |
| 256 | 1460.995 |
| 512 | 1517.183 |
| 1024 | 1655.593 |

Peak throughput: 1655.593 pairs/s at batch size 1024.

Single-request latency (batch=1, 300 iterations): p50 4.571 ms, p99 4.796 ms, mean 4.604 ms
(realtime QPS ~217).

Environment: vLLM 0.27.0 with `tpu-inference` (public), jax/jaxlib 0.11.x, libtpu 0.0.x, Python 3.12,
runtime `v2-alpha-tpuv6e`, `max_model_len=1024`, `max_num_batched_tokens=16384`.

## Files

- `rerank_script.py` - minimal offline reranking / correctness demo.
- `benchmark_reranker.py` - batch-size throughput sweep + single-request latency, writes a JSON report.
