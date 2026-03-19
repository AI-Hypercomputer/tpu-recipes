# Serve Qwen3 with SGLang-Jax on TPU

This guide demonstrates how to serve [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) using SGLang-Jax on TPU.


## Provision TPU Resources

For **Qwen3-8B**, a single v6e chip is sufficient. For **Qwen3-32B**, use 4 chips or more.

### Option 1: Using gcloud CLI

Install and configure gcloud CLI by following the [official installation guide](https://cloud.google.com/sdk/docs/install).

**Create TPU VM:**

```bash
gcloud compute tpus tpu-vm create sgl-jax \
    --zone=us-east5-a \
    --version=v2-alpha-tpuv6e \
    --accelerator-type=v6e-4
```

**Connect to TPU VM:**

```bash
gcloud compute tpus tpu-vm ssh sgl-jax --zone us-east5-a
```

### Option 2: Using SkyPilot (Recommended for Development)

SkyPilot simplifies TPU provisioning and offers automatic cost optimization, instance management, and environment setup.

**Prerequisites:**
- [Install SkyPilot](https://docs.skypilot.co/en/latest/getting-started/installation.html)
- [Configure GCP credentials](https://docs.skypilot.co/en/latest/getting-started/installation.html#gcp)

**Create configuration file `sgl-jax.yaml`:**

```yaml
resources:
  accelerators: tpuv6e-4
  accelerator_args:
    tpu_vm: True
    runtime_version: v2-alpha-tpuv6e

setup: |
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install sglang-jax
```

**Launch TPU cluster:**

```bash
sky launch sgl-jax.yaml \
    --cluster=sgl-jax-skypilot-v6e-4 \
    --infra=gcp \
    -i 30 \
    --down \
    -y \
    --use-spot
```

This command will:
- Find the lowest-cost spot instance across regions
- Automatically shut down after 30 minutes of idleness
- Set up the SGLang-Jax environment automatically

**Connect to cluster:**

```bash
ssh sgl-jax-skypilot-v6e-4
```

> **Note:** SkyPilot manages the external IP automatically, so you don't need to track it manually.

## Installation

> **Note:** If you used SkyPilot to provision resources, the environment is already set up. Skip to the [Launch Server](#launch-server) section.

For gcloud CLI users, install SGLang-Jax using one of the following methods:

### Option 1: Install from PyPI

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install sglang-jax
```

### Option 2: Install from Source

```bash
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e python/
```

## Launch Server

Set the model name and start the SGLang-Jax server:

```bash
export MODEL_NAME="Qwen/Qwen3-8B"  # or "Qwen/Qwen3-32B"

JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
uv run python -u -m sgl_jax.launch_server \
    --model-path ${MODEL_NAME} \
    --trust-remote-code \
    --tp-size=4 \
    --device=tpu \
    --mem-fraction-static=0.8 \
    --chunked-prefill-size=2048 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --max-running-requests 256 \
    --skip-server-warmup \
    --page-size=128
```

### Configuration Parameters

- `--tp-size`: Tensor parallelism size, should equal the number of TPU chips in your instance
- `--mem-fraction-static`: Fraction of memory allocated for static buffers
- `--chunked-prefill-size`: Size of prefill chunks for batching
- `--max-running-requests`: Maximum number of concurrent requests

## Run Benchmark

Test serving performance with different workload configurations:

```bash
uv run python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --dataset-name random \
    --num-prompts 256 \
    --random-input 4096 \
    --random-output 1024 \
    --max-concurrency 64 \
    --random-range-ratio 1 \
    --warmup-requests 0
```

### Benchmark Parameters

- `--backend`: Backend engine (use `sgl-jax`)
- `--random-input`: Input sequence length (e.g., 1024, 4096, 8192)
- `--random-output`: Output sequence length (e.g., 1, 1024)
- `--max-concurrency`: Maximum number of concurrent requests (e.g., 8, 16, 32, 64, 128, 256)
- `--num-prompts`: Total number of prompts to send

You can test various combinations of input/output lengths and concurrency levels to evaluate throughput and latency characteristics.
