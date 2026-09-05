# Serve Qwen3-MoE with SGLang-Jax on TPU

SGLang-Jax supports multiple Mixture-of-Experts (MoE) models from the Qwen3 family with varying hardware requirements:

- **[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)**: Runs on 4 TPU v6e chips
- **[Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)**: Requires 64 TPU v6e chips (16 nodes × 4 chips)
- Other Qwen3 MoE variants with different scale requirements

**This tutorial focuses on deploying Qwen3-Coder-480B**, the largest model requiring a multi-node distributed setup. For smaller models like Qwen3-30B, you can follow similar steps but with adjusted node counts and parallelism settings.

## Hardware Requirements

Running Qwen3-Coder-480B requires a multi-node TPU cluster:

- **Total nodes**: 16
- **TPU chips per node**: 4 (v6e)
- **Total TPU chips**: 64
- **Tensor Parallelism (TP)**: 32 (for non-MoE layers)
- **Expert Tensor Parallelism (ETP)**: 64 (for MoE experts)


## Installation

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
## Launch Distributed Server

### Preparation

1. **Get Node 0 IP address** (coordinator):

```bash
# On node 0
hostname -I | awk '{print $1}'
```

Save this IP as `NODE_RANK_0_IP`.

2. **Download model** (recommended to use shared storage or pre-download on all nodes):

```bash
export HF_TOKEN=your_huggingface_token
huggingface-cli download Qwen/Qwen3-Coder-480B --local-dir /path/to/model
```

### Launch Command

Run the following command **on each node**, replacing:
- `<NODE_RANK_0_IP>`: IP address of node 0
- `<NODE_RANK>`: Current node rank (0-15)
- `<QWEN3_CODER_480B_MODEL_PATH>`: Path to the downloaded model

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python3 -u -m sgl_jax.launch_server \
    --model-path <QWEN3_CODER_480B_MODEL_PATH> \
    --trust-remote-code \
    --dist-init-addr=<NODE_RANK_0_IP>:10011 \
    --nnodes=16 \
    --tp-size=32 \
    --device=tpu \
    --random-seed=3 \
    --mem-fraction-static=0.8 \
    --chunked-prefill-size=2048 \
    --download-dir=/dev/shm \
    --dtype=bfloat16 \
    --max-running-requests=128 \
    --skip-server-warmup \
    --page-size=128 \
    --tool-call-parser=qwen3_coder \
    --node-rank=<NODE_RANK>
```

### Example for Specific Nodes

**Node 0 (coordinator):**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python3 -u -m sgl_jax.launch_server \
    --model-path /path/to/Qwen3-Coder-480B \
    --trust-remote-code \
    --dist-init-addr=10.0.0.2:10011 \
    --nnodes=16 \
    --tp-size=32 \
    --device=tpu \
    --random-seed=3 \
    --mem-fraction-static=0.8 \
    --chunked-prefill-size=2048 \
    --download-dir=/dev/shm \
    --dtype=bfloat16 \
    --max-running-requests=128 \
    --skip-server-warmup \
    --page-size=128 \
    --tool-call-parser=qwen3_coder \
    --node-rank=0
```

**Node 1:**

```bash
# Same command but with --node-rank=1
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python3 -u -m sgl_jax.launch_server \
    --model-path /path/to/Qwen3-Coder-480B \
    --trust-remote-code \
    --dist-init-addr=10.0.0.2:10011 \
    --nnodes=16 \
    --tp-size=32 \
    --device=tpu \
    --random-seed=3 \
    --mem-fraction-static=0.8 \
    --chunked-prefill-size=2048 \
    --download-dir=/dev/shm \
    --dtype=bfloat16 \
    --max-running-requests=128 \
    --skip-server-warmup \
    --page-size=128 \
    --tool-call-parser=qwen3_coder \
    --node-rank=1
```

Repeat for all 16 nodes, incrementing `--node-rank` from 0 to 15.

## Configuration Parameters

### Distributed Training

- `--nnodes`: Number of nodes in the cluster (16)
- `--node-rank`: Rank of the current node (0-15)
- `--dist-init-addr`: Address of the coordinator node (node 0) with port

### Model Parallelism

- `--tp-size`: Tensor parallelism size for non-MoE layers (32)
- **ETP**: Expert tensor parallelism automatically configured to 64 based on total chips

### Memory and Performance

- `--mem-fraction-static`: Memory allocation for static buffers (0.8)
- `--chunked-prefill-size`: Prefill chunk size for batching (2048)
- `--max-running-requests`: Maximum concurrent requests (128)
- `--page-size`: Page size for memory management (128)

### Model-Specific

- `--tool-call-parser`: Parser for tool calls, set to `qwen3_coder` for this model
- `--dtype`: Data type for inference (bfloat16)
- `--random-seed`: Random seed for reproducibility (3)

## Verification

Once all nodes are running, the server will be accessible via the coordinator node (node 0). You can test it with:

```bash
curl http://<NODE_RANK_0_IP>:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Coder-480B",
        "prompt": "def fibonacci(n):",
        "max_tokens": 200,
        "temperature": 0
    }'
```
