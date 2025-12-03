#!/bin/bash

# --- Environment Setup ---
# This script requires uv and a Python 3.11 virtual environment with xpk installed.
# If you haven't set up uv and the environment, please refer to the README.md.

UV_VENV_PATH="${HOME}/.local/bin/venv"
UV_PYTHON_VERSION="3.11"

# Activate the virtual environment
source "${UV_VENV_PATH}/bin/activate"

# Check if xpk is installed in the venv
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the virtual environment. Please install it by running:"
    echo "pip install xpk==0.16.0"
    exit 1
fi
# --- End Environment Setup ---

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---

# --- Environment Variables ---
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE=""
export BASE_OUTPUT_DIR=""
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-llama3-1-70b-8192-4x4x4")-$(date +%Y%m%d-%H%M)"

# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=61440 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=llama3.1-70b \
skip_jax_distributed_system=True \
dtype=bfloat16 \
per_device_batch_size=2 \
profile_periodically_period=10000 \
async_checkpointing=False \
enable_checkpointing=False \
use_iota_embed=True \
remat_policy=custom \
decoder_layer_input=device \
context=device \
query_proj=device \
key_proj=device \
value_proj=device \
qkv_proj=device \
ici_fsdp_parallelism=-1 \
dataset_type=synthetic \
opt_type=adamw \
mu_dtype=bfloat16 \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
tokenizer_type=tiktoken \
tokenizer_path=assets/tokenizer_llama3.tiktoken \
sa_q_layout=SEQ_MINOR \
sa_k_layout=SEQ_MINOR \
sa_v_layout=HEAD_DIM_MINOR \
sa_use_fused_bwd_kernel=True \
use_tokamax_splash=True \
max_target_length=8192 \
profiler=xplane \
skip_first_n_steps_for_profiler=5 \
profiler_steps=2 \
attention=flash \
steps=30 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"

xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --device-type=tpu7x-4x4x4 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -m MaxText.train MaxText/configs/base.yml ${MAXTEXT_ARGS}"
