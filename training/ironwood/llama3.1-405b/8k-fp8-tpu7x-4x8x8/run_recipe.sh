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
    echo "pip install xpk==0.16.1"
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
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-llama3-1-405b-8192-fp8-4x8x8")-$(date +%Y%m%d-%H%M)"

# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_impure_enable_packed_bf16_math_ops=true \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_scoped_vmem_limit_kib=65536 "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=llama3.1-405b \
skip_jax_distributed_system=True \
dtype=bfloat16 \
per_device_batch_size=3 \
profile_periodically_period=10000 \
async_checkpointing=False \
enable_checkpointing=False \
use_iota_embed=True \
remat_policy=custom \
decoder_layer_input=offload \
ici_fsdp_parallelism=-1 \
dataset_type=synthetic \
opt_type=adamw \
quantization=fp8_full \
use_qwix_quantization=True \
mu_dtype=bfloat16 \
use_tokamax_splash=True \
use_max_logit_estimate=-1 \
sa_block_kv=2048 \
sa_block_kv_compute=256 \
sa_block_q=1024 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=1024 \
sa_block_q_dkv=2048 \
sa_k_layout=SEQ_MINOR \
sa_q_layout=HEAD_DIM_MINOR \
sa_v_layout=SEQ_MINOR \
attention=flash \
sa_use_fused_bwd_kernel=True \
max_target_length=8192 \
profiler=xplane \
weight_quantization_calibration_method=fixed,-224,224 \
act_quantization_calibration_method=fixed,-224,224 \
steps=30 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} \
output_dir=${BASE_OUTPUT_DIR}"

xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --device-type=tpu7x-4x8x8 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -m MaxText.train MaxText/configs/base.yml ${MAXTEXT_ARGS}"