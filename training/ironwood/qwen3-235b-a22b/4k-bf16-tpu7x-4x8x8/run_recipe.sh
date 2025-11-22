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
    echo "pip install xpk==0.14.3"
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
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-qwen3-235b-a22b-4096-fsdp-4x8x8")-$(date +%Y%m%d-%H%M)"

# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=61440 \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_all_experimental_scheduler_features=true \
  --xla_tpu_enable_scheduler_memory_pressure_tracking=true \
  --xla_tpu_host_transfer_overlap_limit=24 \
  --xla_tpu_aggressive_opt_barrier_removal=ENABLED \
  --xla_lhs_prioritize_async_depth_over_stall=ENABLED \
  --xla_tpu_enable_ag_backward_pipelining=true \
  --xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
  --xla_should_add_loop_invariant_op_in_chain=ENABLED \
  --xla_max_concurrent_host_send_recv=100 \
  --xla_tpu_scheduler_percent_shared_memory_limit=100 \
  --xla_latency_hiding_scheduler_rerun=2 "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=qwen3-235b-a22b \
per_device_batch_size=16.0 \
max_target_length=4096 \
skip_jax_distributed_system=True \
dtype=bfloat16 \
weight_dtype=float32 \
skip_first_n_steps_for_profiler=5 \
profile_periodically_period=10000 \
async_checkpointing=False \
enable_checkpointing=False \
use_custom_sort_vjp=True \
use_tokamax_gmm=True \
use_random_routing=True \
remat_policy=custom \
decoder_layer_input=offload \
opt_type=adamw \
steps=20 \
megablox=False \
sparse_matmul=True \
profiler=xplane \
wi_tile_fwd_mlp_dim=2048 \
wi_tile_dlhs_mlp_dim=2048 \
wi_tile_drhs_mlp_dim=2048 \
wo_tile_fwd_mlp_dim=2048 \
wo_tile_dlhs_mlp_dim=2048 \
wo_tile_drhs_mlp_dim=2048 \
use_tokamax_splash=True \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=1024 \
sa_block_q_dkv=2048 \
sa_block_q_dq=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dq=2048 \
sa_block_kv_dkv_compute=1024 \
sa_use_fused_bwd_kernel=True \
attention=flash \
dcn_pipeline_parallelism=1 \
dcn_data_parallelism=-1 \
ici_pipeline_parallelism=1 \
ici_fsdp_transpose_parallelism=1 \
ici_fsdp_parallelism=512 \
dataset_type=synthetic \
dataset_path=gs://max-datasets-rogue \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"

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
