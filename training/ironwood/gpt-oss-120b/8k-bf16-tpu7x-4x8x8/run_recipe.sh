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
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-gpt-oss-120b-8192-4x8x8")-$(date +%Y%m%d-%H%M)"

# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_impure_enable_packed_bf16_math_ops=true \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_max_concurrent_async_all_gathers=2 \
  --xla_max_concurrent_async_reduce_scatters=2 \
  --xla_tpu_aggressive_opt_barrier_removal=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=gpt-oss-120b \
per_device_batch_size=10.0 \
max_target_length=8192 \
skip_jax_distributed_system=True \
dtype=bfloat16 \
weight_dtype=float32 \
skip_first_n_steps_for_profiler=5 \
profile_periodically_period=10000 \
async_checkpointing=False \
enable_checkpointing=False \
use_custom_sort_vjp=True \
fsdp_shard_on_exp=True \
use_tokamax_gmm=True \
use_random_routing=True \
remat_policy=custom \
decoder_layer_input=offload \
mlpwo=offload \
opt_type=adamw \
steps=20 \
megablox=True \
sparse_matmul=True \
profiler=xplane \
use_tokamax_splash=True \
sa_block_q=1024 \
sa_block_kv=1024 \
sa_block_kv_compute=1024 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_q_dq=2048 \
sa_block_kv_dq=2048 \
sa_use_fused_bwd_kernel=True \
sa_q_layout=SEQ_MINOR \
sa_k_layout=SEQ_MINOR \
sa_v_layout=SEQ_MINOR \
attention=flash \
dcn_pipeline_parallelism=1 \
dcn_data_parallelism=-1 \
ici_pipeline_parallelism=1 \
ici_fsdp_transpose_parallelism=1 \
ici_fsdp_parallelism=64 \
ici_data_parallelism=8 \
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
