#!/bin/bash

# --- Environment Setup ---
# This script requires uv and a Python 3.13 virtual environment with xpk installed.
# If you haven't set up uv and the environment, please refer to the README.md.

UV_VENV_PATH="${HOME}/.local/bin/venv"
UV_PYTHON_VERSION="3.13"

# Activate the virtual environment
source "${UV_VENV_PATH}/bin/activate"

# Check if xpk is installed in the venv
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the virtual environment. Please install it by running:"
    echo "pip install xpk==1.11.0"
    exit 1
fi
# --- End Environment Setup ---

set -e
set -o pipefail

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---

# --- Environment Variables ---
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE=""
export BASE_OUTPUT_DIR=""
export ARTIFACT_DIR=""
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-gemma4-2b-8192-4x4x4")-$(date +%Y%m%d-%H%M)"


# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_use_tc_device_shape_on_sc=true \
  --xla_sc_disable_megacore_partitioning=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_all_gather_offload_tracing=true "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=gemma4-e2b \
skip_jax_distributed_system=True \
scan_layers=False \
dtype=bfloat16 \
per_device_batch_size=2 \
max_target_length=8192 \
async_checkpointing=False \
enable_checkpointing=False \
use_iota_embed=True \
num_vocab_tiling=8 \
remat_policy=full \
allow_split_physical_axes=True \
ici_data_parallelism=4 \
ici_fsdp_parallelism=-1 \
attention=flash \
use_tokamax_splash=True \
sa_use_fused_bwd_kernel=True \
sa_block_q=1024 \
sa_block_kv=1024 \
sa_block_kv_compute=512 \
sa_block_q_dkv=1024 \
sa_block_kv_dkv=1024 \
sa_block_kv_dkv_compute=256 \
dataset_type=synthetic \
opt_type=adamw \
steps=30 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} \
profiler=xplane \
skip_first_n_steps_for_profiler=5 \
profiler_steps=3"



echo "=== Creating XPK Workload: $WORKLOAD_NAME ==="
xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=medium \
  --max-restarts=0 \
  --device-type=tpu7x-4x4x4 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
   \
   \
  --workload="${WORKLOAD_NAME}" \
   \
  --command="set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
 \
 \
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log && \
gcloud storage cp --no-user-output-enabled train.log ${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID}.log"