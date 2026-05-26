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
    echo "pip install xpk==1.4.0"
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
export OUTPUT_DIR=""
export ARTIFACT_DIR=""
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-wan")-$(date +%Y%m%d-%H%M)"
# DATASET_DIR is where pre-training data was uploaded
export DATASET_DIR=${ARTIFACT_DIR}/PusaV1_training
export CHECKPOINT_BUCKET=''
export DATA_BUCKET=''


# XLA Flags
XLA_FLAGS=" \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_enable_async_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_max_concurrent_async_all_gathers=4 \
  --xla_tpu_enable_async_all_to_all=true \
  --xla_latency_hiding_scheduler_rerun=5 \
  --xla_tpu_rwb_fusion=false \
  --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false \
  --xla_tpu_impure_enable_packed_bf16_math_ops=false \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true \
  --xla_enable_transpose_trace=false "

# MaxDiffusion Workload Overrides
MAXDIFFUSION_ARGS="\
model_name=wan2.1 \
attention=flash \
weights_dtype=bfloat16 \
activations_dtype=bfloat16 \
guidance_scale=5.0 \
flow_shift=5.0 \
fps=16 \
skip_jax_distributed_system=False \
train_data_dir=/mnt/data/wan_tfr_dataset_pusa_v1 \
load_tfrecord_cached=True \
height=1280 \
width=720 \
num_frames=81 \
num_inference_steps=50 \
prompt='a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.' \
enable_profiler=True \
dataset_save_location=/mnt/data/wan_tfr_dataset_pusa_v1 \
remat_policy=FULL \
flash_min_seq_length=0 \
seed=123456789 \
skip_first_n_steps_for_profiler=5 \
profiler_steps=10 \
per_device_batch_size=0.25 \
ici_data_parallelism=32 \
ici_context_parallelism=4 \
ici_tensor_parallelism=1 \
allow_split_physical_axes=True \
flash_block_sizes='{\"block_q\":2048,\"block_kv_compute\":512,\"block_kv\":2048,\"block_q_dkv\":2048,\"block_kv_dkv\":2048,\"block_kv_dkv_compute\":512,\"use_fused_bwd_kernel\":true}' \
checkpoint_every=30 \
enable_single_replica_ckpt_restoring=True \
max_train_steps=150 \
base_output_directory=${ARTIFACT_DIR} \
output_dir=${OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"

echo "=== Attaching storage volume maxdiffusion-data ==="
xpk storage attach ${WORKLOAD_NAME}-maxdiffusion-data --type=gcsfuse --project=${PROJECT_ID} --cluster=${CLUSTER_NAME} --zone=${ZONE} --mount-point=/mnt/data --bucket=${DATA_BUCKET} --mount-options=logging:severity:info --readonly=false --auto-mount=false --size=1_000_000 
echo "=== Attaching storage volume checkpoint ==="
xpk storage attach ${WORKLOAD_NAME}-checkpoint --type=gcsfuse --project=${PROJECT_ID} --cluster=${CLUSTER_NAME} --zone=${ZONE} --mount-point=/mnt/ckpt --bucket=${CHECKPOINT_BUCKET} --mount-options=logging:severity:info,enable-atomic-rename-object:false --readonly=false --auto-mount=false --size=1_000_000 

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
  --storage=${WORKLOAD_NAME}-maxdiffusion-data \
  --storage=${WORKLOAD_NAME}-checkpoint \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && \
export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export JAX_PLATFORMS='tpu,cpu' && \
export ENABLE_PJRT_COMPATIBILITY='true' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
pip install . && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
echo 'Starting WAN training ...' && \
 \
HF_HUB_CACHE=/dev/shm python3 -m src.maxdiffusion.train_wan \
  src/maxdiffusion/configs/base_wan_14b.yml \
  train_data_dir=${DATASET_DIR} \
  jax_cache_dir=${OUTPUT_DIR}/jax_cache/ \
  dataset_save_location=${DATASET_DIR} \
  run_name=${WORKLOAD_NAME} \
  ${MAXDIFFUSION_ARGS}"
