#!/bin/bash

# --- Environment Setup ---
# This script requires the Cluster Toolkit (gcluster) CLI (v1.104.0).
# If you haven't installed gcluster, please refer to the README.md.

export PATH="${HOME}/cluster-toolkit:${PATH}"
CTK_VERSION="1.104.0"
GCLUSTER_BIN="${GCLUSTER_BIN:-gcluster}"
if ! command -v "${GCLUSTER_BIN}" &> /dev/null && [[ ! -x "${GCLUSTER_BIN}" ]]; then
    echo "gcluster not found. Please install Cluster Toolkit v${CTK_VERSION} by running:"
    echo "  mkdir -p \${HOME}/cluster-toolkit"
    echo "  curl -Lo /tmp/gcluster_bundle.tgz https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/download/v${CTK_VERSION}/gcluster_bundle_linux_amd64.tgz"
    echo "  tar -xzf /tmp/gcluster_bundle.tgz -C \${HOME}/cluster-toolkit gcluster"
    echo "  rm -f /tmp/gcluster_bundle.tgz"
    echo "  chmod +x \${HOME}/cluster-toolkit/gcluster"
    echo '  export PATH="${HOME}/cluster-toolkit:${PATH}"'
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
export BASE_OUTPUT_DIR="" # for example, gs://<your_gcs_bucket>
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-wan2-1-$(date +%H%M)}"
# DATASET_DIR is where pre-training data was uploaded.
export DATASET_DIR="${DATASET_DIR:-${BASE_OUTPUT_DIR}/PusaV1_training}"

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
output_dir=${BASE_OUTPUT_DIR} \
train_data_dir=${DATASET_DIR} \
load_tfrecord_cached=True \
height=1280 \
width=720 \
num_frames=81 \
num_inference_steps=50 \
prompt='a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.' \
jax_cache_dir=${BASE_OUTPUT_DIR}/jax_cache/ \
max_train_steps=150 \
enable_profiler=True \
dataset_save_location=${DATASET_DIR} \
remat_policy=FULL \
flash_min_seq_length=0 \
seed=123456789 \
skip_first_n_steps_for_profiler=5 \
profiler_steps=10 \
per_device_batch_size=0.25 \
ici_data_parallelism=32 \
ici_fsdp_parallelism=4 \
ici_tensor_parallelism=1 \
allow_split_physical_axes=True \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"

echo "=== Creating Cluster Toolkit Workload: $WORKLOAD_NAME ==="
"${GCLUSTER_BIN}" job submit \
  --skip-prereqs \
  --queue multislice-queue \
  --cluster "$CLUSTER_NAME" \
  --project "$PROJECT_ID" \
  --location "$ZONE" \
  --priority medium \
  --restarts 0 \
  --compute-type tpu7x \
  --topology 4x4x4 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && \
export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export JAX_PLATFORMS='tpu,cpu' && \
export ENABLE_PJRT_COMPATIBILITY='true' && \
pip install . && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
echo 'Starting WAN training ...' && \
HF_HUB_CACHE=/dev/shm python3 -m src.maxdiffusion.train_wan \
  src/maxdiffusion/configs/base_wan_14b.yml \
  output_dir=${BASE_OUTPUT_DIR} \
  train_data_dir=${DATASET_DIR} \
  jax_cache_dir=${BASE_OUTPUT_DIR}/jax_cache/ \
  dataset_save_location=${DATASET_DIR} \
  base_output_directory=${BASE_OUTPUT_DIR} \
  run_name=${WORKLOAD_NAME} \
  ${MAXDIFFUSION_ARGS}"
