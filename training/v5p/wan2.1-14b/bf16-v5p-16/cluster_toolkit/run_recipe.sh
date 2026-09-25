#!/bin/bash

# --- Environment Setup ---
# This script requires Cluster Toolkit (gcluster v1.104.0) installed.
# If you haven't set up gcluster and the environment, please refer to the README.md.

export PATH="${HOME}/cluster-toolkit:${PATH}"
GCLUSTER_BIN="${GCLUSTER_BIN:-gcluster}"

# Check if gcluster is installed in PATH
if ! command -v "${GCLUSTER_BIN}" &> /dev/null && [[ ! -x "${GCLUSTER_BIN}" ]]; then
    echo "gcluster not found in PATH. Please install Cluster Toolkit v1.104.0 per README.md."
    exit 1
fi
# --- End Environment Setup ---

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
#
# This is a v5p port of the Ironwood (tpu7x) wan2.1-14b recipe.
# Target: 1 x v5p-16 slice (single slice, 8 chips, topology 2x2x2, 2 hosts).
#   - v5p uses megacore: 1 JAX device per chip, so v5p-16 = 8 devices.
#   - All 8 chips are on the same slice -> all collectives run over ICI (no DCN).
# Mesh (single slice, ICI only):
#   ici_data=1 * ici_fsdp=8 * ici_tensor=1 = 8 devices.
#   No dcn_* parallelism (single slice).
#
# NOTE: requires a v5p-16 (multi-host, ct5p-hightpu-4t x2, 2x2x2) node pool in the
# cluster. Unlike the 2x v5p-8 multislice variant, there is no cross-slice DCN here,
# so the "Too many pending sends" DCN backpressure seen in the multislice run is
# not expected; ICI-only collectives should yield higher MFU.
# ---

# --- Environment Variables ---
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE=""
export BASE_OUTPUT_DIR=""
export WORKLOAD_IMAGE=""  # must be pushed: docker push <this>

export WORKLOAD_NAME="$(printf "%.14s" "${USER//_/-}-wan21")-$(date +%Y%m%d-%H%M)"
# DATASET_DIR is where the preprocessed tfrecords were uploaded (NOT the raw HF dataset).
export DATASET_DIR=${BASE_OUTPUT_DIR}/wan_tfr_dataset_pusa_v1

# XLA Flags — v5p-safe, generic async-collective + latency-hiding set.
# The full tpu7x (Ironwood) flag set (SparseCore offload + continuation-fusion)
# HANGS on the v5p 2x2x2 single-slice ICI collectives ("viperfish ... hang, SDC"
# warning). Keep this trimmed set for v5p-16.
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65472 \
  --xla_enable_async_all_gather=true \
  --xla_enable_async_all_reduce=true \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_all_to_all=true \
  --xla_max_concurrent_async_all_gathers=4 \
  --xla_latency_hiding_scheduler_rerun=5 \
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
max_train_steps=30 \
enable_profiler=True \
dataset_save_location=${DATASET_DIR} \
remat_policy=FULL \
flash_min_seq_length=0 \
seed=123456789 \
skip_first_n_steps_for_profiler=5 \
profiler_steps=10 \
per_device_batch_size=0.5 \
ici_data_parallelism=1 \
ici_fsdp_parallelism=8 \
ici_tensor_parallelism=1 \
allow_split_physical_axes=True \
checkpoint_every=10 \
save_final_checkpoint=True \
checkpoint_dir=${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}/checkpoints \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"

"${GCLUSTER_BIN}" job submit --skip-prereqs --queue multislice-queue \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --location=$ZONE \
  --priority=very-high \
  --restarts=0 \
  --compute-type=tpu-v5p --topology=2x2x2 \
  --num-slices=1 \
  --image="${WORKLOAD_IMAGE}" \
  --verbose --gke-namespace=default \
  --name="${WORKLOAD_NAME}" \
  --command="set -e && \
export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export JAX_PLATFORMS='tpu,cpu' && \
export ENABLE_PJRT_COMPATIBILITY='true' && \
pip install . && \
pip install chex && \
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
