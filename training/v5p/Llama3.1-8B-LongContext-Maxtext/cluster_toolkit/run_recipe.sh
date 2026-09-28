#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Environment Setup ---
# This script requires the Cluster Toolkit (gcluster) CLI (v1.104.0).
# If you haven't installed gcluster, please refer to the README.md.

export PATH="${HOME}/cluster-toolkit:${PATH}"
CTK_VERSION="1.104.0"
GCLUSTER_BIN="${GCLUSTER_BIN:-gcluster}"
if ! command -v "${GCLUSTER_BIN}" &> /dev/null; then
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
# Before running this script, export the environment variables below in your
# shell (see README.md), or edit the defaults here.
# ---

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-}"
export CLUSTER_NAME="${CLUSTER_NAME:-}"
export ZONE="${ZONE:-}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-l8b-lc-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"

# Default workload configs (10M sequence length on 128 chips), override by
# passing KEY=VALUE arguments. ICI_CONTEXT_PARALLELISM * ICI_FSDP_PARALLELISM
# must equal the number of chips (128); see README.md for other lengths.
export MAX_TARGET_LENGTH=10485760
export ICI_CONTEXT_PARALLELISM=128
export ICI_FSDP_PARALLELISM=1
export PER_DEVICE_BATCH_SIZE=0.0078125

for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65472 \
  --xla_tpu_use_enhanced_launch_barrier=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_enable_async_collective_permute=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true "

# NOTE: this is a Llama3.1-8B *variant*, not the stock architecture: the attention
# shape is overridden to 16 query heads / 4 KV heads / head_dim 256 (stock is
# 32 / 8 / 128). Total attention width (16 * 256 = 4096) and the 4:1 GQA ratio are
# unchanged, so the parameter count matches stock Llama3.1-8B.
MAXTEXT_ARGS="\
model_name=llama3.1-8b \
steps=8 \
enable_checkpointing=false \
override_model_config=true \
head_dim=256 \
base_num_query_heads=16 \
base_num_kv_heads=4 \
num_vocab_tiling=16 \
per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
max_target_length=${MAX_TARGET_LENGTH} \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} \
dataset_type=synthetic \
packing=false \
attention=flash \
use_tokamax_splash=true \
use_jax_splash=false \
context_parallel_strategy=ring \
context_parallel_load_balance=true \
ici_context_parallelism=${ICI_CONTEXT_PARALLELISM} \
ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
ici_tensor_parallelism=1 \
remat_policy=custom \
context=device \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
ring_scan_unroll=16 \
dq_reduction_steps=3"

echo "=== Creating Cluster Toolkit Workload: $WORKLOAD_NAME ==="
"${GCLUSTER_BIN}" job submit \
  --skip-prereqs \
  --queue multislice-queue \
  --cluster "$CLUSTER_NAME" \
  --project "$PROJECT_ID" \
  --location "$ZONE" \
  --priority medium \
  --restarts 0 \
  --compute-type ct5p-hightpu-4t \
  --topology 4x4x8 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
set +e; \
python3 -u -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"
