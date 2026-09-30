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
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-sdxl-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"
# MaxDiffusion commit to train with.
export COMMITS="${COMMITS:-00150750841e9155669fd1ac4c6f2fcd0e0654e0}"

XLA_FLAGS=" \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_megacore_fusion=false \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_megacore_fusion_allow_ags=false \
  --xla_enable_async_collective_permute=true \
  --xla_tpu_enable_ag_backward_pipelining=true \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true \
  --xla_enable_async_all_reduce=true \
  --xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true \
  --xla_tpu_mosaic_fusion=true \
  --xla_enable_async_reduce_scatter_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true \
  --xla_tpu_spmd_threshold_for_allgather_cse=1000000 \
  --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000 "

MAXDIFFUSION_ARGS="\
revision=refs/pr/95 \
activations_dtype=bfloat16 \
weights_dtype=bfloat16 \
resolution=1024 \
per_device_batch_size=8 \
output_dir=${BASE_OUTPUT_DIR} \
jax_cache_dir=${BASE_OUTPUT_DIR}/cache_dir/ \
max_train_steps=5000 \
attention=flash \
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
  --compute-type ct5p-hightpu-4t \
  --topology 4x4x4 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
rm -rf maxdiffusion && \
git clone https://github.com/google/maxdiffusion.git && \
cd maxdiffusion && \
git checkout ${COMMITS} && \
set +e; \
python3 -u src/maxdiffusion/train_sdxl.py src/maxdiffusion/configs/base_xl.yml ${MAXDIFFUSION_ARGS} | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"
