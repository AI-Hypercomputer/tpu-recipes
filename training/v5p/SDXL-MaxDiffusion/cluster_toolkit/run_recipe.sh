#!/bin/bash

set -eo pipefail

# --- Environment Setup ---
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

BASE_OUTPUT_DIR=${1:-${BASE_OUTPUT_DIR:-}}
COMMITS=${2:-${COMMITS:-00150750841e9155669fd1ac4c6f2fcd0e0654e0}}

# Set environment variables from KEY=VALUE arguments
for ARGUMENT in "$@"; do
    if [[ "$ARGUMENT" == *=* ]]; then
        IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
        export "$KEY"="$VALUE"
    fi
done

export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_megacore_fusion=false --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true'
LIBTPU_INIT_ARGS+=' --xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true'
LIBTPU_INIT_ARGS+=' --xla_enable_async_reduce_scatter_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true'
LIBTPU_INIT_ARGS+=' --xla_tpu_spmd_threshold_for_allgather_cse=1000000 --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000'

export PROJECT_ID="${PROJECT_ID:-}"
export CLUSTER_NAME="${CLUSTER_NAME:-}"
export ZONE="${ZONE:-}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-}"

for var in PROJECT_ID CLUSTER_NAME ZONE BASE_OUTPUT_DIR WORKLOAD_IMAGE; do
    if [[ -z "${!var}" ]]; then
        echo "Error: Environment variable $var is required but not set." >&2
        exit 1
    fi
done

if [[ ! "${BASE_OUTPUT_DIR}" =~ ^gs:// ]]; then
    echo "Error: BASE_OUTPUT_DIR must be a GCS path starting with 'gs://'." >&2
    exit 1
fi
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR%/}"

CLEAN_USER=$(echo "${USER:-workload}" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr -cd 'a-z0-9-' | cut -c1-6)
CLEAN_USER="${CLEAN_USER:-workload}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-${CLEAN_USER}-sdxl-v5p-$(date +%d%H%M%S)}"

"${GCLUSTER_BIN}" job submit \
  --skip-prereqs \
  --queue "${QUEUE:-multislice-queue}" \
  --cluster "${CLUSTER_NAME}" \
  --project "${PROJECT_ID}" \
  --location "${ZONE}" \
  --compute-type tpu-v5p \
  --topology 4x4x4 \
  --num-slices "${NUM_SLICES:-1}" \
  --image "${WORKLOAD_IMAGE}" \
  --gke-namespace "${NAMESPACE:-default}" \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && export LIBTPU_INIT_ARGS='${LIBTPU_INIT_ARGS}' && \
if [ -d maxdiffusion ]; then \
  cd maxdiffusion && (git checkout ${COMMITS} || (git fetch origin && git checkout ${COMMITS})); \
elif [ -d .git ] || git rev-parse --is-inside-work-tree &>/dev/null; then \
  git checkout ${COMMITS} || (git fetch origin && git checkout ${COMMITS}); \
else \
  git clone https://github.com/google/maxdiffusion.git && \
  cd maxdiffusion && \
  git checkout ${COMMITS}; \
fi && \
python src/maxdiffusion/train_sdxl.py src/maxdiffusion/configs/base_xl.yml revision=refs/pr/95 activations_dtype=bfloat16 weights_dtype=bfloat16 resolution=1024 per_device_batch_size=8 output_dir=${BASE_OUTPUT_DIR} jax_cache_dir=${BASE_OUTPUT_DIR}/cache_dir/ max_train_steps=5000 attention=flash run_name=sdxl-fsdp-v5p-ddp"
