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
    echo "  export PATH=\"\${HOME}/cluster-toolkit:\${PATH}\""
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
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-gpt3-175b-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"
# libtpu-nightly build installed at container start (from the XPK recipe's
# test environment).
export LIBTPU_NIGHTLY_VERSION="20241028"


# XLA Flags
# Resolved from the `gpt_3_175b` model config in
# MaxText@e7292a3 benchmarks/maxtext_trillium_model_configs.py.
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=98304 \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_use_bundle_aware_cost_model_for_fusions=false "

# MaxText Workload Overrides
# Resolved from the `gpt_3_175b` model config in
# MaxText@e7292a3 benchmarks/maxtext_trillium_model_configs.py.
# Note: that config sets quantization=int8 (AQT int8), even though
# this recipe directory is named fp8.
MAXTEXT_ARGS="\
model_name=gpt3-175b \
per_device_batch_size=3 \
ici_fsdp_parallelism=-1 \
remat_policy=full \
attention=flash \
quantization=int8 \
gcs_metrics=True \
dataset_type=synthetic \
reuse_example_batch=1 \
enable_checkpointing=False \
sa_block_q=1024 \
sa_block_q_dkv=2048 \
sa_block_q_dq=2048 \
steps=30 \
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
  --compute-type ct6e-standard-4t \
  --topology 16x16 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && set -o pipefail && pip install libtpu-nightly==0.1.dev${LIBTPU_NIGHTLY_VERSION} -f https://storage.googleapis.com/libtpu-releases/index.html && \
export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
export TPU_PREMAPPED_BUFFER_SIZE=4294967296 && \
set +e; \
python3 -u MaxText/train.py MaxText/configs/base.yml ${MAXTEXT_ARGS} | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"
