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


# XLA Flags (from MaxText configs/tpu/v5p/gpt3_175b/gpt3_175b_base.sh)
# The async collective fusion flags from the original script
# (--xla_tpu_enable_async_collective_fusion, ..._fuse_all_gather and
# ..._multiple_steps) are omitted: current libtpu rejects async collective
# fusion on TPU v5p at backend initialization.
XLA_FLAGS=" \
  --xla_tpu_enable_experimental_fusion_cost_model=false \
  --xla_tpu_dot_dot_fusion_duplicated=false \
  --xla_tpu_dot_dot_fusion=false \
  --xla_jf_conv_input_fusion=true \
  --xla_jf_conv_output_fusion=false \
  --xla_tpu_rwb_fusion=false \
  --xla_tpu_copy_fusion_pad_unpad_ratio=300 \
  --xla_tpu_enable_aggressive_loop_fusion_layout_opt=false \
  --xla_tpu_enable_copy_fusion=false \
  --xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=false \
  --xla_tpu_scavenge_vmem_for_fusions=false \
  --xla_tpu_vector_load_fusion_window=256 \
  --xla_tpu_vector_store_fusion_window=64 \
  --xla_tpu_decompose_all_gather_einsum=true \
  --xla_tpu_spmd_rng_bit_generator_unsafe=true \
  --xla_tpu_enable_megacore_fusion=true \
  --xla_enable_async_all_gather=true \
  --xla_enable_async_collective_permute=true \
  --xla_always_enable_all_gather_2d_asymmetric=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_tpu_dcn_max_overlap_estimation=32 "

# MaxText Workload Overrides (from MaxText configs/tpu/v5p/gpt3_175b/v5p_1024.sh)
MAXTEXT_ARGS="\
model_name=gpt3-175b \
enable_checkpointing=false \
async_checkpointing=false \
per_device_batch_size=4 \
ici_data_parallelism=1 \
ici_fsdp_parallelism=64 \
ici_tensor_parallelism=8 \
remat_policy=full \
attention=flash \
quantization=int8 \
dataset_type=synthetic \
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
  --compute-type v5p \
  --topology 8x8x8 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && set -o pipefail && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
set +e; \
python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"
