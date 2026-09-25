#!/bin/bash

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

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-}"
export CLUSTER_NAME="${CLUSTER_NAME:-}"
export ZONE="${ZONE:-}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-}"

# Validate required environment variables
for var in PROJECT_ID CLUSTER_NAME ZONE BASE_OUTPUT_DIR WORKLOAD_IMAGE; do
    if [[ -z "${!var}" ]]; then
        echo "Error: Environment variable $var is required but not set." >&2
        exit 1
    fi
done

if [[ ! "${BASE_OUTPUT_DIR}" =~ ^gs:// ]]; then
    echo "Error: BASE_OUTPUT_DIR must start with 'gs://'" >&2
    exit 1
fi

CLEAN_USER=$(echo "${USER:-workload}" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr -cd 'a-z0-9-' | cut -c1-8)
CLEAN_USER="${CLEAN_USER:-workload}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-${CLEAN_USER}-llama8b-v6e64-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"

# XLA Flags (100% matching training/trillium/Llama3.1-8B-MaxText/v6e-64/README.md:
# DENSE_VMEM_LIMIT_FLAG + LAYOUT_FOR_ALL_REDUCE_SCATTER + DATA_PARALLEL_OVERLAP +
# CF_FOR_ALL_GATHER + HOST_OFFLOAD_FLAGS)
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=98304 \
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true \
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
  --xla_tpu_assign_all_reduce_scatter_layout=true \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_all_experimental_scheduler_features=true \
  --xla_tpu_enable_scheduler_memory_pressure_tracking=true \
  --xla_tpu_host_transfer_overlap_limit=24 \
  --xla_tpu_aggressive_opt_barrier_removal=ENABLED \
  --xla_lhs_prioritize_async_depth_over_stall=ENABLED \
  --xla_tpu_enable_ag_backward_pipelining=true \
  --xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
  --xla_should_add_loop_invariant_op_in_chain=ENABLED \
  --xla_max_concurrent_host_send_recv=100 \
  --xla_tpu_scheduler_percent_shared_memory_limit=100 \
  --xla_latency_hiding_scheduler_rerun=2 "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=llama3.1-8b \
per_device_batch_size=5 \
ici_fsdp_parallelism=-1 \
remat_policy=custom \
decoder_layer_input=offload \
out_proj=offload \
query_proj=offload \
key_proj=offload \
value_proj=offload \
max_target_length=8192 \
attention=flash \
use_iota_embed=True \
dataset_path=gs://max-datasets-rogue \
dataset_type=synthetic \
enable_checkpointing=False \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_q_dq=2048 \
sa_block_kv_dq=2048 \
sa_use_fused_bwd_kernel=True \
steps=30 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} \
profiler=xplane \
skip_first_n_steps_for_profiler=10 \
profiler_steps=5"

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
  --topology 8x8 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "bash -c 'set -e && set -o pipefail && \\
export ENABLE_PATHWAYS_PERSISTENCE=\"1\" && \\
export LIBTPU_INIT_ARGS=\"${XLA_FLAGS}\" && \\
export ARTIFACT_DIR=\"${ARTIFACT_DIR}\" && \\
export JAX_PLATFORMS=\"tpu,cpu\" && \\
export ENABLE_PJRT_COMPATIBILITY=\"true\" && \\
set +e; \\
if [ -f MaxText/train.py ]; then \\
  python3 -u MaxText/train.py MaxText/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee train.log; \\
else \\
  python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee train.log; \\
fi; \\
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \\
if [ -s train.log ]; then \\
  if command -v gcloud &> /dev/null; then \\
    timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \\
  elif command -v gsutil &> /dev/null; then \\
    timeout 30s gsutil cp train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \\
  fi; \\
fi; \\
exit \${TRAIN_EXIT_CODE}'"
