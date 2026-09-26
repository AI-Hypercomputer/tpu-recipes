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
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-llama4-maverick-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"


# XLA Flags (from MaxText benchmarks/maxtext_v5p_model_configs.py `llama4_maverick_dropless_v5p_256` @ tpu-recipes-v0.1.3):
# MOE_VMEM_LIMIT_FLAG + CF_FOR_ALL_GATHER + DATA_PARALLEL_OVERLAP + LAYOUT_FOR_ALL_REDUCE_SCATTER + HOST_OFFLOAD_FLAGS
# The async collective fusion flags from CF_FOR_ALL_GATHER
# (--xla_tpu_enable_async_collective_fusion, ..._fuse_all_gather and
# ..._multiple_steps) are omitted: current libtpu rejects async collective
# fusion on TPU v5p at backend initialization.
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=81920 \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true \
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
  --xla_tpu_assign_all_reduce_scatter_layout=true \
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

# MaxText Workload Overrides (from MaxText benchmarks/maxtext_v5p_model_configs.py `llama4_maverick_dropless_v5p_256` @ tpu-recipes-v0.1.3)
MAXTEXT_ARGS="\
model_name=llama4-17b-128e \
per_device_batch_size=4 \
max_target_length=8192 \
ici_fsdp_parallelism=32 \
ici_tensor_parallelism=4 \
enable_checkpointing=false \
dtype=bfloat16 \
weight_dtype=float32 \
megablox=true \
sparse_matmul=true \
dataset_type=synthetic \
opt_type=adamw \
skip_first_n_steps_for_profiler=5 \
profiler_steps=3 \
profiler=xplane \
remat_policy=custom \
decoder_layer_input=offload \
out_proj=offload \
query_proj=offload \
key_proj=offload \
value_proj=offload \
reuse_example_batch=1 \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_q_dq=2048 \
sa_block_kv_dq=2048 \
tokenizer_path=meta-llama/Llama-4-Maverick-17B-128E \
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
  --topology 4x4x8 \
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
python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"
