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
export BASE_OUTPUT_DIR=""
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-qwen3-fp8-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"

# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=61440 \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
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
model_name=qwen3-235b-a22b \
per_device_batch_size=16.0 \
max_target_length=4096 \
skip_jax_distributed_system=True \
dtype=bfloat16 \
weight_dtype=float32 \
opt_type=adamw \
steps=20 \
profiler=xplane \
skip_first_n_steps_for_profiler=5 \
profiler_steps=3 \
profile_periodically_period=10000 \
async_checkpointing=False \
enable_checkpointing=False \
remat_policy=custom \
decoder_layer_input=offload \
use_custom_sort_vjp=True \
use_random_routing=True \
fsdp_shard_on_exp=False \
megablox=False \
sparse_matmul=True \
use_tokamax_gmm=True \
use_tokamax_splash=True \
sa_use_fused_bwd_kernel=True \
attention=flash \
sa_block_q=1024 \
sa_block_kv=1024 \
sa_block_kv_compute=512 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=1024 \
sa_block_q_dq=1024 \
sa_block_kv_dq=1024 \
sa_q_layout=HEAD_DIM_MINOR \
sa_k_layout=SEQ_MINOR \
sa_v_layout=HEAD_DIM_MINOR \
dcn_pipeline_parallelism=1 \
dcn_data_parallelism=-1 \
ici_pipeline_parallelism=1 \
ici_fsdp_transpose_parallelism=1 \
ici_fsdp_parallelism=-1 \
use_qwix_quantization=True \
quantization=fp8_full \
weight_quantization_calibration_method=fixed,-224,224 \
act_quantization_calibration_method=fixed,-224,224 \
wi_tile_fwd_batch_seq=512 \
wi_tile_fwd_embed_dim=4096 \
wi_tile_fwd_mlp_dim=1536 \
wi_tile_dlhs_batch_seq=512 \
wi_tile_dlhs_embed_dim=1536 \
wi_tile_dlhs_mlp_dim=4096 \
wi_tile_drhs_batch_seq=1024 \
wi_tile_drhs_embed_dim=1024 \
wi_tile_drhs_mlp_dim=1536 \
wo_tile_fwd_batch_seq=512 \
wo_tile_fwd_embed_dim=1536 \
wo_tile_fwd_mlp_dim=4096 \
wo_tile_dlhs_batch_seq=512 \
wo_tile_dlhs_embed_dim=4096 \
wo_tile_dlhs_mlp_dim=1536 \
wo_tile_drhs_batch_seq=512 \
wo_tile_drhs_embed_dim=768 \
wo_tile_drhs_mlp_dim=4096 \
dataset_type=synthetic \
dataset_path=gs://max-datasets-rogue \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} "

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
  --topology 4x8x8 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --gke-disable-parallel-containers \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
set +e; \
python3 -u -m MaxText.train MaxText/configs/base.yml ${MAXTEXT_ARGS} | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"
