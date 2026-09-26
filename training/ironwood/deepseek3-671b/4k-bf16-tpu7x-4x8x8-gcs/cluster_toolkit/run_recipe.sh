#!/bin/bash

# --- Environment Setup ---
# This script requires the Cluster Toolkit (gcluster) CLI (v1.105.0).
# If you haven't installed gcluster, please refer to the README.md.

export PATH="${HOME}/cluster-toolkit:${PATH}"
CTK_VERSION="1.105.0"
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
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-dsv3-gcs-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"

# Dataset bucket (name only, no gs:// prefix) and the path it is mounted at
# inside the container. gcluster mounts it inline with the GCS FUSE CSI driver;
# the mount options and volume attributes below match the GCS FUSE
# PersistentVolume that the XPK version of this recipe provisioned.
export DATASET_BUCKET=""
export DATASET_BUCKET_MOUNTED_PATH="/tmp/dataset"
DATASET_MOUNT_OPTIONS="implicit-dirs,metadata-cache:negative-ttl-secs:0,metadata-cache:ttl-secs:-1,metadata-cache:stat-cache-max-size-mb:-1,metadata-cache:type-cache-max-size-mb:-1,write:enable-streaming-writes:true"
DATASET_VOLUME_ATTRIBUTES="gcsfuseMetadataPrefetchOnMount=true"


# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_dvfs_p_state=3 \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_aggressive_opt_barrier_removal=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=deepseek3-671b \
per_device_batch_size=8.0 \
max_target_length=4096 \
dcn_pipeline_parallelism=1 \
dcn_data_parallelism=-1 \
ici_pipeline_parallelism=1 \
ici_fsdp_transpose_parallelism=2 \
ici_fsdp_parallelism=-1 \
allow_split_physical_axes=True \
use_iota_embed=True \
remat_policy=custom \
decoder_layer_input=offload \
opt_type=adamw \
mu_dtype=bfloat16 \
grad_dtype=bfloat16 \
use_random_routing=True \
megablox=True \
sparse_matmul=True \
use_custom_sort_vjp=True \
shard_exp_on_fsdp=False \
use_2d_fsdp_sharding=True \
sa_use_fused_bwd_kernel=True \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_kv_dq=2048 \
sa_block_q_dq=2048 \
attention=flash \
use_tokamax_splash=True \
use_max_logit_estimate=-1 \
cost_estimate_flops_fwd=5000000000000 \
cost_estimate_flops_bwd=5000000000000 \
float32_weight_sum=False \
use_tokamax_gmm=True \
tokenizer_path='deepseek-ai/DeepSeek-V3-Base' \
tokenizer_type=huggingface \
enable_checkpointing=True \
checkpoint_storage_concurrent_gb=400 \
async_checkpointing=true \
enable_single_replica_ckpt_restoring=true \
checkpoint_storage_target_data_file_size_bytes=209715200 \
dataset_type='grain' \
grain_file_type=arrayrecord \
grain_train_files=${DATASET_BUCKET_MOUNTED_PATH}/multilingual-c4/array-record/c4/multilingual/3.0.1/*.arrayrecord \
grain_worker_count=2 \
steps=30 \
checkpoint_period=25 \
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
  --topology 4x8x8 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --mount "gs://${DATASET_BUCKET};${DATASET_BUCKET_MOUNTED_PATH};ro;options=${DATASET_MOUNT_OPTIONS};attributes=${DATASET_VOLUME_ATTRIBUTES}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
set +e; \
python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-\${JOB_COMPLETION_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"
