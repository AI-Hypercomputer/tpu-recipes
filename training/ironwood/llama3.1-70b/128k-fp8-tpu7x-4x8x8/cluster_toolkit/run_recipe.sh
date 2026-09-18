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
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-l70b-128k-fp8-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"

# XLA Flags
XLA_FLAGS=" \
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
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=llama3.1-70b \
skip_jax_distributed_system=True \
dtype=bfloat16 \
per_device_batch_size=0.125 \
profile_periodically_period=10000 \
async_checkpointing=False \
enable_checkpointing=False \
use_iota_embed=True \
remat_policy=custom \
decoder_layer_input=device \
context=device \
query_proj=device \
key_proj=device \
value_proj=device \
mlpwo=device \
num_vocab_tiling=4 \
ici_fsdp_parallelism=-1 \
ici_context_parallelism=8 \
dataset_type=synthetic \
opt_type=adam_pax \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=512 \
sa_block_q_dkv=1024 \
sa_block_kv_dkv=4096 \
sa_block_kv_dkv_compute=256 \
sa_q_layout=SEQ_MINOR \
sa_k_layout=HEAD_DIM_MINOR \
sa_v_layout=SEQ_MINOR \
sa_use_fused_bwd_kernel=True \
dq_reduction_steps=3 \
use_tokamax_splash=True \
use_max_logit_estimate=30 \
max_target_length=131072 \
profiler=xplane \
skip_first_n_steps_for_profiler=5 \
profiler_steps=2 \
attention=flash \
quantization=fp8_full \
use_qwix_quantization=True \
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
  --compute-type tpu7x \
  --topology 4x8x8 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log && \
gsutil cp train.log ${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID}.log"
