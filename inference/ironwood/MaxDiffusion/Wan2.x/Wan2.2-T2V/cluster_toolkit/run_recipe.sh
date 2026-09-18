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

# Environmental Variables
export PROJECT_ID="${PROJECT_ID}"
export CLUSTER_NAME="${CLUSTER_NAME}"
export ZONE="${ZONE}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR}"
export HF_TOKEN="${HF_TOKEN}"

export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-<YOUR_CONTAINER_REGISTRY>/<YOUR_PROJECT_ID>/<YOUR_IMAGE_NAME>:latest}"
random_suffix=$(tr -dc 'a-z0-9' < /dev/urandom | head -c 5 || true)
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.20s" "${USER//_/-}-wan2-2-t2v")-${random_suffix}-$(date +%Y%m%d-%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"
export BASE_YAML_CONFIG="src/maxdiffusion/configs/base_wan_27b.yml"
export SCRIPT_PATH="src/maxdiffusion/generate_wan.py"

# Default COMMAND_PREFIX tailored for Ironwood (7x)
# NOTE: HF_HUB_CACHE points at /dev_shm rather than /dev/shm. Cluster Toolkit
# refuses to mount onto the reserved system path /dev/shm, so the host tmpfs is
# mounted at /dev_shm instead (see the --mount flag on the job submit below).
export COMMAND_PREFIX="bash setup.sh MODE=stable DEVICE=tpu && pip install jax[tpu]==0.10.0 && pip install -e . --no-deps && export HF_HUB_CACHE=/dev_shm && export HF_HUB_ENABLE_HF_TRANSFER=1"

# XLA Flags optimized for Ironwood
XLA_FLAGS=" \
--xla_tpu_dvfs_p_state=7 \
--xla_tpu_spmd_rng_bit_generator_unsafe=true \
--xla_tpu_enable_dot_strength_reduction=true \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_enable_async_collective_permute=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true \
--xla_tpu_scoped_vmem_limit_kib=81920 \
--xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_tpu_enable_ici_ag_pipelining=true \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2 \
--xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_tpu_memory_bound_loop_optimizer_options=enabled:true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_enable_concurrent_sparse_core_offloading=true \
--xla_tpu_assign_all_reduce_scatter_layout=true"

# MaxDiffusion Workload Overrides
MAXDIFFUSION_ARGS="\
model_name=wan2.2 \
attention=ulysses_custom \
num_inference_steps=40 \
num_frames=81 \
width=1280 \
height=720 \
per_device_batch_size=0.125 \
vae_spatial=8 \
ici_data_parallelism=2 \
ici_context_parallelism=4 \
fps=16 \
use_kv_cache=True \
use_base2_exp=True \
use_experimental_scheduler=True \
use_batched_text_encoder=True \
flash_block_sizes='{\"block_kv\":1024,\"block_kv_compute\":1024,\"block_kv_compute_in\":1024,\"block_kv_dkv\":2048,\"block_kv_dkv_compute\":2048,\"block_kv_dq\":2048,\"block_q\":4864,\"block_q_dkv\":3024,\"block_q_dq\":3024,\"heads_per_tile\":1}' \
max_train_steps=30 \
base_output_directory=${BASE_OUTPUT_DIR}/${WORKLOAD_NAME} \
output_dir=${BASE_OUTPUT_DIR}/ \
run_name=${WORKLOAD_NAME}"

echo "=== Creating Cluster Toolkit Workload: $WORKLOAD_NAME ==="
"${GCLUSTER_BIN}" job submit \
  --skip-prereqs \
  --queue multislice-queue \
  --mount "/dev/shm;/dev_shm;rw" \
  --cluster "$CLUSTER_NAME" \
  --project "$PROJECT_ID" \
  --location "$ZONE" \
  --priority medium \
  --restarts 0 \
  --compute-type tpu7x \
  --topology 2x2x1 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && \
export ARTIFACT_DIR=\${ARTIFACT_DIR} && \
export OUTPUT_DIR=\${BASE_OUTPUT_DIR}/ && \
export LIBTPU_INIT_ARGS='\${XLA_FLAGS}' && \
\${COMMAND_PREFIX} && export HF_TOKEN=\${HF_TOKEN} && \
set +e; \
python \${SCRIPT_PATH} \
  \${BASE_YAML_CONFIG} \
  \${MAXDIFFUSION_ARGS} | tee generate.log; \
GENERATE_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s generate.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled generate.log \${ARTIFACT_DIR}/logs/generate-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${GENERATE_EXIT_CODE}"
