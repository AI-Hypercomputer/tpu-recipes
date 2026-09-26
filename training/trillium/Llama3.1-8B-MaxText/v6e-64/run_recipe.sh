#!/bin/bash

# --- Environment Setup ---
# This script requires uv and a Python 3.13 virtual environment with xpk installed.
# If you haven't set up uv and the environment, please refer to the README.md.

UV_VENV_PATH="${HOME}/.local/bin/venv"
UV_PYTHON_VERSION="3.13"

# Activate the virtual environment
source "${UV_VENV_PATH}/bin/activate"

# Check if xpk is installed in the venv
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the virtual environment. Please install it by running:"
    echo "pip install xpk==1.16.0"
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
export ARTIFACT_DIR=""
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-llama3-1-8b-8192-no-collective-matmul")-$(date +%Y%m%d-%H%M)"


# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=98304 \
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true \
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
  --xla_tpu_assign_all_reduce_scatter_layout=true \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_all_reduce_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=true \
  --xla_sc_enable_instruction_fusion=false \
  --xla_sc_disjoint_spmem=false \
  --xla_sc_disable_megacore_partitioning=true \
  --deepsea_chip_config_name=megachip_tccontrol \
  --xla_tpu_enable_all_experimental_scheduler_features=true \
  --xla_tpu_enable_scheduler_memory_pressure_tracking=true \
  --xla_tpu_host_transfer_overlap_limit=32 \
  --xla_tpu_aggressive_opt_barrier_removal=ENABLED \
  --xla_lhs_prioritize_async_depth_over_stall=ENABLED \
  --xla_tpu_enable_ag_backward_pipelining=true \
  --xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
  --xla_should_add_loop_invariant_op_in_chain=ENABLED \
  --xla_max_concurrent_host_send_recv=128 \
  --xla_tpu_scheduler_percent_shared_memory_limit=100 \
  --xla_latency_hiding_scheduler_rerun=2 \
  --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000 "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=llama3.1-8b \
dataset_type=synthetic \
dataset_path=gs://max-datasets-rogue \
ici_fsdp_parallelism=-1 \
remat_policy=custom \
decoder_layer_input=offload \
query_proj=offload \
key_proj=offload \
value_proj=offload \
max_target_length=8192 \
attention=flash \
use_iota_embed=True \
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
per_device_batch_size=5 \
out_proj=device \
steps=30 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"



echo "=== Creating XPK Workload: $WORKLOAD_NAME ==="
xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=high \
  --max-restarts=0 \
  --device-type=v6e-64 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
   \
   \
  --workload="${WORKLOAD_NAME}" \
   \
  --command="set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
 \
 \
set +e; \
python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-0}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"