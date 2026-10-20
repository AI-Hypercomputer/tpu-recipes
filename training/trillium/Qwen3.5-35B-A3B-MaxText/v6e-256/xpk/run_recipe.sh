#!/bin/bash

set -e
set -o pipefail

# --- Environment Setup ---
# Activate fallback virtual environment if xpk is not already in PATH.
UV_VENV_PATH="${UV_VENV_PATH:-${HOME}/.local/bin/venv}"
if ! command -v xpk &> /dev/null && [[ -f "${UV_VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${UV_VENV_PATH}/bin/activate"
fi

if ! command -v xpk &> /dev/null; then
  echo "Error: 'xpk' CLI not found. Activate your XPK virtual environment or install it with:"
  echo "  pip install xpk==1.11.0"
  exit 1
fi
# --- End Environment Setup ---

# --- Environment Variables ---
# Set these variables below or export them in your shell before running ./run_recipe.sh.
export PROJECT_ID="${PROJECT_ID:-}"
export CLUSTER_NAME="${CLUSTER_NAME:-}"
export ZONE="${ZONE:-}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-qwen35-35b-v6e256-$(date +%m%d-%H%M)}"

for REQUIRED_VAR in PROJECT_ID CLUSTER_NAME ZONE BASE_OUTPUT_DIR WORKLOAD_IMAGE; do
  if [[ -z "${!REQUIRED_VAR}" ]]; then
    echo "Error: ${REQUIRED_VAR} is empty. Export ${REQUIRED_VAR} in your shell or set it in run_recipe.sh."
    exit 1
  fi
done

# XLA Compiler Flags (TPU v6e)
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=49152 \
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
  --xla_latency_hiding_scheduler_rerun=2 \
  --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000 "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=qwen3.5-35b-a3b \
dataset_type=synthetic \
per_device_batch_size=8 \
gradient_accumulation_steps=1 \
num_moe_token_chunks=1 \
moe_use_direct_token_gather=True \
moe_cache_argsort=True \
router_topk_matmul_vjp=True \
qwen3_5_sublayer_remat=True \
gdn_chunk_impl=fast \
gdn_recurrence_impl=fused \
gdn_pallas_batch_map=False \
gdn_inverse_precision=default \
gdn_pallas_heads_per_block=32 \
moe_pallas_row_gather=True \
moe_pallas_row_gather_sites=route \
moe_pallas_combine=True \
max_target_length=8192 \
ici_fsdp_parallelism=256 \
ici_expert_parallelism=1 \
shard_exp_on_fsdp=True \
num_vocab_tiling=16 \
dtype=bfloat16 \
weight_dtype=bfloat16 \
scan_layers=True \
remat_policy=full \
attention=flash \
megablox=True \
sparse_matmul=True \
use_tokamax_gmm=False \
use_custom_sort_vjp=True \
use_tokamax_splash=False \
wi_tile_fwd_batch_seq=512 \
wi_tile_fwd_embed_dim=2048 \
wi_tile_fwd_mlp_dim=512 \
wi_tile_dlhs_batch_seq=512 \
wi_tile_dlhs_mlp_dim=512 \
wi_tile_dlhs_embed_dim=2048 \
wi_tile_drhs_batch_seq=512 \
wi_tile_drhs_embed_dim=2048 \
wi_tile_drhs_mlp_dim=512 \
wo_tile_fwd_batch_seq=512 \
wo_tile_fwd_mlp_dim=512 \
wo_tile_fwd_embed_dim=2048 \
wo_tile_dlhs_batch_seq=512 \
wo_tile_dlhs_embed_dim=2048 \
wo_tile_dlhs_mlp_dim=512 \
wo_tile_drhs_batch_seq=512 \
wo_tile_drhs_mlp_dim=512 \
wo_tile_drhs_embed_dim=2048 \
sa_block_q=1024 \
sa_block_kv=1024 \
sa_block_kv_compute=512 \
sa_block_q_dkv=1024 \
sa_block_kv_dkv=1024 \
sa_block_kv_dkv_compute=512 \
sa_block_q_dq=1024 \
sa_block_kv_dq=1024 \
sa_use_fused_bwd_kernel=True \
reuse_example_batch=1 \
use_iota_embed=True \
enable_checkpointing=False \
opt_type=adamw \
gcs_metrics=False \
enable_tensorboard=False \
save_config_to_gcs=False \
steps=30 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"

echo "=== Creating XPK Workload: ${WORKLOAD_NAME} ==="
xpk workload create \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --priority=high \
  --max-restarts=0 \
  --device-type=v6e-256 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && set -o pipefail && \
export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export JAX_PLATFORMS='tpu,cpu' && \
export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -u -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml ${MAXTEXT_ARGS}"
