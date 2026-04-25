#!/bin/bash

# --- Environment Setup ---
# This script requires uv and a Python 3.11 virtual environment with xpk installed.
# If you haven't set up uv and the environment, please refer to the README.md.

UV_VENV_PATH="${HOME}/.local/bin/venv"
UV_PYTHON_VERSION="3.11"

# Activate the virtual environment
source "${UV_VENV_PATH}/bin/activate"

# Check if xpk is installed in the venv
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the virtual environment. Please install it by running:"
    echo "pip install xpk==1.8.0"
    exit 1
fi
# --- End Environment Setup ---

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---

# --- Environment Variables ---
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE=""
export BASE_OUTPUT_DIR=""
export ARTIFACT_DIR="${BASE_OUTPUT_DIR}"
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-deepseekv3-671b-4096-fsdp-fp8")-$(date +%Y%m%d-%H%M)"

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
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_aggressive_opt_barrier_removal=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_accumulate_into_mrb=true \
  --xla_tpu_mosaic_fusion=false \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
  --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=204800 \
  --xla_max_concurrent_async_all_gathers=1 \
  --xla_tpu_scheduler_percent_shared_memory_limit=140 \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_enable_collective_pipeliner=true \
  --xla_tpu_enable_ici_rs_pipelining=false \
  --xla_tpu_enable_tree_use_collective_pipeliner=true \
  --xla_latency_hiding_scheduler_rerun=0 \
  --xla_tpu_host_transfer_overlap_limit=1 \
  --xla_tpu_ici_rs_pipelining_threshold_kib=1048576 \
  --xla_tpu_impure_use_lmr_on_gxc=true \
  --xla_tpu_dot_dot_fusion_duplicated=true \
  --xla_tpu_rwb_fusion=false \
  --xla_tpu_order_dot_after_layout=true \
  --xla_tpu_prefetch_interval_picker_size_override=0 \
  --xla_tpu_async_copy_bandwidth_scaling_factor=0.5 "

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
megablox=True \
sparse_matmul=True \
use_custom_sort_vjp=True \
shard_exp_on_fsdp=True \
moe_fsdp_use_two_stage_all_gather=True \
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
use_max_logit_estimate=22 \
attn_logits_soft_cap=15 \
cost_estimate_flops_fwd=5000000000000 \
cost_estimate_flops_bwd=5000000000000 \
float32_weight_sum=False \
use_tokamax_gmm=True \
tokenizer_path=assets/tokenizer.mistral-v3 \
dataset_type=synthetic \
use_qwix_quantization=True \
quantization=fp8_full \
wi_tile_fwd_batch_seq=256 \
wi_tile_fwd_embed_dim=7168 \
wi_tile_fwd_mlp_dim=2048 \
wi_tile_dlhs_batch_seq=256 \
wi_tile_dlhs_embed_dim=2048 \
wi_tile_dlhs_mlp_dim=7168 \
wi_tile_drhs_batch_seq=512 \
wi_tile_drhs_embed_dim=1024 \
wi_tile_drhs_mlp_dim=2048 \
wo_tile_fwd_batch_seq=256 \
wo_tile_fwd_embed_dim=2048 \
wo_tile_fwd_mlp_dim=7168 \
wo_tile_dlhs_batch_seq=256 \
wo_tile_dlhs_embed_dim=7168 \
wo_tile_dlhs_mlp_dim=2048 \
wo_tile_drhs_batch_seq=512 \
wo_tile_drhs_embed_dim=512 \
wo_tile_drhs_mlp_dim=7168 \
wi_tile_fwd_buffer_count=2 \
wi_tile_dlhs_buffer_count=2 \
wi_tile_drhs_buffer_count=3 \
wo_tile_fwd_buffer_count=2 \
wo_tile_dlhs_buffer_count=2 \
wo_tile_drhs_buffer_count=3 \
weight_quantization_calibration_method='fixed,-224,224' \
act_quantization_calibration_method='fixed,-224,224' \
enable_checkpointing=False \
steps=30 \
base_output_directory=<your-gcs-bucket-path> \
run_name=${WORKLOAD_NAME} \
output_dir=${BASE_OUTPUT_DIR}"



xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --device-type=tpu7x-4x8x8 \
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
pip install git+https://github.com/openxla/tokamax.git@4936e75cb40bac9a746f0f10c4bb6887f4c217d8 --no-deps &&  \
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log && \
gsutil cp train.log ${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}/logs/train-\${TPU_WORKER_ID}.log"
