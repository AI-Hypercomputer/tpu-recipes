#!/bin/bash

# --- Environment Setup ---
# This script requires uv and a Python 3.12 virtual environment with xpk installed.
# If you haven't set up uv and the environment, please refer to the README.md.

UV_VENV_PATH="${HOME}/.local/bin/venv"
UV_PYTHON_VERSION="3.12"

# Activate the virtual environment
source "${UV_VENV_PATH}/bin/activate"

# Check if xpk is installed in the venv
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the virtual environment. Please install it by running:"
    echo "pip install xpk==0.16.1"
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
export BASE_OUTPUT_DIR="" # for example, gs://<your_gcs_bucket>
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="$(printf "%.24s" "${USER//_/-}-wan22")-$(date +%Y%m%d-%H%M)"
# DATASET_DIR is where preprocessed TFRecord training data was uploaded.
export DATASET_DIR=${BASE_OUTPUT_DIR}/wan2.2-t2v-tfrecords

# Optimized XLA Flags for TPU v7x (Ironwood)
XLA_FLAGS=" \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_use_tc_device_shape_on_sc=true \
  --xla_sc_disable_megacore_partitioning=true \
  --xla_enable_async_all_gather=true \
  --xla_max_concurrent_async_all_gathers=4 \
  --xla_tpu_enable_async_collective_fusion=false \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_reduce=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_enable_async_all_to_all=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_latency_hiding_scheduler_rerun=5 \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=105 \
  --xla_tpu_rwb_fusion=false \
  --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false \
  --xla_tpu_impure_enable_packed_bf16_math_ops=false \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_enable_transpose_trace=false "

# MaxDiffusion Workload Overrides (40.18% MFU on tpu7x-2x4x4)
MAXDIFFUSION_ARGS="\
model_name=wan2.2 \
attention=ulysses_ring_custom \
ulysses_shards=4 \
ulysses_attention_chunks=2 \
mask_padding_tokens=False \
use_base2_exp=True \
weights_dtype=bfloat16 \
activations_dtype=bfloat16 \
guidance_scale_low=3.0 \
guidance_scale_high=4.0 \
boundary_ratio=0.875 \
flow_shift=12.0 \
fps=16 \
skip_jax_distributed_system=False \
output_dir=${BASE_OUTPUT_DIR} \
train_data_dir=${DATASET_DIR}/train \
eval_data_dir=${DATASET_DIR}/eval \
load_tfrecord_cached=True \
height=1280 \
width=720 \
num_frames=81 \
num_inference_steps=40 \
prompt='a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.' \
jax_cache_dir=${BASE_OUTPUT_DIR}/jax_cache/ \
opt_enable_grad_global_norm_clipping=False \
max_train_steps=150 \
enable_profiler=True \
skip_first_n_steps_for_profiler=5 \
profiler_steps=5 \
dataset_save_location=${DATASET_DIR} \
remat_policy=CUSTOM \
names_which_can_be_saved='[\"attn_output\",\"self_attn\",\"cross_attn\",\"hidden_states\"]' \
names_which_can_be_offloaded='[]' \
flash_block_sizes='{\"block_q\": 4096, \"block_kv\": 2048, \"block_kv_compute\": 1024, \"block_kv_compute_in\": 256, \"block_q_dkv\": 2048, \"block_kv_dkv\": 2048, \"block_kv_dkv_compute\": 1024, \"block_kv_dkv_compute_in\": 256, \"dq_reduction_steps\": 3, \"use_fused_bwd_kernel\": true}' \
flash_min_seq_length=0 \
seed=123456789 \
per_device_batch_size=0.25 \
ici_data_parallelism=1 \
ici_fsdp_parallelism=16 \
ici_context_parallelism=4 \
ici_tensor_parallelism=1 \
vae_spatial=1 \
replicate_vae=True \
allow_split_physical_axes=True \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"

xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=3 \
  --device-type=tpu7x-2x4x4 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && \
export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export JAX_PLATFORMS='tpu,cpu' && \
export ENABLE_PJRT_COMPATIBILITY='true' && \
pip install . && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
echo 'Starting Wan 2.2 T2V 14B training on TPU v7x ...' && \
HF_HUB_CACHE=/dev/shm python3 -m src.maxdiffusion.train_wan \
  src/maxdiffusion/configs/base_wan_27b.yml \
  output_dir=${BASE_OUTPUT_DIR} \
  train_data_dir=${DATASET_DIR}/train \
  jax_cache_dir=${BASE_OUTPUT_DIR}/jax_cache/ \
  dataset_save_location=${DATASET_DIR} \
  base_output_directory=${BASE_OUTPUT_DIR} \
  run_name=${WORKLOAD_NAME} \
  ${MAXDIFFUSION_ARGS}"
