#!/bin/bash
set -e

# --- Environment Setup ---
# This script requires uv and a Python virtual environment with xpk installed.
# If you haven't set up uv and the environment, please refer to the README.md.

# Activate the virtual environment
export UV_VENV_PATH="${UV_VENV_PATH:-${HOME}/.local/bin/venv}"
if [ -f "${UV_VENV_PATH}/bin/activate" ]; then
    source "${UV_VENV_PATH}/bin/activate"
else
    echo "Error: Virtual environment not found at ${UV_VENV_PATH}. Check README.md."
    exit 1
fi

# Check if xpk is installed in the venv
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the virtual environment. Please install it by running:"
    echo "pip install xpk==1.3.0"
    exit 1
fi

# --- End Environment Setup ---

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
export TPU_TYPE="${TPU_TYPE:-v6e-8}"
export RESOLUTION="${RESOLUTION:-720p}"

export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-<YOUR_CONTAINER_REGISTRY>/<YOUR_PROJECT_ID>/<YOUR_IMAGE_NAME>:latest}"
random_suffix=$(tr -dc 'a-z0-9' < /dev/urandom | head -c 5)
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.20s" "${USER//_/-}-wan2-2-t2v")-${random_suffix}-$(date +%Y%m%d-%H%M)}"
export BASE_YAML_CONFIG="src/maxdiffusion/configs/base_wan_27b.yml"
export SCRIPT_PATH="src/maxdiffusion/generate_wan.py"

# Default COMMAND_PREFIX tailored for Trillium (v6e)
export COMMAND_PREFIX="bash setup.sh MODE=stable DEVICE=tpu && pip install jax[tpu]==0.10.0 && pip install -e . --no-deps && export HF_HUB_CACHE=/dev/shm && export HF_HUB_ENABLE_HF_TRANSFER=1 && export TORCHINDUCTOR_FREEZING=1 && export TORCHINDUCTOR_CPP_WRAPPER=1 && export TORCHINDUCTOR_MEMORY_PLANNING=1 && export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 && export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 && export JAX_DEFAULT_MATMUL_PRECISION=bfloat16"

# XLA Flags optimized for Trillium (v6e)
XLA_FLAGS="'\"'\"' \
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
--xla_tpu_scoped_vmem_limit_kib=32768 \
--xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
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
--xla_tpu_assign_all_reduce_scatter_layout=true'\"'\"'"

# Resolution Configuration
case "$RESOLUTION" in
    "720p")
        WIDTH=1280
        HEIGHT=720
        ;;
    "480p")
        WIDTH=832
        HEIGHT=480
        ;;
    *)
        echo "Error: Unsupported resolution '$RESOLUTION'. Supported resolutions: 720p, 480p"
        exit 1
        ;;
esac

# Topology and Parallelism Configuration
# Note: TPU v6e has 1 Tensor Core per physical chip.
# - v6e-8 represents 8 TPU cores (8 physical chips with a v6e-8 GKE topology)
# - v6e-16 represents 16 TPU cores (16 physical chips with a v6e-16 GKE topology)
case "$TPU_TYPE" in
    "v6e-8")
        XPK_TPU_TYPE="v6e-8"
        ICI_DATA_PARALLELISM=2
        ICI_CONTEXT_PARALLELISM=4
        PER_DEVICE_BATCH_SIZE=0.125
        ;;
    "v6e-16")
        XPK_TPU_TYPE="v6e-16"
        ICI_DATA_PARALLELISM=2
        ICI_CONTEXT_PARALLELISM=8
        PER_DEVICE_BATCH_SIZE=0.0625
        ;;
    *)
        echo "Error: Unsupported TPU_TYPE '$TPU_TYPE'. Supported values: v6e-8, v6e-16"
        exit 1
        ;;
esac

# MaxDiffusion Workload Overrides
MAXDIFFUSION_ARGS="\
model_name='\"'\"'wan2.2'\"'\"' \
attention='\"'\"'ulysses_custom'\"'\"' \
num_inference_steps=40 \
seed=12345 \
num_frames=81 \
width=${WIDTH} \
height=${HEIGHT} \
per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
vae_spatial=4 \
vae_decode_chunk=4 \
vae_weights_dtype='bfloat16' \
vae_dtype='bfloat16' \
text_encoder_dtype='bfloat16' \
compile_text_encoder=true \
ici_data_parallelism=${ICI_DATA_PARALLELISM} \
ici_context_parallelism=${ICI_CONTEXT_PARALLELISM} \
fps=16 \
use_kv_cache=True \
use_base2_exp=True \
use_experimental_scheduler=True \
use_batched_text_encoder=true \
flash_block_sizes='\"'\"'{\"block_q\":3328,\"block_kv_compute\":256,\"block_kv\":2816,\"block_kv_compute_in\":256,\"block_q_dkv\":3328,\"block_kv_dkv\":2816,\"block_kv_dkv_compute\":256,\"block_q_dq\":3328,\"block_kv_dq\":2816,\"heads_per_tile\":1}'\"'\"' \
base_output_directory='\"'\"'${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}'\"'\"' \
output_dir='\"'\"'${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}'\"'\"' \
run_name='\"'\"'${WORKLOAD_NAME}'\"'\"'"

echo "Deploying workload via xpk..."

cmd="xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --tpu-type=$XPK_TPU_TYPE \
  --num-slices=1 \
  --docker-image=\"${WORKLOAD_IMAGE}\" \
  --enable-debug-logs \
  --workload=\"${WORKLOAD_NAME}\" \
  --command='set -e && \
export ARTIFACT_DIR=${BASE_OUTPUT_DIR}/${WORKLOAD_NAME} && \
export OUTPUT_DIR=${BASE_OUTPUT_DIR}/${WORKLOAD_NAME} && \
export LIBTPU_INIT_ARGS=${XLA_FLAGS} && \
${COMMAND_PREFIX} && export HF_TOKEN=${HF_TOKEN} && \
  python ${SCRIPT_PATH}  \
  ${BASE_YAML_CONFIG} \
  ${MAXDIFFUSION_ARGS}'"

eval ${cmd}
