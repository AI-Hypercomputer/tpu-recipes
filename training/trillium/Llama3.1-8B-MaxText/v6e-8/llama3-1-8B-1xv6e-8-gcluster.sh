#!/bin/bash

# Exit on error
set -e

# Read the configuration from environment variables or use defaults.
CLUSTER_NAME=${CLUSTER_NAME:-"v6e-8-brahmai-cluster"}
PROJECT=${PROJECT:-"gke-aishared-gsc-dev"}
LOCATION=${LOCATION:-"southamerica-west1"}
IMAGE_NAME=${IMAGE_NAME:-"southamerica-west1-docker.pkg.dev/gke-aishared-gsc-dev/jetstream-maxtext-ar-kimi-k2/bhramai-base-image:latest"}
OUTPUT_DIR=${OUTPUT_DIR:-"gs://gke-aishared-gsc-dev/maxtext_output"}
RUN_NAME=${RUN_NAME:-"llama3-1-8b-run-$(date +%Y%m%d-%H%M)"}
QUEUE=${QUEUE:-"multislice-queue"}
COMPUTE_TYPE=${COMPUTE_TYPE:-"v6e-4"}
TOPOLOGY=${TOPOLOGY:-"2x4"}
JOB_NAME=${JOB_NAME:-"llama3-1-8b-${TOPOLOGY}-$(date +%H%M)"}


# Read the inline training workload command to execute inside the container
read -r -d '' COMMAND_CONTENT << 'EOF' || true
#!/bin/bash
set -e
echo "Starting MaxText Workload..."

# 1. Set environment variables
export LIBTPU_INIT_ARGS=" --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_assign_all_reduce_scatter_layout=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_aggressive_opt_barrier_removal=ENABLED"
export JAX_PLATFORMS=""
export ENABLE_PJRT_COMPATIBILITY="true"

# 2. Detect MaxText module to run pre-training
TRAIN_CMD=""
if python3 -c "import importlib.util; exit(0 if importlib.util.find_spec('maxtext.trainers.pre_train.train') is not None else 1)" &>/dev/null; then
    TRAIN_CMD="python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml"
elif python3 -c "import importlib.util; exit(0 if importlib.util.find_spec('maxtext.train') is not None else 1)" &>/dev/null; then
    TRAIN_CMD="python3 -m maxtext.train maxtext/configs/base.yml"
elif python3 -c "import importlib.util; exit(0 if importlib.util.find_spec('MaxText.train') is not None else 1)" &>/dev/null; then
    if [ -d "/deps/src/maxtext/configs" ]; then
        TRAIN_CMD="python3 -m MaxText.train /deps/src/maxtext/configs/base.yml"
    else
        TRAIN_CMD="python3 -m MaxText.train MaxText/configs/base.yml"
    fi
else
    echo "Error: Could not find MaxText training module. Please verify MaxText installation inside the container."
    exit 1
fi

# 3. Execute training
$TRAIN_CMD \
    model_name="llama3.1-8b" \
    run_name="${RUN_NAME}" \
    steps=30 \
    base_output_directory="${OUTPUT_DIR}" \
    per_device_batch_size=3 \
    ici_fsdp_parallelism=-1 \
    remat_policy="custom" \
    decoder_layer_input="offload" \
    out_proj="offload" \
    query_proj="offload" \
    key_proj="offload" \
    value_proj="offload" \
    max_target_length=8192 \
    attention="flash" \
    use_iota_embed=True \
    dataset_path="gs://max-datasets-rogue" \
    dataset_type="synthetic" \
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
    profiler="xplane" \
    skip_first_n_steps_for_profiler=10 \
    profiler_steps=5 \
    use_vertex_tensorboard=false
EOF

echo "Submitting MaxText Llama3.1-8B job to cluster $CLUSTER_NAME..."
echo "  PROJECT:     $PROJECT"
echo "  LOCATION:    $LOCATION"
echo "  IMAGE:       $IMAGE_NAME"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  RUN_NAME:    $RUN_NAME"
echo "  QUEUE:       $QUEUE"

# Find the gcluster binary dynamically by checking local and home paths first
GCLUSTER="gcluster"
for candidate in \
  "$(dirname "${BASH_SOURCE[0]}")/gcluster" \
  "$(dirname "${BASH_SOURCE[0]}")/../../../../../cluster-toolkit/gcluster" \
  "./gcluster" \
  "../gcluster" \
  "${HOME}/cluster-toolkit/gcluster" \
  "${HOME}/xpk-devwork/cluster-toolkit/gcluster"; do
  if [ -f "$candidate" ]; then
    GCLUSTER="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
    break
  fi
done

# Submit job using gcluster job submit
$GCLUSTER --cluster "$CLUSTER_NAME" --location "$LOCATION" --project "$PROJECT" job submit \
    --name "$JOB_NAME" \
    --image "$IMAGE_NAME" \
    --command "export RUN_NAME='$RUN_NAME' OUTPUT_DIR='$OUTPUT_DIR' && $COMMAND_CONTENT" \
    --compute-type "$COMPUTE_TYPE" \
    --topology "$TOPOLOGY" \
    --priority medium \
    --queue "$QUEUE"
