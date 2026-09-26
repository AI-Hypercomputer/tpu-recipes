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

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---

# Environmental Variables
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE=""
export BASE_OUTPUT_DIR=""
export WORKLOAD_IMAGE=""
# Optional: Hugging Face token. The Wan2.1-T2V-14B-Diffusers weights are public.
export HF_TOKEN="${HF_TOKEN:-}"

# NOTE: `head -c 5` closes the pipe early, which kills `tr` with SIGPIPE. The
# `|| true` keeps that from tripping `set -o pipefail` and aborting the script.
random_suffix=$(tr -dc 'a-z0-9' < /dev/urandom | head -c 5 || true)
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.8s" "${USER//_/-}-wan21")-${random_suffix}-$(date +%Y%m%d-%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"
export BASE_YAML_CONFIG="src/maxdiffusion/configs/base_wan_14b.yml"
export SCRIPT_PATH="src/maxdiffusion/generate_wan.py"

# NOTE: HF_HUB_CACHE points at /dev_shm rather than /dev/shm. Cluster Toolkit
# refuses to mount onto the reserved system path /dev/shm, so the host tmpfs is
# mounted at /dev_shm instead (see the --mount flag on the job submit below).
export COMMAND_PREFIX="bash setup.sh MODE=stable DEVICE=tpu && pip install jax[tpu]==0.9.2 && pip install -e . --no-deps && export HF_HUB_CACHE=/dev_shm && export HF_HUB_ENABLE_HF_TRANSFER=1"

# XLA Flags
XLA_FLAGS=" \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_reduce=true"

# Topology and Parallelism Configuration
# Note: TPU v6e has 1 Tensor Core per physical chip.
# - v6e-16 represents 16 TPU cores (16 physical chips with a 4x4 GKE topology,
#   4 ct6e-standard-4t hosts)
TPU_TOPOLOGY="4x4"

# MaxDiffusion Workload Overrides
MAXDIFFUSION_ARGS="\
model_name=wan2.1 \
attention=flash \
num_inference_steps=50 \
num_frames=81 \
width=1280 \
height=720 \
jax_cache_dir=${BASE_OUTPUT_DIR}/jax_cache/ \
skip_jax_distributed_system=False \
per_device_batch_size=0.25 \
vae_spatial=8 \
ici_data_parallelism=4 \
ici_context_parallelism=4 \
allow_split_physical_axes=True \
guidance_scale=5.0 \
flow_shift=5.0 \
enable_profiler=False \
prompt='a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.' \
flash_min_seq_length=0 \
seed=118445 \
flash_block_sizes='{\"block_kv\":2048,\"block_kv_compute\":1024,\"block_kv_dkv\":2048,\"block_kv_dkv_compute\":1024,\"block_kv_dq\":2048,\"block_q\":3024,\"block_q_dkv\":3024,\"block_q_dq\":3024,\"use_fused_bwd_kernel\":false}' \
base_output_directory=${ARTIFACT_DIR} \
output_dir=${BASE_OUTPUT_DIR}/${WORKLOAD_NAME} \
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
  --compute-type ct6e-standard-4t \
  --topology "${TPU_TOPOLOGY}" \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && \
export ARTIFACT_DIR=${ARTIFACT_DIR} && \
export OUTPUT_DIR=${BASE_OUTPUT_DIR}/${WORKLOAD_NAME} && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
${COMMAND_PREFIX} && export HF_TOKEN=${HF_TOKEN} && \
set +e; \
python ${SCRIPT_PATH} \
  ${BASE_YAML_CONFIG} \
  ${MAXDIFFUSION_ARGS} 2>&1 | tee generate.log; \
GENERATE_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s generate.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled generate.log \${ARTIFACT_DIR}/logs/generate-\${TPU_WORKER_ID:-\${JOBSET_WORKER_INDEX:-\${HOSTNAME:-0}}}.log || true; \
fi; \
exit \${GENERATE_EXIT_CODE}"
