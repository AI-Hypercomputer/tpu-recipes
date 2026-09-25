#!/bin/bash

# --- Environment Setup ---
# This script requires the Cluster Toolkit (gcluster) CLI (v1.104.0).
# If you haven't installed gcluster, please refer to the README.md.

export PATH="${HOME}/cluster-toolkit:${PATH}"
CTK_VERSION="1.104.0"
GCLUSTER_BIN="${GCLUSTER_BIN:-gcluster}"
if ! command -v "${GCLUSTER_BIN}" &> /dev/null; then
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

set -euo pipefail

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-}"
export CLUSTER_NAME="${CLUSTER_NAME:-}"
export ZONE="${ZONE:-}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-}"

# Validate required environment variables before submitting
for var in PROJECT_ID CLUSTER_NAME ZONE BASE_OUTPUT_DIR WORKLOAD_IMAGE; do
    if [[ -z "${!var}" ]]; then
        echo "Error: Environment variable $var is required but not set." >&2
        exit 1
    fi
done

if [[ ! "${BASE_OUTPUT_DIR}" =~ ^gs:// ]]; then
    echo "Error: BASE_OUTPUT_DIR must start with 'gs://'" >&2
    exit 1
fi

TIMESTAMP=$(date +%m%d%H%M)
SHORT_USER="${USER:-anon}"
SHORT_USER="${SHORT_USER//_/-}"
SHORT_USER="${SHORT_USER,,}"
SHORT_USER=$(echo "${SHORT_USER}" | tr -cd 'a-z0-9-' | cut -c1-6)
SHORT_USER="${SHORT_USER:-anon}"
DEFAULT_NAME="${SHORT_USER}-gm3-12b-1x256-${TIMESTAMP}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-${DEFAULT_NAME:0:26}}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"

# XLA Flags matching MaxText@tpu-recipes-v0.1.5 (xla_flags_library.CUSTOM_VMEM_LIMIT_FLAG(vmem_limit=122880))
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=122880"

# MaxText Workload Overrides matching gemma3_12b_32768_v6e256 in maxtext_trillium_model_configs.py
MAXTEXT_ARGS=" \
model_name=gemma3-12b \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} \
steps=${STEPS:-30} \
per_device_batch_size=1 \
num_vocab_tiling=16 \
ici_fsdp_parallelism=-1 \
remat_policy=custom \
decoder_layer_input=device \
query_proj=remat \
key_proj=remat \
value_proj=remat \
max_target_length=32768 \
attention=flash \
gcs_metrics=True \
use_iota_embed=True \
dataset_path=gs://max-datasets-rogue \
dataset_type=synthetic \
reuse_example_batch=1 \
enable_checkpointing=False \
profiler=xplane \
skip_first_n_steps_for_profiler=10 \
profiler_steps=2 \
tokenizer_path=assets/tokenizer.gemma3 \
sa_block_q=1024 \
sa_block_kv=1024 \
sa_block_kv_compute=1024 \
sa_block_q_dkv=512 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=512 \
sa_block_q_dq=1024 \
sa_block_kv_dq=1024"

echo "=== Creating Cluster Toolkit Workload: $WORKLOAD_NAME ==="
"${GCLUSTER_BIN}" job submit \
  --skip-prereqs \
  --queue "${QUEUE:-multislice-queue}" \
  --cluster "$CLUSTER_NAME" \
  --project "$PROJECT_ID" \
  --location "$ZONE" \
  --priority medium \
  --restarts 0 \
  --compute-type ct6e-standard-4t \
  --topology 16x16 \
  --num-slices 1 \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace "${NAMESPACE:-default}" \
  --name "${WORKLOAD_NAME}" \
  --command "bash -c 'set -e && set -o pipefail && \
export ENABLE_PATHWAYS_PERSISTENCE=\"1\" && \
export LIBTPU_INIT_ARGS=\"${XLA_FLAGS}\" && \
export ARTIFACT_DIR=\"${ARTIFACT_DIR}\" && \
export JAX_PLATFORMS=\"tpu,cpu\" && \
export ENABLE_PJRT_COMPATIBILITY=\"true\" && \
if [ -d /deps/MaxText ]; then cd /deps/MaxText; elif [ -d /deps ]; then cd /deps; fi && \
export PYTHONPATH=.:./src:\\${PYTHONPATH:-} && \
mkdir -p assets && \
if [ -f assets/tokenizers/tokenizer.gemma3 ] && [ ! -f assets/tokenizer.gemma3 ]; then \
  cp assets/tokenizers/tokenizer.gemma3 assets/tokenizer.gemma3 || true; \
fi && \
if [ ! -f assets/tokenizer.gemma3 ] && [ -f /deps/src/maxtext/assets/tokenizers/tokenizer.gemma3 ]; then \
  cp /deps/src/maxtext/assets/tokenizers/tokenizer.gemma3 assets/tokenizer.gemma3 || true; \
fi && \
set +e; \
if [ -f MaxText/train.py ]; then \
  python3 -u MaxText/train.py MaxText/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee train.log; \
else \
  python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee train.log; \
fi; \
TRAIN_EXIT_CODE=\\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  if command -v gcloud &> /dev/null; then \
    timeout 30s gcloud storage cp --no-user-output-enabled train.log \\${ARTIFACT_DIR}/logs/train-\\${TPU_WORKER_ID:-\\${JOBSET_WORKER_INDEX:-\\${HOSTNAME:-0}}}.log || true; \
  elif command -v gsutil &> /dev/null; then \
    timeout 30s gsutil cp train.log \\${ARTIFACT_DIR}/logs/train-\\${TPU_WORKER_ID:-\\${JOBSET_WORKER_INDEX:-\\${HOSTNAME:-0}}}.log || true; \
  fi; \
fi; \
exit \\${TRAIN_EXIT_CODE}'"
