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
    echo "pip install xpk==1.4.0"
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
export ARTIFACT_DIR=""
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-gemma4-31b")-$(date +%Y%m%d-%H%M)"

# XLA Flags
XLA_FLAGS=" \
   "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=gemma4-31b \
per_device_batch_size=8 \
max_target_length=8192 \
async_checkpointing=False \
enable_checkpointing=False \
dataset_type=synthetic \
steps=30 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"



xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --device-type=tpu7x-4x4x4 \
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
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log && \
gsutil cp train.log ${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID}.log"