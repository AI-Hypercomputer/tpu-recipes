#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

set -e
set -o pipefail

# --- Configuration ---
# Before running this script, export the environment variables below in your
# shell (see README.md), or edit the defaults here.
# ---

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-}"
export CLUSTER_NAME="${CLUSTER_NAME:-}"
export ZONE="${ZONE:-}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1}"
# Number of v6e-256 slices (1, 2 or 4). Selects configs/${NUM_SLICES}x_v6e_256.yaml.
export NUM_SLICES="${NUM_SLICES:-1}"
export BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/${NUM_SLICES}x_v6e_256.yaml}"
# Optional: GCS path of a custom benchmark config (see README.md).
export GCS_CONFIG_URI="${GCS_CONFIG_URI:-}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(printf "%.11s" "${USER//_/-}")-coll-${NUM_SLICES}x-$(date +%H%M)}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"

LIBTPU_FLAGS="--megascale_grpc_premap_memory_bytes=17179869184 --xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction=true"

FETCH_CONFIG=""
if [ -n "${GCS_CONFIG_URI}" ]; then
  BENCHMARK_CONFIG="configs/custom_config.yaml"
  FETCH_CONFIG="gcloud storage cp ${GCS_CONFIG_URI} ${BENCHMARK_CONFIG} && "
fi

echo "=== Creating Cluster Toolkit Workload: $WORKLOAD_NAME ==="
"${GCLUSTER_BIN}" job submit \
  --skip-prereqs \
  --queue multislice-queue \
  --cluster "$CLUSTER_NAME" \
  --project "$PROJECT_ID" \
  --location "$ZONE" \
  --priority medium \
  --restarts 0 \
  --compute-type ct6e-standard-4t \
  --topology 16x16 \
  --num-slices "${NUM_SLICES}" \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace default \
  --name "${WORKLOAD_NAME}" \
  --command "set -e && set -o pipefail && \
export LIBTPU_INIT_ARGS='${LIBTPU_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git && \
cd accelerator-microbenchmarks && \
git checkout trillium-collectives && \
pip install -r requirements.txt && \
echo \"net.ipv4.tcp_rmem: \$(cat /proc/sys/net/ipv4/tcp_rmem)\" && \
${FETCH_CONFIG}set +e; \
python3 -u src/run_benchmark.py --config=${BENCHMARK_CONFIG} | tee benchmark.log; \
BENCHMARK_EXIT_CODE=\${PIPESTATUS[0]}; \
WORKER_ID=\${MEGASCALE_SLICE_ID:-0}-\${TPU_WORKER_ID:-\${HOSTNAME:-0}}; \
if [ -s benchmark.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled benchmark.log \${ARTIFACT_DIR}/logs/benchmark-\${WORKER_ID}.log || true; \
fi; \
if [ -d /tmp/microbenchmarks/collectives ]; then \
  timeout 60s gcloud storage cp --recursive --no-user-output-enabled /tmp/microbenchmarks/collectives \${ARTIFACT_DIR}/results/worker-\${WORKER_ID}/ || true; \
fi; \
exit \${BENCHMARK_EXIT_CODE}"
