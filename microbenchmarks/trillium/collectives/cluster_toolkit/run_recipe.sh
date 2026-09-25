#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Environment Setup ---
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

export PROJECT_ID="${PROJECT_ID:-${PROJECT:-}}"
export CLUSTER_NAME="${CLUSTER_NAME:-}"
export ZONE="${ZONE:-}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1}"
export NUM_SLICES="${NUM_SLICES:-2}"
export BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/${NUM_SLICES}x_v6e_256.yaml}"
export GCS_CONFIG_URI="${GCS_CONFIG_URI:-}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"

for var in PROJECT_ID CLUSTER_NAME ZONE WORKLOAD_IMAGE; do
    if [[ -z "${!var}" ]]; then
        echo "Error: Environment variable $var is required but not set." >&2
        exit 1
    fi
done

TIMESTAMP=$(date +%m%d%H%M)
SHORT_USER="${USER:-anon}"
SHORT_USER="${SHORT_USER//_/-}"
SHORT_USER="${SHORT_USER,,}"
SHORT_USER=$(echo "${SHORT_USER}" | tr -cd 'a-z0-9-' | cut -c1-6)
SHORT_USER="${SHORT_USER:-anon}"
DEFAULT_NAME="${SHORT_USER}-coll-${NUM_SLICES}x256-${TIMESTAMP}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-${DEFAULT_NAME:0:26}}"

LIBTPU_FLAGS="--megascale_grpc_premap_memory_bytes=17179869184 --xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction=true"

echo "=== Creating Cluster Toolkit Workload: $WORKLOAD_NAME ==="
"${GCLUSTER_BIN}" job submit \
  --skip-prereqs \
  --queue "${QUEUE:-multislice-queue}" \
  --project "${PROJECT_ID}" \
  --cluster "${CLUSTER_NAME}" \
  --location "${ZONE}" \
  --priority medium \
  --restarts 0 \
  --compute-type ct6e-standard-4t \
  --topology 16x16 \
  --num-slices "${NUM_SLICES}" \
  --image "${WORKLOAD_IMAGE}" \
  --verbose \
  --gke-namespace "${NAMESPACE:-default}" \
  --name "${WORKLOAD_NAME}" \
  --command "bash -c 'set -e && set -o pipefail && \
git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git && \
cd accelerator-microbenchmarks && \
git checkout trillium-collectives && \
pip install -r requirements.txt && \
echo "4096 41943040 314572800" > /proc/sys/net/ipv4/tcp_rmem && \
export LIBTPU_INIT_ARGS="${LIBTPU_FLAGS}" && \
if [[ -n "${GCS_CONFIG_URI}" ]]; then \
  gcloud storage cp "${GCS_CONFIG_URI}" configs/custom_config.yaml && \
  python src/run_benchmark.py --config=configs/custom_config.yaml; \
else \
  python src/run_benchmark.py --config="${BENCHMARK_CONFIG}"; \
fi && \
if [[ -n "${BASE_OUTPUT_DIR}" && -d /tmp/microbenchmarks/collectives ]]; then \
  gcloud storage cp --recursive /tmp/microbenchmarks/collectives "${BASE_OUTPUT_DIR%/}/${WORKLOAD_NAME}/" || true; \
fi'"
