#!/bin/bash

# Configuration variables
export PROJECT_ID="<your-project-id>"
export CLUSTER_NAME="<your-cluster-name>"
export ZONE="<your-zone>"
export WORKLOAD_NAME="${USER}-muse-glimmer-30b-$(date +%H%M)"
export CONTAINER_REGISTRY="gcr.io"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${USER}-torchtitan-runner"
export BASE_OUTPUT_DIR="gs://<your-gcs-bucket-name>"
export HF_TOKEN="<your-huggingface-token>"

export PATH="${HOME}/cluster-toolkit:${PATH}"
GCLUSTER_BIN="${GCLUSTER_BIN:-gcluster}"
DRY_RUN_ARG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN_ARG=("--dry-run")
      shift
      ;;
    --dry-run-out=*)
      DRY_RUN_ARG=("--dry-run-out" "${1#--dry-run-out=}")
      shift
      ;;
    --dry-run-out)
      DRY_RUN_ARG=("--dry-run-out" "$2")
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Note: Each TPU v7x chip has 2 TensorCores (logical devices), so a 4-chip node runs 8 processes (--nproc_per_node=8).
"${GCLUSTER_BIN}" job submit \
  --skip-prereqs \
  --mount "${BASE_OUTPUT_DIR};/job-logs;rw" \
    --name "${WORKLOAD_NAME}" \
    --cluster "${CLUSTER_NAME}" \
    --project "${PROJECT_ID}" \
    --location "${ZONE}" \
    --priority low \
    --restarts 0 \
    --compute-type tpu7x \
    --topology 4x4x4 \
    --node-constraint cloud.google.com/placement-policy-name=tpu7x-128-4x4x4-placement-policy \
    --num-slices 1 \
    --image "${WORKLOAD_IMAGE}" \
    --gke-disable-parallel-containers \
    --env=WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME} \
    --env=JAX_PLATFORMS= \
    --env='LIBTPU_INIT_ARGS=--xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true --xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true' \
    --env=HF_TOKEN="${HF_TOKEN}" \
    --env="TORCHRUN_WORKLOAD_OVERRIDE=--training.steps=30 --dump-folder=/job-logs/${WORKLOAD_NAME} --metrics.save-tb-folder=tb --metrics.enable-tensorboard --debug.save-config-file=torchtitan_config.json --module=torchtitan.experiments.tpu.muse_glimmer --config=muse_glimmer_30b_full_logits --parallelism.enable-tpu-topo-mesh" \
    "${DRY_RUN_ARG[@]}" \
    --command ' torchrun --nnodes=16 --nproc_per_node=8 --rdzv_backend=static --rdzv_endpoint=${WORKERS_0_HOSTNAME}:29501 --node_rank=${TPU_WORKER_ID}  -m torchtitan.experiments.tpu.train ${TORCHRUN_WORKLOAD_OVERRIDE}' 

