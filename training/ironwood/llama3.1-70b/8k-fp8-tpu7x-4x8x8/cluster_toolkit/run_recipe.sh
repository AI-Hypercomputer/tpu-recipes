#!/bin/bash
# Launches the benchmark workload with Cluster Toolkit.
#
# Reads the environment variables described in README.md, which also
# covers provisioning a cluster that can accept this workload. Export them
# beforehand or edit the defaults below, then run this script.
set -euo pipefail

export PROJECT_ID="${PROJECT_ID:-<your-project-id>}"
export CLUSTER_NAME="${CLUSTER_NAME:-<your-cluster-name>}"
export ZONE="${ZONE:-<your-zone>}"
export REGION="${REGION:-${ZONE%-*}}"
export NAMESPACE="${NAMESPACE:-default}"
# Kueue LocalQueue to submit to. Leave empty to let gcluster auto-discover the
# cluster's LocalQueue (Cluster Toolkit blueprints install `user-queue`).
export KUEUE_QUEUE="${KUEUE_QUEUE:-}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(echo "${USER:-runner}" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr -cd 'a-z0-9-' | sed 's/^-*//' | cut -c1-10)-benchmark-$(date +%H%M)}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-<your-container-image>}"
export BUCKET_NAME="${BUCKET_NAME:-<your-gcs-bucket>}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-gs://${BUCKET_NAME}}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}}"

for var in PROJECT_ID CLUSTER_NAME REGION WORKLOAD_IMAGE BASE_OUTPUT_DIR ARTIFACT_DIR; do
  if [[ -z "${!var:-}" || "${!var}" == *"<your"* ]]; then
    echo "Error: ${var} is unset or still set to a placeholder ('${!var:-}')." >&2
    exit 1
  fi
done

if [[ ! "${ARTIFACT_DIR}" =~ ^gs:// ]]; then
  echo "Error: ARTIFACT_DIR must be a valid gs:// URI (got: '${ARTIFACT_DIR}')." >&2
  exit 1
fi

gcluster job submit --name "${WORKLOAD_NAME}" --cluster "${CLUSTER_NAME}" --project "${PROJECT_ID}" --location "${REGION}" --compute-type tpu7x --num-slices 1 --topology 4x8x8 --image "${WORKLOAD_IMAGE}" --env ARTIFACT_DIR="${ARTIFACT_DIR}" --env JOB_IDENTIFIER="${WORKLOAD_NAME}" --env JAX_COMPILATION_CACHE_DIR="${BASE_OUTPUT_DIR}"/JAX_XLA_CACHE/maxtext/llama3-1-70b --gke-namespace "${NAMESPACE}" --priority medium --restarts 0 --gke-disable-parallel-containers --command '#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Graceful Shutdown Logic ---
_sigterm() {
  echo "Caught SIGTERM signal, sending to child process..."
  kill -SIGTERM "$PID" 2>/dev/null
}
trap _sigterm SIGTERM

MAXTEXT_CONFIG_PATH="maxtext/configs/base.yml"
# The config'"'"'s own XLA flags. The workload'"'"'s arrive in XLA_FLAGS.
CONFIG_XLA_FLAGS='"'"'--xla_tpu_scoped_vmem_limit_kib=61440 --xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_enable_offloading_copy_to_sparsecore=false --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=false'"'"'

# Fallback check to support different image structures
if [ ! -f "${MAXTEXT_CONFIG_PATH}" ] && [ -f "src/${MAXTEXT_CONFIG_PATH}" ]; then
  echo "MAXTEXT_CONFIG_PATH not found at ${MAXTEXT_CONFIG_PATH}, falling back to src/${MAXTEXT_CONFIG_PATH}"
  MAXTEXT_CONFIG_PATH="src/${MAXTEXT_CONFIG_PATH}"
fi

# --- TPU Worker Hostnames & ID ---
if [ ! -z "${NNODES}" ] && [ ! -z "${JOBSET_NAME}" ] && [ ! -z "${SUBDOMAIN}" ]; then
  HOSTS=""
  for i in $(seq 0 $((NNODES - 1))); do
    # JobSet pod naming: ${JOBSET_NAME}-workload-0-${i}.${SUBDOMAIN}
    HOSTS="${HOSTS}${JOBSET_NAME}-workload-0-${i}.${SUBDOMAIN},"
  done
  export TPU_WORKER_HOSTNAMES=${HOSTS%,}
  echo "Computed TPU_WORKER_HOSTNAMES: ${TPU_WORKER_HOSTNAMES}"
fi

# --- Log for Debugging ---
echo "MaxText config overrides from command line:"
echo "  model_name='"'"'llama3.1-70b'"'"' skip_jax_distributed_system=True dtype='"'"'bfloat16'"'"' per_device_batch_size=2 profile_periodically_period=10000 async_checkpointing=False enable_checkpointing=False use_iota_embed=True remat_policy='"'"'custom'"'"' decoder_layer_input='"'"'device'"'"' context='"'"'device'"'"' query_proj='"'"'device'"'"' key_proj='"'"'device'"'"' value_proj='"'"'device'"'"' qkv_proj='"'"'device'"'"' ici_fsdp_parallelism=-1 dataset_type='"'"'synthetic'"'"' opt_type='"'"'adamw'"'"' mu_dtype='"'"'bfloat16'"'"' sa_block_q=1024 sa_block_kv=1024 sa_block_kv_compute=512 sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=256 tokenizer_type='"'"'tiktoken'"'"' tokenizer_path='"'"'assets/tokenizer_llama3.tiktoken'"'"' sa_q_layout='"'"'SEQ_MINOR'"'"' sa_k_layout='"'"'HEAD_DIM_MINOR'"'"' sa_v_layout='"'"'HEAD_DIM_MINOR'"'"' sa_use_fused_bwd_kernel=True use_tokamax_splash=True max_target_length=8192 profiler='"'"'xplane'"'"' skip_first_n_steps_for_profiler=5 profiler_steps=2 attention='"'"'flash'"'"' quantization='"'"'fp8_full'"'"' use_qwix_quantization=True weight_quantization_calibration_method='"'"'fixed,-224,224'"'"' act_quantization_calibration_method='"'"'fixed,-224,224'"'"' steps=30 base_output_directory=${ARTIFACT_DIR%/${JOB_IDENTIFIER}}"


# --- Main Execution ---
# The python process is run in the background (&)
(
  # Set up the environment
  export ENABLE_PATHWAYS_PERSISTENCE='"'"'1'"'"'
  export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"
  export ENABLE_PJRT_COMPATIBILITY='"'"'true'"'"'

  # libtpu reads XLA flags from LIBTPU_INIT_ARGS and splits it on single
  # spaces only, so re-join the config'"'"'s flags followed by the environment'"'"'s
  # (the last value of a repeated flag wins) with single spaces, globbing off.
  # XLA_FLAGS is then cleared because the CPU client rejects TPU-only flags.
  # On a Pathways head this runs on CPU, where LIBTPU_INIT_ARGS has no effect.
  set -f
  # shellcheck disable=SC2086
  set -- ${CONFIG_XLA_FLAGS} ${XLA_FLAGS:-}
  set +f
  if [ "$#" -gt 0 ]; then
    export LIBTPU_INIT_ARGS="$*"
  fi
  echo "LIBTPU_INIT_ARGS: ${LIBTPU_INIT_ARGS-<unset>}"
  unset XLA_FLAGS

  # Execute the training script, passing the config file AND the command-line overrides
  python3 -m maxtext.trainers.pre_train.train "${MAXTEXT_CONFIG_PATH}" model_name='"'"'llama3.1-70b'"'"' skip_jax_distributed_system=True dtype='"'"'bfloat16'"'"' per_device_batch_size=2 profile_periodically_period=10000 async_checkpointing=False enable_checkpointing=False use_iota_embed=True remat_policy='"'"'custom'"'"' decoder_layer_input='"'"'device'"'"' context='"'"'device'"'"' query_proj='"'"'device'"'"' key_proj='"'"'device'"'"' value_proj='"'"'device'"'"' qkv_proj='"'"'device'"'"' ici_fsdp_parallelism=-1 dataset_type='"'"'synthetic'"'"' opt_type='"'"'adamw'"'"' mu_dtype='"'"'bfloat16'"'"' sa_block_q=1024 sa_block_kv=1024 sa_block_kv_compute=512 sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=256 tokenizer_type='"'"'tiktoken'"'"' tokenizer_path='"'"'assets/tokenizer_llama3.tiktoken'"'"' sa_q_layout='"'"'SEQ_MINOR'"'"' sa_k_layout='"'"'HEAD_DIM_MINOR'"'"' sa_v_layout='"'"'HEAD_DIM_MINOR'"'"' sa_use_fused_bwd_kernel=True use_tokamax_splash=True max_target_length=8192 profiler='"'"'xplane'"'"' skip_first_n_steps_for_profiler=5 profiler_steps=2 attention='"'"'flash'"'"' quantization='"'"'fp8_full'"'"' use_qwix_quantization=True weight_quantization_calibration_method='"'"'fixed,-224,224'"'"' act_quantization_calibration_method='"'"'fixed,-224,224'"'"' steps=30 base_output_directory=${ARTIFACT_DIR%/${JOB_IDENTIFIER}} run_name=${JOB_IDENTIFIER}
) & PID=$!

# Wait for the python process to finish
wait $PID
EXIT_CODE=$?

echo "MaxText Job Ended at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE' ${KUEUE_QUEUE:+--queue "${KUEUE_QUEUE}"}
