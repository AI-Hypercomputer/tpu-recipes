#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Graceful Shutdown Logic ---
_sigterm() {
  echo "Caught SIGTERM signal, sending to child process..."
  kill -SIGTERM "$PID" 2>/dev/null
}
trap _sigterm SIGTERM

MAXTEXT_CONFIG_PATH="maxtext/configs/base.yml"
# The config's own XLA flags. The workload's arrive in XLA_FLAGS.
CONFIG_XLA_FLAGS='--xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_all_gather_offload_tracing=true --xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false --xla_enable_async_all_gather=true --xla_tpu_prefer_async_allgather_to_allreduce=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_offloading_copy_to_sparsecore=false --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=false'

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
echo "  model_name='llama3.1-405b' skip_jax_distributed_system=True dtype='bfloat16' per_device_batch_size=3 profile_periodically_period=10000 async_checkpointing=False enable_checkpointing=False use_iota_embed=True remat_policy='custom' decoder_layer_input='offload' ici_fsdp_parallelism=-1 dataset_type='synthetic' opt_type='adamw' quantization='fp8_full' use_qwix_quantization=True mu_dtype='bfloat16' use_tokamax_splash=True use_max_logit_estimate=-1 sa_block_kv=2048 sa_block_kv_compute=256 sa_block_q=1024 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=1024 sa_block_q_dkv=2048 sa_k_layout='SEQ_MINOR' sa_q_layout='HEAD_DIM_MINOR' sa_v_layout='SEQ_MINOR' attention='flash' sa_use_fused_bwd_kernel=True max_target_length=8192 profiler='xplane' weight_quantization_calibration_method='fixed,-224,224' act_quantization_calibration_method='fixed,-224,224' steps=30 base_output_directory=${ARTIFACT_DIR%/${JOB_IDENTIFIER}}"

# --- RL Sampler Patches & Env Setup ---
if [ "maxtext.trainers.pre_train.train" = "maxtext.trainers.post_train.rl.train_rl" ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export HF_HUB_DISABLE_XET=1
  export PYTHONPATH=/app/src:$PYTHONPATH
  export NUM_PRECOMPILE_WORKERS=1
  export NEW_MODEL_DESIGN=1
  export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
fi

# --- Main Execution ---
# The python process is run in the background (&)
(
  # Set up the environment
  export ENABLE_PATHWAYS_PERSISTENCE='1'
  export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"
  export ENABLE_PJRT_COMPATIBILITY='true'

  # libtpu reads XLA flags from LIBTPU_INIT_ARGS and splits it on single
  # spaces only, so re-join the config's flags followed by the environment's
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
  python3 -m maxtext.trainers.pre_train.train "${MAXTEXT_CONFIG_PATH}" model_name='llama3.1-405b' skip_jax_distributed_system=True dtype='bfloat16' per_device_batch_size=3 profile_periodically_period=10000 async_checkpointing=False enable_checkpointing=False use_iota_embed=True remat_policy='custom' decoder_layer_input='offload' ici_fsdp_parallelism=-1 dataset_type='synthetic' opt_type='adamw' quantization='fp8_full' use_qwix_quantization=True mu_dtype='bfloat16' use_tokamax_splash=True use_max_logit_estimate=-1 sa_block_kv=2048 sa_block_kv_compute=256 sa_block_q=1024 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=1024 sa_block_q_dkv=2048 sa_k_layout='SEQ_MINOR' sa_q_layout='HEAD_DIM_MINOR' sa_v_layout='SEQ_MINOR' attention='flash' sa_use_fused_bwd_kernel=True max_target_length=8192 profiler='xplane' weight_quantization_calibration_method='fixed,-224,224' act_quantization_calibration_method='fixed,-224,224' steps=30 base_output_directory=${ARTIFACT_DIR%/${JOB_IDENTIFIER}} run_name=${JOB_IDENTIFIER}
) & PID=$!

# Wait for the python process to finish
wait $PID
EXIT_CODE=$?

echo "MaxText Job Ended at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
