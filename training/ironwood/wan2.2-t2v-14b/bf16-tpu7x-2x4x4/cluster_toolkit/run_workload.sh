#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Graceful Shutdown Logic ---
_sigterm() {
  echo "Caught SIGTERM signal, sending to child process..."
  kill -SIGTERM "$PID" 2>/dev/null
}
trap _sigterm SIGTERM

# Everything the run needs is rendered into this script, so it depends only on
# ARTIFACT_DIR, DATASET_DIR and JOB_IDENTIFIER being set by the recipe.
MAXDIFFUSION_BASE_CONFIG=src/maxdiffusion/configs/base_wan_27b.yml
MAXDIFFUSION_SCRIPT_PATH=src/maxdiffusion/train_wan.py
CONFIG_XLA_FLAGS='--xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_enable_async_all_gather=true --xla_max_concurrent_async_all_gathers=4 --xla_tpu_enable_async_collective_fusion=false --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false --xla_enable_async_all_reduce=true --xla_tpu_prefer_async_allgather_to_allreduce=true --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 --xla_tpu_enable_all_gather_offload_tracing=true --xla_tpu_enable_async_all_to_all=true --xla_tpu_overlap_compute_collective_tc=true --xla_latency_hiding_scheduler_rerun=5 --xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_scheduler_percent_shared_memory_limit=105 --xla_tpu_rwb_fusion=false --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false --xla_tpu_impure_enable_packed_bf16_math_ops=false --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true --xla_enable_transpose_trace=false'

# MaxDiffusion writes TensorBoard, checkpoints and metrics under
# ${output_dir}/${run_name}, so output_dir is the parent of ARTIFACT_DIR
# unless a GCSFuse checkpoint mount path is configured.
ARTIFACT_DIR="${ARTIFACT_DIR%/}"
OUTPUT_DIR="${ARTIFACT_DIR%/*}"

# --- Log for Debugging ---
echo "MaxDiffusion base config: ${MAXDIFFUSION_BASE_CONFIG}"
echo "MaxDiffusion script: ${MAXDIFFUSION_SCRIPT_PATH}"
echo "MaxDiffusion config overrides from command line:"
cat <<'CONFIG_OVERRIDES_EOF'
  attention='ulysses_ring_custom' ulysses_shards=4 ulysses_attention_chunks=2 mask_padding_tokens=False use_base2_exp=True weights_dtype='bfloat16' activations_dtype='bfloat16' guidance_scale_low=3.0 guidance_scale_high=4.0 boundary_ratio=0.875 flow_shift=12.0 fps=16 skip_jax_distributed_system=False train_data_dir="${DATASET_DIR}/train" load_tfrecord_cached=True height=1280 width=720 num_frames=81 num_inference_steps=40 prompt='a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.' opt_enable_grad_global_norm_clipping=False enable_profiler=True dataset_save_location="${DATASET_DIR}" remat_policy='CUSTOM' names_which_can_be_saved='['\''attn_output'\'', '\''self_attn'\'', '\''cross_attn'\'', '\''hidden_states'\'']' names_which_can_be_offloaded='[]' flash_block_sizes='{"block_q":2816,"block_kv":4096,"block_kv_compute":4096,"block_kv_compute_in":4096,"block_q_dkv":2048,"block_kv_dkv":4096,"block_kv_dkv_compute":4096,"block_kv_dkv_compute_in":256,"dq_reduction_steps":3,"use_fused_bwd_kernel":true}' flash_min_seq_length=0 seed=123456789 skip_first_n_steps_for_profiler=5 profiler_steps=10 per_device_batch_size=0.25 ici_data_parallelism=1 ici_fsdp_parallelism=16 ici_context_parallelism=4 ici_tensor_parallelism=1 vae_spatial=1 replicate_vae=True allow_split_physical_axes=True max_train_steps=30
CONFIG_OVERRIDES_EOF

# --- Main Execution ---
# The python process is run in the background (&)
(
  # Set up the environment
  export ENABLE_PATHWAYS_PERSISTENCE='1'
  export JAX_PLATFORMS='tpu,cpu'
  export ENABLE_PJRT_COMPATIBILITY='true'

  # libtpu reads XLA flags from LIBTPU_INIT_ARGS and splits it on single
  # spaces only, so re-join the config's flags followed by the environment's
  # (the last value of a repeated flag wins) with single spaces, globbing off.
  # XLA_FLAGS is then cleared because the CPU client rejects TPU-only flags.
  set -f
  # shellcheck disable=SC2086
  set -- ${CONFIG_XLA_FLAGS} ${XLA_FLAGS:-}
  set +f
  if [ "$#" -gt 0 ]; then
    export LIBTPU_INIT_ARGS="$*"
  fi
  unset XLA_FLAGS

  # Chained with && because set -e does not stop on a failure inside an &&
  # list, so a failed setup step would otherwise still start the run.
  pip install . && export HF_HUB_CACHE=/dev/shm && \
  python3 "${MAXDIFFUSION_SCRIPT_PATH}" "${MAXDIFFUSION_BASE_CONFIG}" attention='ulysses_ring_custom' ulysses_shards=4 ulysses_attention_chunks=2 mask_padding_tokens=False use_base2_exp=True weights_dtype='bfloat16' activations_dtype='bfloat16' guidance_scale_low=3.0 guidance_scale_high=4.0 boundary_ratio=0.875 flow_shift=12.0 fps=16 skip_jax_distributed_system=False train_data_dir="${DATASET_DIR}/train" load_tfrecord_cached=True height=1280 width=720 num_frames=81 num_inference_steps=40 prompt='a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.' opt_enable_grad_global_norm_clipping=False enable_profiler=True dataset_save_location="${DATASET_DIR}" remat_policy='CUSTOM' names_which_can_be_saved='['\''attn_output'\'', '\''self_attn'\'', '\''cross_attn'\'', '\''hidden_states'\'']' names_which_can_be_offloaded='[]' flash_block_sizes='{"block_q":2816,"block_kv":4096,"block_kv_compute":4096,"block_kv_compute_in":4096,"block_q_dkv":2048,"block_kv_dkv":4096,"block_kv_dkv_compute":4096,"block_kv_dkv_compute_in":256,"dq_reduction_steps":3,"use_fused_bwd_kernel":true}' flash_min_seq_length=0 seed=123456789 skip_first_n_steps_for_profiler=5 profiler_steps=10 per_device_batch_size=0.25 ici_data_parallelism=1 ici_fsdp_parallelism=16 ici_context_parallelism=4 ici_tensor_parallelism=1 vae_spatial=1 replicate_vae=True allow_split_physical_axes=True max_train_steps=30 base_output_directory="${ARTIFACT_DIR}" output_dir="${OUTPUT_DIR}" run_name="${JOB_IDENTIFIER}"
) & PID=$!

# Wait for the python process to finish
wait $PID
EXIT_CODE=$?

echo "MaxDiffusion Job Ended at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
