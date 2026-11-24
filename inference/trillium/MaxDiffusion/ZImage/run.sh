#!/bin/bash
set -e

# Default to base_zimage_turbo.yml config if no config argument is provided
if [ "$#" -gt 0 ] && [[ "$1" == *.yml || "$1" == *.yaml ]]; then
  CONFIG_FILE="$1"
  shift
else
  CONFIG_FILE="src/maxdiffusion/configs/base_zimage_turbo.yml"
fi

export LIBTPU_INIT_ARGS='--xla_tpu_dvfs_p_state=7 \
--xla_tpu_spmd_rng_bit_generator_unsafe=true \
--xla_tpu_enable_dot_strength_reduction=true \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_enable_async_collective_permute=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true \
--xla_tpu_scoped_vmem_limit_kib=131072 \
--xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_tpu_enable_ici_ag_pipelining=true \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2 \
--xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_tpu_enable_ag_backward_pipelining=true \
--xla_tpu_enable_megacore_fusion=true \
--xla_tpu_megacore_fusion_allow_ags=true \
--xla_tpu_assign_all_reduce_scatter_layout=true'

python src/maxdiffusion/generate_zimage.py "${CONFIG_FILE}" \
  run_name="my_run" \
  "$@"
