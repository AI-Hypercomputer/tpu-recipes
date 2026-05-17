export JAX_PLATFORMS="tpu,cpu"

export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_spmd_threshold_for_allgather_cse=1000000 --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000'
LIBTPU_INIT_ARGS+=' --xla_sc_disable_megacore_partitioning=true --xla_tpu_use_tc_device_shape_on_sc=true --tpu_use_continuations=true --xla_sc_enable_instruction_fusion=false --xla_sc_disjoint_spmem=false --2a886c8_chip_config_name=megachip_tccontrol --xla_jf_crs_combiner_threshold_count=10 --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true'

checkpoints=${OUTPUT_DIR}/checkpoints
dataset_path=${OUTPUT_DIR}/dataset

ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true python src/maxdiffusion/train_sdxl.py src/maxdiffusion/configs/base_xl.yml pretrained_model_name_or_path=${checkpoints}/models--stabilityai--stable-diffusion-xl-base-1.0 \
revision=refs/pr/95 activations_dtype=bfloat16 weights_dtype=bfloat16 dataset_name=${dataset_path}/pokemon-gpt4-captions_xl resolution=1024 per_device_batch_size=1 jax_cache_dir=${OUT_DIR}/cache_dir/ max_train_steps=100 attention=flash run_name=trillium-sdxl enable_profiler=True output_dir=${OUT_DIR}
