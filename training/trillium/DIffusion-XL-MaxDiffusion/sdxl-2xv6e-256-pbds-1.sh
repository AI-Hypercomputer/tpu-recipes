export JAX_PLATFORMS="tpu,cpu"

checkpoints=${OUTPUT_DIR}/checkpoints
dataset_path=${OUTPUT_DIR}/dataset

ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true python src/maxdiffusion/train_sdxl.py src/maxdiffusion/configs/base_xl.yml pretrained_model_name_or_path=${checkpoints}/models--stabilityai--stable-diffusion-xl-base-1.0 \
revision=refs/pr/95 activations_dtype=bfloat16 weights_dtype=bfloat16 dataset_name=${dataset_path}/pokemon-gpt4-captions_xl resolution=1024 per_device_batch_size=1 jax_cache_dir=${OUT_DIR}/cache_dir/ max_train_steps=100 attention=flash run_name=trillium-sdxl enable_profiler=True output_dir=${OUT_DIR}
