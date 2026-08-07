# Setup

## Step 1: Installing the dependencies:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc

conda create -n tpu python=3.10 
source activate tpu

git clone https://github.com/google/maxdiffusion.git && cd maxdiffusion
git checkout main

pip install -e .
pip install -r requirements.txt
pip install -U --pre jax[tpu] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Step 2: Running the inference benchmarks

Run the command below to benchmark Z-Image-Turbo inference:

### Z-Image-Turbo
```bash
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

python src/maxdiffusion/generate_zimage.py src/maxdiffusion/configs/base_zimage_turbo.yml run_name="my_run"
```

### Using `run.sh`
You can also run the benchmark using the provided `run.sh` script:
```bash
bash run.sh src/maxdiffusion/configs/base_zimage_turbo.yml run_name="my_run"
```

---

## Attention Kernel Configurations

### `tokamax_flash` Attention Kernel
MaxDiffusion supports `dot_product`, `flash`, and `tokamax_flash` attention mechanisms. `tokamax_flash` is a high-performance custom attention kernel optimized for TPU v6e (Trillium) and other TPU platforms.

To configure the attention kernel, set:
```yaml
attention: 'tokamax_flash'
```

### Flash Block Sizes (`flash_block_sizes`)
Tuning block sizes can help align computation tiles to the hardware structure of TPUs (e.g. Trillium). The optimal block sizes configuration for Z-Image-Turbo can be customized like so:
```yaml
flash_block_sizes: {
  "block_q" : 1024,
  "block_kv_compute" : 512,
  "block_kv" : 1024,
  "block_q_dkv" : 1024,
  "block_kv_dkv" : 1024,
  "block_kv_dkv_compute" : 512,
  "block_q_dq" : 1024,
  "block_kv_dq" : 1024,
  "use_fused_bwd_kernel": True,
}
```

---

## Performance and Mesh Sharding (TPU v6e-8)

When running 1024x1024 image generation with Z-Image-Turbo (9 denoising steps), the default mesh sharding is `data=8`. Measured alternatives at batch size 8 on TPU v6e-8 include:
- `data=8`: Default mesh sharding 0.28s/img
- `ctx=8`: 0.519s/img
- `fsdp=8`: 1.179s/img
