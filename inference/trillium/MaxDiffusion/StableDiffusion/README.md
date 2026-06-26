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

MaxDiffusion supports multiple Stable Diffusion models. Run the appropriate command below depending on the model version you wish to benchmark.

### Stable Diffusion 1.5
```bash
LIBTPU_INIT_ARGS="--xla_tpu_rwb_fusion=false --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_scoped_vmem_limit_kib=131072" python -m src.maxdiffusion.generate src/maxdiffusion/configs/base15.yml run_name="my_run"
```

### Stable Diffusion 2.1
```bash
LIBTPU_INIT_ARGS="--xla_tpu_rwb_fusion=false --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_scoped_vmem_limit_kib=131072" python -m src.maxdiffusion.generate src/maxdiffusion/configs/base21.yml run_name="my_run"
```

### Stable Diffusion XL
```bash
LIBTPU_INIT_ARGS="--xla_tpu_rwb_fusion=false --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_scoped_vmem_limit_kib=131072" python -m src.maxdiffusion.generate src/maxdiffusion/configs/base_xl.yml run_name="my_run"
```

---

## Attention Kernel Configurations

### `tokamax_flash` Attention Kernel
MaxDiffusion supports `dot_product`, `flash`, and `tokamax_flash` attention mechanisms. `tokamax_flash` is a high-performance custom attention kernel optimized for TPU v6e (Trillium) and other TPU platforms.

To configure the attention kernel, set:
```yaml
attention: 'tokamax_flash'
```

### Flash Attention Block Filtering (`flash_min_seq_length`)
Using Flash Attention on short sequences can sometimes introduce overhead compared to standard dot product attention. To restrict Flash Attention to larger, compute-intensive attention blocks, you can configure a minimum sequence length threshold:
```yaml
flash_min_seq_length: 4096
```
For example, in Stable Diffusion 1.5 (1024x1024 inference), the two largest self-attention sequence lengths are 16,384 and 4,096. Setting `flash_min_seq_length: 4096` ensures that only these two large self-attention blocks use Flash Attention, while cross-attention (which has a short text sequence length of 77) falls back to the standard dot product attention.

### Flash Block Sizes (`flash_block_sizes`)
Tuning block sizes can help align computation tiles to the hardware structure of TPUs (e.g. Trillium). The optimal block sizes configuration for queries (`block_q`) and keys/values (`block_kv`) can be customized like so:
```yaml
flash_block_sizes: {
  "block_q" : 2048,
  "block_kv_compute" : 1024,
  "block_kv" : 1024,
  "block_q_dkv" : 2048,
  "block_kv_dkv" : 1024,
  "block_kv_dkv_compute" : 1024
}
```
