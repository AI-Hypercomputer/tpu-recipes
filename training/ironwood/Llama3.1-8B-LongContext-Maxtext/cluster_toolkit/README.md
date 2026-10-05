# Instructions for training Llama3.1-8B with long context (up to 16M) on TPU Ironwood (v7x)

This document presents steps to run ultra long-context (from 64K up to 16M tokens) Llama3.1-8B [MaxText](https://github.com/AI-Hypercomputer/maxtext) workloads with ring context parallelism on a 64-chip TPU Ironwood slice (`tpu7x-4x4x4`, 128 TensorCore devices) through [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit) (`gcluster`).

## Model configuration

> [!IMPORTANT]
> This recipe trains a **variant** of Llama3.1-8B, not the stock architecture. The attention shape is overridden on the command line (`override_model_config=true`) to use fewer, wider heads:
>
> | Config | Stock `llama3.1-8b` | This recipe |
> | ---------------------- | ------------------- | ----------- |
> | `head_dim`             | 128                 | 256         |
> | `base_num_query_heads` | 32                  | 16          |
> | `base_num_kv_heads`    | 8                   | 4           |
>
> The total attention width (`base_num_query_heads * head_dim = 4096`) and the 4:1 GQA ratio are unchanged, so the parameter count is identical to stock Llama3.1-8B. Every other model dimension (`base_emb_dim`, `base_mlp_dim`, `base_num_decoder_layers`, `vocab_size`) is stock. The performance numbers in this recipe were measured with this variant; running the stock head configuration will give different results.

## Cluster Toolkit setup

Please follow the [Cluster Toolkit Cloud TPU deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview) to create your GKE cluster with Cluster Toolkit (`gcluster`) v1.104.0.

## Run script

1. Clone [MaxText](https://github.com/AI-Hypercomputer/maxtext) repo.
```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
```

2. Build a docker image and push it to an Artifact Registry Docker repository (`REPOSITORY`).

```bash
cd maxtext
bash src/dependencies/scripts/docker_build_dependency_image.sh MODE=nightly DEVICE=tpu
export WORKLOAD_IMAGE=us-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${USER}_runner
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${WORKLOAD_IMAGE}
```

3. (Optional) Install Cluster Toolkit (`gcluster` v1.104.0) if you haven't set it up. `run_recipe.sh` prints the install commands if `gcluster` is not on your `PATH`.

4. Run the workload from the `cluster_toolkit` directory:

```bash
cd tpu-recipes/training/ironwood/Llama3.1-8B-LongContext-Maxtext/cluster_toolkit
export PROJECT_ID=$PROJECT
export CLUSTER_NAME=$CLUSTER_NAME
export ZONE=$ZONE
export BASE_OUTPUT_DIR=$OUTPUT_DIR
export WORKLOAD_IMAGE=$WORKLOAD_IMAGE
chmod +x run_recipe.sh
./run_recipe.sh
```

The script defaults to a 10M (`10,485,760`) sequence length on a 64-chip (`128-TensorCore`) TPU Ironwood slice (topology `4x4x4`, `tpu7x-4x4x4`) with `ici_context_parallelism=128`, `ici_fsdp_parallelism=1`, and `per_device_batch_size=0.0078125` (global batch size `1`). Attention uses ring context parallelism with causal load balancing, Tokamax Splash attention kernels (`sa_block_*=2048` with `96 MiB` scoped VMEM via `--xla_tpu_scoped_vmem_limit_kib=98304` and `--xla_tpu_dvfs_p_state=7`), custom remat policy, and device context.

5. (Optional) Other sequence lengths (from 64K up to 16M tokens on 64 chips).

You can override sequence length, parallelism, and activation/parameter host offload flags via `KEY=VALUE` arguments to `./run_recipe.sh`:

| Sequence length | `ICI_CONTEXT_PARALLELISM` | `ICI_FSDP_PARALLELISM` | `PER_DEVICE_BATCH_SIZE` | `CONTEXT` | `DECODER_LAYER_INPUT` | `OPTIMIZER_MEMORY_HOST_OFFLOAD` | `PARAMETER_MEMORY_HOST_OFFLOAD` | Measured MFU (`tpu7x-4x4x4`) | Measured Step Time |
| --------------- | ------------------------- | ---------------------- | ----------------------- | --------- | --------------------- | ------------------------------- | ------------------------------- | ---------------------------- | ------------------ |
| 65536 (64K)     | 2                         | 64                     | 0.5                     | `device`  | `device`              | `false`                         | `false`                         | 51.7%                        | 5.31 s             |
| 262144 (256K)   | 8                         | 16                     | 0.125                   | `device`  | `device`              | `false`                         | `false`                         | 50.7%                        | 14.08 s            |
| 1048576 (1M)    | 32                        | 4                      | 0.03125                 | `device`  | `device`              | `false`                         | `false`                         | 50.0% – 50.9%                | 49.58 s (UBench) / 48.54 s |
| 2097152 (2M)    | 64                        | 2                      | 0.015625                | `device`  | `device`              | `false`                         | `false`                         | 50.8% – 51.0%                | 94.60 s (UBench) / 94.74 s |
| 4194304 (4M)    | 128                       | 1                      | 0.0078125               | `device`  | `device`              | `false`                         | `false`                         | 50.9% (44.6% w/o p7)         | 186.48 s / 213.68 s |
| 10485760 (10M)  | 128                       | 1                      | 0.0078125               | `device`  | `device`              | `true`                          | `false`                         | 52.1% – 52.5%                | 1,124.12 s (UBench) / 1,130.04 s |
| 14680064 (14M)  | 128                       | 1                      | 0.0078125               | `offload` | `offload`             | `true`                          | `true`                          | 48.7% – 48.8%                | 2,361.05 s (UBench) / 2,365.96 s |
| 16777216 (16M)  | 128                       | 1                      | 0.0078125               | `offload` | `offload`             | `true`                          | `true`                          | 48.8%                        | 3,082.71 s         |

Examples:
```bash
# 1M context length on 64 Ironwood chips (4x4x4):
./run_recipe.sh MAX_TARGET_LENGTH=1048576 ICI_CONTEXT_PARALLELISM=32 ICI_FSDP_PARALLELISM=4 PER_DEVICE_BATCH_SIZE=0.03125 OPTIMIZER_MEMORY_HOST_OFFLOAD=false

# 2M context length on 64 Ironwood chips (4x4x4):
./run_recipe.sh MAX_TARGET_LENGTH=2097152 ICI_CONTEXT_PARALLELISM=64 ICI_FSDP_PARALLELISM=2 PER_DEVICE_BATCH_SIZE=0.015625 OPTIMIZER_MEMORY_HOST_OFFLOAD=false

# 4M context length on 64 Ironwood chips (4x4x4):
./run_recipe.sh MAX_TARGET_LENGTH=4194304 ICI_CONTEXT_PARALLELISM=128 ICI_FSDP_PARALLELISM=1 PER_DEVICE_BATCH_SIZE=0.0078125 OPTIMIZER_MEMORY_HOST_OFFLOAD=false

# 10M context length on 64 Ironwood chips (4x4x4, default):
./run_recipe.sh MAX_TARGET_LENGTH=10485760 ICI_CONTEXT_PARALLELISM=128 ICI_FSDP_PARALLELISM=1 PER_DEVICE_BATCH_SIZE=0.0078125 CONTEXT=device OPTIMIZER_MEMORY_HOST_OFFLOAD=true

# 14M context length on 64 Ironwood chips (4x4x4):
./run_recipe.sh MAX_TARGET_LENGTH=14680064 ICI_CONTEXT_PARALLELISM=128 ICI_FSDP_PARALLELISM=1 PER_DEVICE_BATCH_SIZE=0.0078125 CONTEXT=offload DECODER_LAYER_INPUT=offload OPTIMIZER_MEMORY_HOST_OFFLOAD=true PARAMETER_MEMORY_HOST_OFFLOAD=true
```

## Verified benchmark performance

The configurations below were verified on a 64-chip TPU Ironwood (`v7x`) slice (`tpu7x-4x4x4`, 128 TensorCore devices, `1,153.5 TFLOP/s/core` / `2,307 TFLOP/s/chip` BF16 peak):

| Context Length | Parallelism & Offload Config | UBench / Benchmark Run ID | Steady-State Step Time | Measured TFLOP/s/core | Measured MFU (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1M** (`1,048,576`) | `CP=32, FSDP=4`, `pdbs=0.03125`, `sa_block_*=2048`, `context=device`, `dvfs_p_state=7` | `chengnuojin-ubench-iwwssuhp` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-05_172350-76c5205d-c288-4aed-89e2-375398c92c9a`) | **49.58 s** | **574.8 TFLOP/s** (`1,149.6 TFLOP/s/chip`) | **49.98%** (`50.90%` in `B3F_1m_cp32_ideal`) |
| **2M** (`2,097,152`) | `CP=64, FSDP=2`, `pdbs=0.015625`, `sa_block_*=2048`, `context=device`, `dvfs_p_state=7` | `chengnuojin-ubench-c79shbyc` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-05_174711-a6caf10a-50e6-423e-a0ad-ef2090917698`) | **94.60 s** | **586.9 TFLOP/s** (`1,173.8 TFLOP/s/chip`) | **51.03%** (`50.80%` in `B3F_2m_cp64_IDEALBRACKET`) |
| **4M** (`4,194,304`) | `CP=128, FSDP=1`, `pdbs=0.0078125`, `sa_block_*=2048`, `context=device` | `chengnuojin-ubench-8vkgzrtc` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-05_182345-03a74883-8a36-41ae-aace-a7e443e87e43`) | **213.68 s** (`186.48 s` w/ `p7`) | **512.7 TFLOP/s** (`587.5 TFLOP/s` w/ `p7`) | **44.60%** (`50.93%` w/ `p7` in `B3F_4m_cp128_IDEALBRACKET`) |
| **10M** (`10,485,760`) | `CP=128, FSDP=1`, `pdbs=0.0078125`, `sa_block_*=2048`, `context=device`, `opt_off=true`, `dvfs_p_state=7` | `chengnuojin-ubench-kabnsj6h` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-05_223247-053f00f3-e418-43ce-b1bf-0e567cec4625`) | **1,124.12 s** | **604.2 TFLOP/s** (`1,208.5 TFLOP/s/chip`) | **52.54%** (`52.11%` in `B3F_10m_cp128_IDEALBRACKET`) |
| **14M** (`14,680,064`) | `CP=128, FSDP=1`, `pdbs=0.0078125`, `sa_block_*=2048`, `context=offload`, `decoder_layer_input=offload`, `opt_off=true`, `par_off=true`, `dvfs_p_state=7` | `chengnuojin-ubench-j4kidcsm` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-05_223942-c16bccfa-ee84-462f-9f2a-1cdbd721b32b`) | **2,361.05 s** | **563.0 TFLOP/s** (`1,125.9 TFLOP/s/chip`) | **48.80%** (`48.71%` in `B3F_14m_cp128_inoff`) |

## Cleanup

If you need to cancel or clean up your workload, run:

```bash
export WORKLOAD_NAME_TO_DELETE=llama3-1-8b-10m-test

gcluster job cancel ${WORKLOAD_NAME_TO_DELETE} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```
