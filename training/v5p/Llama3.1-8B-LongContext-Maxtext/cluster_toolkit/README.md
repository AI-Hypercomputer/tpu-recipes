# Instructions for training Llama3.1-8B with long context (up to 14M) on TPU v5p

This document presents steps to run long-context (from 64K up to 14M tokens) Llama3.1-8B [MaxText](https://github.com/AI-Hypercomputer/maxtext) workloads with ring context parallelism on a 64-chip TPU v5p slice (`v5p-128`, topology `4x4x4`) through [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit) (`gcluster`).

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
bash src/dependencies/scripts/docker_build_dependency_image.sh MODE=stable DEVICE=tpu
export WORKLOAD_IMAGE=us-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${USER}_runner
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${WORKLOAD_IMAGE}
```

3. (Optional) Install Cluster Toolkit (`gcluster` v1.104.0) if you haven't set it up. `run_recipe.sh` prints the install commands if `gcluster` is not on your `PATH`.

4. Run the workload from the `cluster_toolkit` directory:

```bash
cd tpu-recipes/training/v5p/Llama3.1-8B-LongContext-Maxtext/cluster_toolkit
export PROJECT_ID=$PROJECT
export CLUSTER_NAME=$CLUSTER_NAME
export ZONE=$ZONE
export BASE_OUTPUT_DIR=$OUTPUT_DIR
export WORKLOAD_IMAGE=$WORKLOAD_IMAGE
chmod +x run_recipe.sh
./run_recipe.sh
```

The script defaults to a 1M (`1,048,576`) sequence length on a 64-chip TPU v5p slice (topology `4x4x4`, `v5p-128`) with `ici_context_parallelism=64`, `ici_fsdp_parallelism=1`, and `per_device_batch_size=0.03125` (global batch size `2`). Attention uses ring context parallelism with causal load balancing, Tokamax Splash attention kernels (`sa_block_*=1024`), custom remat policy, and attention context host offload (`context=offload`).

5. (Optional) Other sequence lengths (up to 14M tokens on 64 chips).

You can override sequence length, parallelism, and activation offload/remat flags via `KEY=VALUE` arguments to `./run_recipe.sh`:

| Sequence length | `ICI_CONTEXT_PARALLELISM` | `ICI_FSDP_PARALLELISM` | `PER_DEVICE_BATCH_SIZE` | `REMAT_POLICY` | `CONTEXT` | `DECODER_LAYER_INPUT` | Measured MFU (`v5p-128`) | Measured Step Time |
| --------------- | ------------------------- | ---------------------- | ----------------------- | -------------- | --------- | --------------------- | ------------------------ | ------------------ |
| 65536 (64K)     | 4                         | 16                     | 0.5                     | `custom`       | `device`  | `device`              | 67.3%                    | 10.25 s            |
| 262144 (256K)   | 16                        | 4                      | 0.125                   | `custom`       | `offload` | `offload`             | 68.1%                    | 26.31 s            |
| 1048576 (1M)    | 64                        | 1                      | 0.03125                 | `custom`       | `offload` | `device`              | 64.9% – 68.7%            | 95.73 s (UBench) / 90.30 s |
| 2097152 (2M)    | 64                        | 1                      | 0.03125                 | `custom`       | `offload` | `offload`             | 67.6% – 69.9%            | 357.68 s (UBench) / 346.18 s |
| 4194304 (4M)    | 64                        | 1                      | 0.03125                 | `custom`       | `offload` | `offload`             | 69.5% – 70.3%            | 1,373.73 s (UBench) / 1,358.40 s |
| 10485760 (10M)  | 64                        | 1                      | 0.015625                | `full`         | `device`  | `device`              | 36.7%                    | 8,060.20 s (UBench) / 8,056.40 s |
| 14680064 (14M)  | 64                        | 1                      | 0.015625                | `custom`       | `remat`   | `offload`             | 36.7% – 37.1%            | 15,766.82 s (UBench) / 15,623.09 s |

Examples:
```bash
# 2M context length on 64 chips (4x4x4):
./run_recipe.sh MAX_TARGET_LENGTH=2097152 ICI_CONTEXT_PARALLELISM=64 ICI_FSDP_PARALLELISM=1 PER_DEVICE_BATCH_SIZE=0.03125 REMAT_POLICY=custom CONTEXT=offload DECODER_LAYER_INPUT=offload

# 4M context length on 64 chips (4x4x4):
./run_recipe.sh MAX_TARGET_LENGTH=4194304 ICI_CONTEXT_PARALLELISM=64 ICI_FSDP_PARALLELISM=1 PER_DEVICE_BATCH_SIZE=0.03125 REMAT_POLICY=custom CONTEXT=offload DECODER_LAYER_INPUT=offload

# 10M context length on 64 chips (4x4x4):
./run_recipe.sh MAX_TARGET_LENGTH=10485760 ICI_CONTEXT_PARALLELISM=64 ICI_FSDP_PARALLELISM=1 PER_DEVICE_BATCH_SIZE=0.015625 REMAT_POLICY=full

# 14M context length on 64 chips (4x4x4):
./run_recipe.sh MAX_TARGET_LENGTH=14680064 ICI_CONTEXT_PARALLELISM=64 ICI_FSDP_PARALLELISM=1 PER_DEVICE_BATCH_SIZE=0.015625 REMAT_POLICY=custom CONTEXT=remat DECODER_LAYER_INPUT=offload
```

## Verified benchmark performance

The configurations below were verified on a 64-chip TPU v5p slice (`v5p-128`, topology `4x4x4`, 64 Megacore devices, `459 TFLOP/s/device` BF16 peak):

| Context Length | Parallelism & Offload Config | UBench Run Name & BigQuery `run_id` | Steady-State Step Time | Throughput (`TFLOP/s` & `Tokens/s/chip`) | Measured MFU (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1M** (`1,048,576`) | `CP=64, FSDP=1`, `pdbs=0.03125`, `sa_block_*=1024`, `context=offload` | `chengnuojin-ubench-wm87mo94` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-05_173709-20730566-a238-463e-918e-1f353c039538`) | **95.73 s** | **297.7 TFLOP/s** (`342.4 tok/s/chip`) | **64.86%** (`68.75%` in `O_h2_1024k_ring_cp64_b2_r2`) |
| **2M** (`2,097,152`) | `CP=64, FSDP=1`, `pdbs=0.03125`, `sa_block_*=1024`, `context=offload`, `decoder_layer_input=offload` | `chengnuojin-ubench-w3jvwbkr` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-05_183624-083a4c98-0783-46af-a0a8-cd91be9efeb5`) | **357.68 s** | **310.4 TFLOP/s** (`183.2 tok/s/chip`) | **67.63%** (`69.88%` in `O_r5_2m_ring_b2_inoff`) |
| **4M** (`4,194,304`) | `CP=64, FSDP=1`, `pdbs=0.03125`, `sa_block_*=1024`, `context=offload`, `decoder_layer_input=offload` | `chengnuojin-ubench-asb6kxmv` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-05_215125-766e1e52-4ecf-4a87-a439-8a0d23bacc1c`) | **1,373.73 s** | **319.0 TFLOP/s** (`95.4 tok/s/chip`) | **69.50%** (`70.29%` in `O_r5_4m_ring_b2_inoff`) |
| **10M** (`10,485,760`) | `CP=64, FSDP=1`, `pdbs=0.015625`, `sa_block_*=1024`, `remat_policy=full` | `chengnuojin-ubench-uujda4kd` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-06_140929-befcf112-30da-4290-a424-9d9f39be4deb`) | **8,060.20 s** (`8,056.40 s` ref) | **168.5 TFLOP/s** (`20.3 tok/s/chip`) | **36.72% – 36.74%** |
| **14M** (`14,680,064`) | `CP=64, FSDP=1`, `pdbs=0.015625`, `sa_block_*=1024`, `context=remat`, `decoder_layer_input=offload` | `chengnuojin-ubench-bjzjtsvd` (`maxtext_training_with_xpk-llama3_1_8b-2026-10-06_141408-f2a71578-1159-4612-b1a5-c97197dd1493`) | **15,766.82 s** (`15,623.09 s` ref) | **168.6 TFLOP/s** (`170.2` ref, `14.5 tok/s/chip`) | **36.73% – 37.07%** |

## Cleanup

If you need to cancel or clean up your workload, run:

```bash
export WORKLOAD_NAME_TO_DELETE=llama3-1-8b-1m-test

gcluster job cancel ${WORKLOAD_NAME_TO_DELETE} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```
