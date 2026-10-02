# Instructions for training Llama3.1-8B with long context (1M) on TPU v5p

This document presents steps to run a 1M (1,048,576 tokens) long-context Llama3.1-8B [MaxText](https://github.com/AI-Hypercomputer/maxtext) workload with ring context parallelism through [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit) (`gcluster`).

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

1. Clone [Maxtext](https://github.com/AI-Hypercomputer/maxtext) repo.
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
./run_recipe.sh
```

The script runs a 1M (1,048,576) sequence length on a 128-chip TPU v5p slice (topology `4x4x8`, `v5p-256`) with `ici_context_parallelism=16`, `ici_fsdp_parallelism=8`, and `per_device_batch_size=0.0625`. Attention uses ring context parallelism with causal load balancing, Splash attention kernels, custom remat policy, and device context.

## Verified benchmark performance

The 1M configuration was benchmarked on TPU v5p `4x4x8` (128 chips / 32 hosts of `ct5p-hightpu-4t`):

| Metric | Measured Value |
| :--- | :--- |
| **Sequence Length** | 1,048,576 tokens (1M) |
| **Topology** | TPU v5p `4x4x8` (128 chips, `v5p-256`) |
| **Mesh Parallelism** | `ici_context_parallelism=16`, `ici_fsdp_parallelism=8` |
| **Per-Device Batch Size** | 0.0625 (Global Batch = 8 sequences) |
| **Steady-State Step Time** | 401.62 s (std: 0.006 s) |
| **Per-Chip Throughput** | 163.18 tok/s/chip |
| **Total Workload Throughput** | 20,886.89 tok/s |
| **TFLOP/s/device** | 141.91 (30.92% MFU of 459 TFLOP/s peak) |
| **Exit Code** | 0 (All 32 workers completed successfully) |

## Cleanup

If you need to cancel or clean up your workload, run:

```bash
export WORKLOAD_NAME_TO_DELETE=llama3-1-8b-1m-test

gcluster job cancel ${WORKLOAD_NAME_TO_DELETE} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --location ${ZONE}
```
