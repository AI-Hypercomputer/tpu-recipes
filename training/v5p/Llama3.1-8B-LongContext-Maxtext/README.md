# Instructions for training Llama3.1-8B with long context on TPU v5p

This document presents steps to run an ultra long-context (up to 10M sequence length) Llama3.1-8B [MaxText](https://github.com/AI-Hypercomputer/maxtext) workload with ring context parallelism through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

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

## XPK setup

Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK.

## Run script

1. Clone [Maxtext](https://github.com/AI-Hypercomputer/maxtext) repo.
```
git clone https://github.com/AI-Hypercomputer/maxtext.git
```

2. Build a local docker image with default name `maxtext_base_image`.

```
cd maxtext
bash docker_build_dependency_image.sh MODE=stable DEVICE=tpu
```

3. (Optional) Install XPK if you haven't set it up.

```
pip install xpk
```

4. Specify workload configs.

```
export CLUSTER_NAME=v5p-demo #<your cluster name>
export WORKLOAD_NAME=llama3-1-8b-10m-test #<your workload name>
export RUN_NAME=llama3-1-8b-10m-run #<your run name>
export TPU_TYPE=v5p-128 #<your TPU Type: 64 chips / 128 cores>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export OUTPUT_PATH=gs://v5p-demo/ #<your GCS folder for results>
```

5. Copy `scripts/run_llama3.1-8b-long-context.sh` script, paste it to `src/maxtext/configs` folder, and run workload in the maxtext github root directory.

```
xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--base-docker-image maxtext_base_image \
--command "bash src/maxtext/configs/run_llama3.1-8b-long-context.sh RUN_NAME=${RUN_NAME} OUTPUT_PATH=${OUTPUT_PATH}"
```

The script defaults to a 10M (10,485,760) sequence length on a 64-chip (128-core) TPU v5p topology (`v5p-128`). Attention uses ring context parallelism with causal load balancing, Splash attention kernels, custom remat policy, and device context.

6. (Optional) Other sequence lengths.

Shorter contexts can use a smaller context parallelism degree, with the remaining chips assigned to FSDP:

| Sequence length | ici_context_parallelism | ici_fsdp_parallelism | per_device_batch_size |
| --------------- | ----------------------- | -------------------- | --------------------- |
| 1048576 (1M)    | 16                      | 8                    | 0.0625                |
| 2097152 (2M)    | 32                      | 4                    | 0.03125               |
| 5242880 (5M)    | 64                      | 2                    | 0.015625              |
| 10485760 (10M)  | 128                     | 1                    | 0.0078125             |

```
--command "bash src/maxtext/configs/run_llama3.1-8b-long-context.sh RUN_NAME=${RUN_NAME} OUTPUT_PATH=${OUTPUT_PATH} MAX_TARGET_LENGTH=1048576 ICI_CONTEXT_PARALLELISM=16 ICI_FSDP_PARALLELISM=8 PER_DEVICE_BATCH_SIZE=0.0625"
```

7. (Optional) If you need to delete any of your workload, you can run the following command:
```
export WORKLOAD_NAME_TO_DELETE=llama3-1-8b-10m-test

xpk workload delete \
--workload ${WORKLOAD_NAME_TO_DELETE} \
--cluster ${CLUSTER_NAME}
```
