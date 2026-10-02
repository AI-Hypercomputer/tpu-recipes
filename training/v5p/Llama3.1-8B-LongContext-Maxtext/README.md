# Instructions for training Llama3.1-8B with long context on TPU v5p

This document presents steps to run a long-context (1M sequence length) Llama3.1-8B [MaxText](https://github.com/AI-Hypercomputer/maxtext) workload with ring context parallelism on a 64-chip TPU v5p slice (`v5p-128`) through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

## Model configuration

This recipe uses the stock MaxText `llama3.1-8b` model config (`head_dim=128`, `base_num_query_heads=32`, `base_num_kv_heads=8`) with no model overrides.

## Performance

Measured on TPU v5p-128 (4x4x4 topology, 64 chips, 1 slice) with the default settings of `scripts/run_llama3.1-8b-long-context.sh`:

| Sequence length | Global batch size | Step time | TFLOP/s/chip | MFU    | Tokens/s/chip |
| --------------- | ----------------- | --------- | ------------ | ------ | ------------- |
| 1048576 (1M)    | 2                 | 111.05 s  | 256.6        | 55.9%  | 295           |

Software versions used for the measurement: MaxText `9af4c5e`, JAX `0.11.1.dev20260811`, libtpu `0.0.45`. Results with other versions may differ.

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
export WORKLOAD_NAME=llama3-1-8b-1m-test #<your workload name>
export RUN_NAME=llama3-1-8b-1m-run #<your run name>
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

The script defaults to a 1M (1,048,576) sequence length on a 64-chip (128-core) TPU v5p topology (`v5p-128`, one JAX device per chip) with `per_device_batch_size=0.03125` (global batch size 2). Attention uses ring context parallelism across all 64 chips (`ici_context_parallelism=64`, `ici_fsdp_parallelism=1`) with causal load balancing and Splash attention kernels, the `custom` remat policy, and host-offloaded context (`context=offload`).

6. (Optional) If you need to delete any of your workload, you can run the following command:
```
export WORKLOAD_NAME_TO_DELETE=llama3-1-8b-1m-test

xpk workload delete \
--workload ${WORKLOAD_NAME_TO_DELETE} \
--cluster ${CLUSTER_NAME}
```
