# Instructions for training Llama3.1-8B with long context on TPU Ironwood (v7x)

This document presents steps to run an ultra long-context (up to 10M sequence length) Llama3.1-8B [MaxText](https://github.com/AI-Hypercomputer/maxtext) workload with ring context parallelism through the [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

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
export CLUSTER_NAME=tpu7x-cluster #<your cluster name>
export WORKLOAD_NAME=llama3-1-8b-10m-test #<your workload name>
export RUN_NAME=llama3-1-8b-10m-run #<your run name>
export TPU_TYPE=tpu7x-4x4x4 #<your TPU Type: 64 chips / 128 cores>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export OUTPUT_PATH=gs://your-bucket/ #<your GCS folder for results>
```

5. Copy `scripts/run_llama3.1-8b-long-context.sh` script, paste it to `src/maxtext/configs` folder, and run workload in the maxtext github root directory.

```
xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--device-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--base-docker-image maxtext_base_image \
--command "bash src/maxtext/configs/run_llama3.1-8b-long-context.sh RUN_NAME=${RUN_NAME} OUTPUT_PATH=${OUTPUT_PATH}"
```

The script defaults to a 10M (10,485,760) sequence length on a 64-chip (128-core) TPU v7x topology (`tpu7x-4x4x4`). Attention uses ring context parallelism with causal load balancing, Splash attention kernels, custom remat policy, and device context.

6. (Optional) Other sequence lengths.

Shorter contexts can use a smaller context parallelism degree, with remaining chips assigned to FSDP:

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
