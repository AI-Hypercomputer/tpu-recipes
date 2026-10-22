# Instructions for training Llama3.1-8B with long context on TPU v5p

This document presents steps to run a long-context (up to 1M sequence length) Llama3.1-8B [MaxText](https://github.com/AI-Hypercomputer/maxtext) workload with ring context parallelism through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

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
export TPU_TYPE=v5p-128 #<your TPU Type>
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

The script defaults to a 1M sequence length. Attention uses ring context parallelism with causal load balancing, so per-chip communication overlaps with attention compute, and the attention context tensors are offloaded to host memory to fit the activations.

6. (Optional) Other sequence lengths.

Shorter contexts use a smaller context parallelism degree, with the remaining chips assigned to FSDP. Pass the overrides as arguments to the script:

| Sequence length | ici_context_parallelism | ici_fsdp_parallelism | per_device_batch_size |
| --------------- | ----------------------- | -------------------- | --------------------- |
| 262144 (256K)   | 16                      | 4                    | 0.125                 |
| 524288 (512K)   | 16                      | 4                    | 0.125                 |
| 1048576 (1M)    | 64                      | 1                    | 0.03125               |

```
--command "bash src/maxtext/configs/run_llama3.1-8b-long-context.sh RUN_NAME=${RUN_NAME} OUTPUT_PATH=${OUTPUT_PATH} MAX_TARGET_LENGTH=262144 ICI_CONTEXT_PARALLELISM=16 ICI_FSDP_PARALLELISM=4 PER_DEVICE_BATCH_SIZE=0.125"
```

7. (Optional) If you need to delete any of your workload, you can run the following command:
```
export WORKLOAD_NAME_TO_DELETE=llama3-1-8b-1m-test

xpk workload delete \
--workload ${WORKLOAD_NAME_TO_DELETE} \
--cluster ${CLUSTER_NAME}
```
