# Instructions for training Mixtral-8X7B Maxtext on TPU v5p

This documents present steps to run Mixtral-8x7B [MaxText](https://github.com/google/maxtext) workload through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

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
export WORKLOAD_NAME=Mixtral-8x7b-test #<your workload name>
export RUN_NAME=Mixtral-8x7b-run #<your run name>
export TPU_TYPE=v5p-128 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export OUTPUT_PATH=gs://v5p-demo/ #<your GCS folder for results>
```

5. Copy `scripts/run_mixtral-8x7b.sh` script, paste it to `MaxText/configs` folder, and run workload in the maxtext github root directory.

```
xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--base-docker-image maxtext_base_image \
--command "bash MaxText/configs/run_mixtral-8x7b.sh RUN_NAME=${RUN_NAME} OUTPUT_PATH=${OUTPUT_PATH}"
```

6. (Optional) Clean up the GKE cluster.

```
xpk cluster delete --cluster ${CLUSTER_NAME}
```
