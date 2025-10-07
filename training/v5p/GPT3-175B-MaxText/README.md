# Instructions for training GPT3-175B-Maxtext on TPU v5p

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext GPT3-175B workloads on GKE
1. Clone [Maxtext](https://github.com/AI-Hypercomputer/maxtext) repo and move to its directory
```
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
```

2. Run the following commands to build the docker image
```
bash docker_build_dependency_image.sh MODE=stable DEVICE=tpu
```

3. Upload your docker image to Container Registry
```
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${USER}_runner
```

4. Create your GCS bucket
```
GCS_PATH=gs://v5p-demo #<your_GCS_folder_for_results>
gcloud storage buckets create ${GCS_PATH}  --project ${PROJECT}
```

5. Specify your workload configs
```
export CLUSTER_NAME=v5p-demo #<your_cluster_name>
export WORKLOAD_NAME=gpt3-175b-test #<your_workload_name>
export RUN_NAME=gpt3-175b-run #<your_run_name>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export LOCAL_IMAGE_NAME=gcr.io/${PROJECT}/${USER}_runner
export OUTPUT_PATH=gs://v5p-demo/ #<your_GCS_folder_for_results>
```

## Run Maxtext GPT3-175B workloads on GKE

### Configs based on TPU type

#### v5p-1024

```
export TPU_TYPE=v5p-1024
export SCRIPT=MaxText/configs/v5p/gpt3_175b/v5p_1024.sh
```

#### v5p-2048

```
export TPU_TYPE=v5p-2048
export SCRIPT=MaxText/configs/v5p/gpt3_175b/v5p_2048.sh
```

#### v5p-3072

```
export TPU_TYPE=v5p-3072
export SCRIPT=MaxText/configs/v5p/gpt3_175b/v5p_3072.sh
```

#### v5p-4096

This will require a custom slice topology of 4x8x64
```
export TPU_TYPE=v5p-4096
export SCRIPT=MaxText/configs/v5p/gpt3_175b/v5p_4096.sh
```

#### v5p-8192

This will require a custom slice topology of 8x16x32
```
export TPU_TYPE=v5p-8192
export SCRIPT=MaxText/configs/v5p/gpt3_175b/v5p_8192.sh
```

#### v5p-12288

This will require a custom slice topology of 8x16x48
```
export TPU_TYPE=v5p-12288
export SCRIPT=MaxText/configs/v5p/gpt3_175b/v5p_12288.sh
```

### Starting workload

From the MaxText root directory, start your GPT3-175B workload

```
python3 ../xpk.py workload create \
--project ${PROJECT} \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=1 \
--base-docker-image=${LOCAL_IMAGE_NAME} \
--command "bash $SCRIPT $RUN_NAME $OUTPUT_PATH"
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 2, seconds: 22.197, TFLOP/s/device: 397.246, Tokens/s/device: 369.059, total_weights: 4194304, loss: 0.000
```

[Optional] If you need to delete your workload, you can run the following command:
```
cd .. # Switch back to the xpk directory
export WORKLOAD_NAME_TO_DELETE=gpt3-175b-test

python3 xpk.py workload delete \
--workload ${WORKLOAD_NAME_TO_DELETE} \
--cluster ${CLUSTER_NAME}
```
