# Instructions for training Llama2-7B-Maxtext on TPU v5p

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK

## Run Maxtext Llama2-7B workloads on GKE
1. Clone [Maxtext](https://github.com/AI-Hypercomputer/maxtext) repo
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
export WORKLOAD_NAME=llam2-7b-test #<your_workload_name>
export RUN_NAME=llama2-7b-run #<your_run_name>
export TPU_TYPE=v5p-512 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export LOCAL_IMAGE_NAME=gcr.io/${PROJECT}/${USER}_runner
export OUTPUT_PATH=gs://v5p-demo/ #<your_GCS_folder_for_results>
```

6. Switch back to your XPK folder and run Llama2-7B workload
```
cd ../ #Make sure you are in the XPK folder

python3 xpk.py workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--docker-image=${LOCAL_IMAGE_NAME} \
--command "\
   bash MaxText/configs/v5p/llama2_7b.sh RUN_NAME=$RUN_NAME OUTPUT_PATH=$OUTPUT_PATH"
```

7. [Optional] If you need to delete any of your workload, you can run the following command:
```
export WORKLOAD_NAME_TO_DELETE=llam2-7b-test

python3 xpk.py workload delete \
--workload ${WORKLOAD_NAME_TO_DELETE} \
--cluster ${CLUSTER_NAME}
```
