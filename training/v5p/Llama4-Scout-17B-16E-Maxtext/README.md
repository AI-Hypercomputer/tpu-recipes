# Instructions for training Llama4-Scout-17B-16E Maxtext on TPU v5p-256, v5p-512, and v5p-1024

> [!NOTE]
> A [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
> (`gcluster`) version of this recipe is available in
> [cluster_toolkit/README.md](cluster_toolkit/README.md).

This documents present steps to run Llama4-Scout-17B-16E [MaxText](https://github.com/google/maxtext) workload through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

## XPK setup

Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md) to create your GKE cluster with XPK.

## Prep for Maxtext

1. Clone [Maxtext](https://github.com/AI-Hypercomputer/maxtext) repo
```
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout 3eb77db3c94580f56f1b738f8d254b03bd205e35
```

2. Run the following commands to build the docker image
```
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable JAX_VERSION=0.7.0
```

3. Create your new GCS bucket

This is the GCS folder for storing test results. You can re-use any of your existing GCS buckets. To create a new bucket:
```
GCS_PATH=gs://v5p-demo #<your_GCS_folder_for_results>
gcloud storage buckets create ${GCS_PATH}  --project ${PROJECT}
```

4. Specify your workload enviroment variables

```
export PROJECT=#<your_compute_project>
export ZONE=#<your_compute_zone>
export CLUSTER_NAME=#<your_cluster_name>
export OUTPUT_DIR=gs://v5p-demo/ #<your_GCS_folder_for_results>
export DEVICE_TYPE=${DEVICE_TYPE} # v5p-256, v5p-512, or v5p-1024
```

## Run workloads

5. From the MaxText root directory, start your workload:
```
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=${DEVICE_TYPE} \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="llama4_scout_dropless_v5p_256" \
    --base_docker_image=maxtext_base_image
```

6. Check the training log

From your workload logs, you should see step time logs like the following, as training progresses:
```
completed step: 11, seconds: 31.652, TFLOP/s/device: 225.491, Tokens/s/device: 2070.487, total_weights: 33554432, loss: 11.825
```

7. Workload configuration

Workload configuration details can be found [here](https://github.com/AI-Hypercomputer/maxtext/blob/3eb77db3c94580f56f1b738f8d254b03bd205e35/benchmarks/maxtext_v5p_model_configs.py) in MaxText GitHub repo. Look for the configuration `llama4_scout_dropless_v5p_256`.

