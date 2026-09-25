# Instructions for training Stable Diffusion XL on TPU v5p

This document presents steps to run a StableDiffusion [MaxDiffusion](https://github.com/google/maxdiffusion/tree/main/src/maxdiffusion) workload through the [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit) (`gcluster`) tool.

Set up Cluster Toolkit (`gcluster` v1.104.0) and create a cluster following the [Cluster Toolkit Cloud TPU deployment guide](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview).

Build and push a docker image:

```bash
LOCAL_IMAGE_NAME=maxdiffusion_base_image
docker build --no-cache --network host -f ./docker/maxdiffusion.Dockerfile -t ${LOCAL_IMAGE_NAME} .
```

Run the workload using Cluster Toolkit (`./cluster_toolkit/run_recipe.sh`):

```bash
cd tpu-recipes/training/v5p/SDXL-MaxDiffusion/cluster_toolkit
export PROJECT_ID=$PROJECT
export CLUSTER_NAME=$CLUSTER_NAME
export ZONE=$ZONE
export BASE_OUTPUT_DIR=gs://output_bucket
export NUM_SLICES=1
export WORKLOAD_IMAGE=<YOUR_MAXDIFFUSION_RUNNER_IMAGE>
./run_recipe.sh
```
